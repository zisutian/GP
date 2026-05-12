from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch import nn
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel


DEFAULT_INTERNVL_PATH = Path(__file__).resolve().parents[1] / "InternVL" / "pretrained" / "OpenGVLab" / "InternVL2_5-1B"
DEFAULT_SIGLIP_PATH = "google/siglip-so400m-patch14-384"
DEFAULT_PALIGEMMA_PATH = (
    Path(__file__).resolve().parent
    / "pretrained"
    / "paligemma2-3b-mix-224"
)


def has_model_weights(model_path: Path) -> bool:
    return any(model_path.glob("*.safetensors")) or any(model_path.glob("*.bin")) or any(model_path.glob("*.pt"))


class SigLIPImageProjector(nn.Module):
    """VCoTGrasp-style linear projector: SigLIP hidden states -> LLM hidden space."""

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(vision_hidden_size, llm_hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear(image_features)


class SigLIPPrefixInternVLWrapper:
    """
    Inject SigLIP image embeddings directly into InternVL's language model.

    This follows the VCoTGrasp idea:
      image_encoder(pixel_values).last_hidden_state
        -> linear image_projector
        -> concat with text token embeddings
        -> language_model.generate(inputs_embeds=...)

    It intentionally does not use InternVL's <image> / IMG_CONTEXT token replacement path.
    """

    def __init__(
        self,
        internvl_path: str | Path = DEFAULT_INTERNVL_PATH,
        siglip_path: str | Path = DEFAULT_SIGLIP_PATH,
        paligemma_path: str | Path | None = DEFAULT_PALIGEMMA_PATH,
        projector_path: str | Path | None = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        use_flash_attn: bool = False,
        train_language_model: bool = False,
    ):
        self.internvl_path = Path(internvl_path)
        self.siglip_path = str(siglip_path)
        self.paligemma_path = Path(paligemma_path) if paligemma_path else None
        self.projector_path = Path(projector_path) if projector_path else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device.startswith("cuda") else torch.float32)
        self.use_flash_attn = use_flash_attn
        self.train_language_model = train_language_model

        self.tokenizer = None
        self.internvl_model = None
        self.language_model = None
        self.siglip_processor = None
        self.siglip_model = None
        self.image_projector = None

    def load(self):
        if not self.internvl_path.exists():
            raise FileNotFoundError(f"InternVL model path does not exist: {self.internvl_path}")
        if not has_model_weights(self.internvl_path):
            raise FileNotFoundError(f"No InternVL weight files found in {self.internvl_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.internvl_path, trust_remote_code=True, use_fast=False)
        self.internvl_model = AutoModel.from_pretrained(
            self.internvl_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=self.use_flash_attn,
            trust_remote_code=True,
        ).eval()
        self.language_model = self.internvl_model.language_model.to(self.device)
        self.language_model.train(self.train_language_model)
        for param in self.language_model.parameters():
            param.requires_grad = self.train_language_model

        self.siglip_processor, self.siglip_model = self._load_siglip()
        self.siglip_model = self.siglip_model.eval().to(self.device, dtype=self.dtype)
        for param in self.siglip_model.parameters():
            param.requires_grad = False

        vision_hidden_size = self.siglip_model.config.hidden_size
        llm_hidden_size = self.language_model.config.hidden_size
        self.image_projector = SigLIPImageProjector(vision_hidden_size, llm_hidden_size).to(self.device, dtype=self.dtype)
        if self.projector_path is not None:
            state_dict = torch.load(self.projector_path, map_location="cpu")
            self.image_projector.load_state_dict(state_dict)
        for param in self.image_projector.parameters():
            param.requires_grad = True
        self.image_projector.train()
        return self

    def _load_siglip(self) -> tuple[SiglipImageProcessor, SiglipVisionModel]:
        if self.paligemma_path is not None and self.paligemma_path.exists():
            processor = SiglipImageProcessor.from_pretrained(self.paligemma_path)
            config_data = json.loads((self.paligemma_path / "config.json").read_text())
            vision_config = SiglipVisionConfig(**config_data["vision_config"])
            model = SiglipVisionModel(vision_config)
            state_dict = self._load_paligemma_vision_state_dict(self.paligemma_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected SigLIP keys from PaliGemma checkpoint: {unexpected[:10]}")
            missing = [key for key in missing if not key.startswith("vision_model.head.")]
            if missing:
                raise RuntimeError(f"Missing SigLIP keys from PaliGemma checkpoint: {missing[:10]}")
            return processor, model

        processor = SiglipImageProcessor.from_pretrained(self.siglip_path)
        model = SiglipVisionModel.from_pretrained(self.siglip_path, torch_dtype=self.dtype)
        return processor, model

    @staticmethod
    def _load_paligemma_vision_state_dict(paligemma_path: Path) -> dict[str, torch.Tensor]:
        index_path = paligemma_path / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing PaliGemma safetensors index: {index_path}")
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
        vision_keys = [key for key in weight_map if key.startswith("vision_tower.")]
        shard_names = sorted({weight_map[key] for key in vision_keys})

        state_dict = {}
        for shard_name in shard_names:
            shard = load_file(paligemma_path / shard_name)
            for key in vision_keys:
                if weight_map[key] != shard_name:
                    continue
                state_dict[key.removeprefix("vision_tower.")] = shard[key]
        return state_dict

    def trainable_parameters(self):
        if self.image_projector is not None:
            yield from self.image_projector.parameters()
        if self.train_language_model and self.language_model is not None:
            yield from self.language_model.parameters()

    def encode_image(self, image: str | Path | Image.Image) -> torch.Tensor:
        if self.siglip_model is None or self.siglip_processor is None or self.image_projector is None:
            self.load()
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            pil_image = Image.open(image).convert("RGB")

        inputs = self.siglip_processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        with torch.no_grad():
            image_outputs = self.siglip_model(pixel_values=pixel_values)
        image_features = image_outputs.last_hidden_state
        image_embeds = self.image_projector(image_features)
        image_embeds = image_embeds / (self.language_model.config.hidden_size**0.5)
        return image_embeds

    def format_prompt(self, question: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": question}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return question

    def build_inputs_embeds(self, question: str, image: str | Path | Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.language_model is None or self.tokenizer is None:
            self.load()
        image_embeds = self.encode_image(image)
        prompt = self.format_prompt(question)
        text_inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs["attention_mask"].to(self.device)
        text_embeds = self.language_model.get_input_embeddings()(input_ids).to(dtype=self.dtype)

        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        image_attention_mask = torch.ones(image_embeds.shape[:2], dtype=attention_mask.dtype, device=self.device)
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        return inputs_embeds, attention_mask

    @torch.inference_mode()
    def generate(
        self,
        question: str,
        image: str | Path | Image.Image,
        max_new_tokens: int = 128,
        do_sample: bool = False,
    ) -> str:
        if self.language_model is None or self.tokenizer is None:
            self.load()
        inputs_embeds, attention_mask = self.build_inputs_embeds(question, image)
        output_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
