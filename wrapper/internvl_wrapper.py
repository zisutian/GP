from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "InternVL" / "pretrained" / "OpenGVLab" / "InternVL2_5-1B"


def build_transform(input_size: int = 448) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 6, image_size: int = 448, use_thumbnail: bool = True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def has_model_weights(model_path: Path) -> bool:
    return any(model_path.glob("*.safetensors")) or any(model_path.glob("*.bin")) or any(model_path.glob("*.pt"))


class NativeInternVLWrapper:
    """Native InternVL wrapper using the model's built-in <image> token path."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_dynamic_patch: int = 6,
        use_flash_attn: Optional[bool] = None,
    ):
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device.startswith("cuda") else torch.float32)
        self.max_dynamic_patch = max_dynamic_patch
        self.use_flash_attn = False if use_flash_attn is None else use_flash_attn
        self.tokenizer = None
        self.model = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        if not has_model_weights(self.model_path):
            raise FileNotFoundError(
                f"No weight files found in {self.model_path}. Run:\n"
                f"  cd {self.model_path.parents[2]}\n"
                f"  huggingface-cli download OpenGVLab/InternVL2_5-1B "
                f"--local-dir pretrained/OpenGVLab/InternVL2_5-1B --local-dir-use-symlinks False"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=self.use_flash_attn,
            trust_remote_code=True,
        ).eval()
        self.model.to(self.device)
        return self

    def load_image(self, image_path: str | Path, input_size: int = 448) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(
            image,
            image_size=input_size,
            use_thumbnail=True,
            max_num=self.max_dynamic_patch,
        )
        pixel_values = torch.stack([transform(img) for img in images])
        return pixel_values.to(self.dtype).to(self.device)

    @torch.inference_mode()
    def chat(
        self,
        question: str,
        image_path: str | Path | None = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()
        pixel_values = None
        prompt = question
        if image_path is not None:
            pixel_values = self.load_image(image_path)
            if not prompt.lstrip().startswith("<image>"):
                prompt = "<image>\n" + prompt
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        return self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
