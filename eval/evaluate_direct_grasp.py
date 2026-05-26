from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNVL_CHAT_ROOT = REPO_ROOT / "InternVL/internvl_chat"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(INTERNVL_CHAT_ROOT))

import cv2
import torch
from internvl.conversation import get_conv_template
from internvl.model import split_model
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from data_tools.vcot_direct_lmdb import build_direct_grasp_item
from PIL import Image
from safetensors.torch import safe_open
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from grasp_settings import build_settings


LOC_RE = re.compile(r"<loc(\d{4})>")
VCOT_IOU_THRESHOLD = 0.25
VCOT_ANGLE_THRESHOLD = 30.0
SETTINGS = build_settings()
DEFAULT_DATA_INDEX_ROOT = Path(SETTINGS["GRASP_DIRECT_INDEX_ROOT"])
DEFAULT_OUT_DIR = Path(SETTINGS["GRASP_DIRECT_RESULT_ROOT"])
DEFAULT_DATASETS = {
    "test_seen": "test_seen.jsonl",
    "test_unseen": "test_unseen.jsonl",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate direct grasp generation on Grasp-Anything data indexes.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", default="test_seen,test_unseen")
    parser.add_argument(
        "--data-index-root",
        default=None,
        help="Override the checkpoint config data_index_root.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-num", type=int, default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vcot-iou-threshold", type=float, default=VCOT_IOU_THRESHOLD)
    parser.add_argument("--vcot-angle-threshold", type=float, default=VCOT_ANGLE_THRESHOLD)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true")
    return parser.parse_args()


def has_lora_modules_to_save(checkpoint: str | Path) -> bool:
    model_path = Path(checkpoint) / "model.safetensors"
    if not model_path.exists():
        return False
    with safe_open(str(model_path), framework="pt", device="cpu") as f:
        return any("modules_to_save" in key for key in f.keys())


def load_model_and_tokenizer(args):
    config = InternVLChatConfig.from_pretrained(args.checkpoint)
    if has_lora_modules_to_save(args.checkpoint):
        config.llm_lora_modules_to_save = ["embed_tokens", "lm_head"]

    kwargs = {}
    if args.auto:
        kwargs["device_map"] = split_model(config.llm_config.num_hidden_layers)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        **kwargs,
    ).eval()
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        model = model.cuda()
    return model, tokenizer


def find_vcot_config(checkpoint: str | Path) -> Path:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    candidates = []
    if checkpoint_path.is_dir():
        candidates.append(checkpoint_path / "vcot_config.json")
        candidates.append(checkpoint_path.parent / "vcot_config.json")
    else:
        candidates.append(checkpoint_path.parent / "vcot_config.json")
        candidates.append(checkpoint_path.parent.parent / "vcot_config.json")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"vcot_config.json not found for checkpoint {checkpoint_path}. "
        "Expected it in the checkpoint directory or its parent."
    )


def decode_loc_tokens(text: str, bins: int = 1024) -> list[float] | None:
    matches = LOC_RE.findall(text)
    if len(matches) < 5:
        return None
    values = [max(0, min(bins - 1, int(token))) / float(bins - 1) for token in matches[:5]]
    return values


def denormalize_grasp(values: list[float], image_size: int = 416) -> list[float]:
    return [
        values[0] * image_size,
        values[1] * image_size,
        values[2] * image_size,
        values[3] * image_size,
        values[4] * 180.0,
    ]


def denormalize_grasp_like_vcot(values: list[float], image_size: int = 416) -> list[float]:
    return [int(value * image_size) for value in values[:4]] + [float(values[4] * 180.0)]


def angle_diff_180(angle1: float, angle2: float) -> float:
    diff = abs(float(angle1) - float(angle2))
    return min(diff, 180.0 - diff)


def rotated_rect_iou(grasp1: list[float], grasp2: list[float]) -> float:
    rect1 = ((grasp1[0], grasp1[1]), (grasp1[2], grasp1[3]), grasp1[4])
    rect2 = ((grasp2[0], grasp2[1]), (grasp2[2], grasp2[3]), grasp2[4])
    box1 = cv2.boxPoints(rect1)
    box2 = cv2.boxPoints(rect2)
    intersection, _ = cv2.intersectConvexConvex(box1, box2)
    area1 = grasp1[2] * grasp1[3]
    area2 = grasp2[2] * grasp2[3]
    union = area1 + area2 - intersection
    return float(intersection / union) if union > 0 else 0.0


def vcot_grasp_success(
    pred: list[float],
    labels: list[list[float]],
    iou_threshold: float,
    angle_threshold: float,
) -> tuple[bool, float, float, float, float]:
    best_iou = 0.0
    best_angle_diff = 180.0
    best_joint_iou = 0.0
    best_joint_angle_diff = 180.0
    for label in labels:
        iou = rotated_rect_iou(pred, label)
        angle_diff = angle_diff_180(pred[4], label[4])
        if iou > best_iou:
            best_iou = iou
        if angle_diff < best_angle_diff:
            best_angle_diff = angle_diff
        if iou > best_joint_iou:
            best_joint_iou = iou
            best_joint_angle_diff = angle_diff
        if iou >= iou_threshold and angle_diff <= angle_threshold:
            return True, iou, angle_diff, best_iou, best_angle_diff
    return False, best_joint_iou, best_joint_angle_diff, best_iou, best_angle_diff


@torch.inference_mode()
def chat_preserve_loc_tokens(model, tokenizer, pixel_values, question, generation_config, num_patches_list):
    if pixel_values is not None and "<image>" not in question:
        question = "<image>\n" + question

    img_context_token = "<IMG_CONTEXT>"
    img_context_token_id = tokenizer.convert_tokens_to_ids(img_context_token)
    model.img_context_token_id = img_context_token_id

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    for num_patches in num_patches_list:
        image_tokens = "<img>" + img_context_token * model.num_image_token * num_patches + "</img>"
        query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    device = torch.device(model.language_model.device if torch.cuda.is_available() else "cpu")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    generation_config["eos_token_id"] = eos_token_id
    generation_config["pad_token_id"] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    generation_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config,
    )
    response = tokenizer.batch_decode(generation_output, skip_special_tokens=False)[0]
    return response.split(template.sep.strip())[0].strip()


class DirectGraspDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        prompt: str = "grasp the {obj_name}",
        image_size: int = 448,
        vcot_image_size: int = 416,
        dynamic_image_size: bool = True,
        use_thumbnail: bool = True,
        max_num: int = 6,
        limit: int | None = None,
    ):
        self.records = []
        with Path(manifest).open("r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))
                if limit is not None and len(self.records) >= limit:
                    break
        self.prompt = prompt
        self.vcot_image_size = vcot_image_size
        self.image_size = image_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        item = build_direct_grasp_item(record, image_size=self.vcot_image_size, include_all_grasps=True)
        image: Image.Image = item["image"]
        images = dynamic_preprocess(
            image,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
            max_num=self.max_num,
        ) if self.dynamic_image_size else [image]
        pixel_values = torch.stack([self.transform(image) for image in images])
        target = decode_loc_tokens(item["conversations"][1]["value"])
        if target is None:
            raise ValueError(f"Could not decode target loc tokens for record {idx}")
        question = self.prompt.format(obj_name=record["obj_name"])
        return {
            "pixel_values": pixel_values,
            "question": question,
            "num_patches": len(images),
            "target": torch.tensor(target, dtype=torch.float32),
            "target_labels": item["target_labels"],
            "record": record,
        }


def collate_fn(batch):
    assert len(batch) == 1, "Only batch size 1 is supported for variable dynamic image patches."
    item = batch[0]
    return item["pixel_values"], item["question"], item["num_patches"], item["target"], item["target_labels"], item["record"]


class InferenceSampler(torch.utils.data.Sampler):
    def __init__(self, size: int):
        self.size = int(size)
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        shard = self.size // self.world_size
        left = self.size % self.world_size
        sizes = [shard + int(rank < left) for rank in range(self.world_size)]
        self.begin = sum(sizes[:self.rank])
        self.end = min(sum(sizes[: self.rank + 1]), self.size)

    def __iter__(self):
        yield from range(self.begin, self.end)

    def __len__(self):
        return self.end - self.begin


def summarize(
    outputs,
    vcot_image_size: int = 416,
    vcot_iou_threshold: float = VCOT_IOU_THRESHOLD,
    vcot_angle_threshold: float = VCOT_ANGLE_THRESHOLD,
):
    valid = [out for out in outputs if out["pred_norm"] is not None]
    total = len(outputs)
    parsed = len(valid)
    if not valid:
        return {"total": total, "parsed": parsed, "parse_rate": 0.0}

    pred = torch.tensor([out["pred_norm"] for out in valid], dtype=torch.float32)
    target = torch.tensor([out["target_norm"] for out in valid], dtype=torch.float32)
    abs_norm = (pred - target).abs()
    pred_px = torch.tensor([denormalize_grasp(out["pred_norm"], vcot_image_size) for out in valid])
    target_px = torch.tensor([denormalize_grasp(out["target_norm"], vcot_image_size) for out in valid])
    abs_px = (pred_px - target_px).abs()

    xy_dist = torch.linalg.vector_norm(pred_px[:, :2] - target_px[:, :2], dim=1)
    wh_dist = torch.linalg.vector_norm(pred_px[:, 2:4] - target_px[:, 2:4], dim=1)
    angle_err_raw = abs_px[:, 4]
    angle_err_circular = torch.minimum(angle_err_raw, 180.0 - angle_err_raw)
    strict = (xy_dist <= 10) & (wh_dist <= 10) & (angle_err_circular <= 10)
    loose = (xy_dist <= 20) & (wh_dist <= 20) & (angle_err_circular <= 20)

    vcot_success = 0
    vcot_ious = []
    vcot_angle_diffs = []
    vcot_best_ious = []
    vcot_best_angle_diffs = []
    vcot_top1_success = 0
    target_label_counts = []
    for out in valid:
        pred_grasp = denormalize_grasp_like_vcot(out["pred_norm"], vcot_image_size)
        target_labels = out.get("target_labels", [])
        target_label_counts.append(len(target_labels))
        success, joint_iou, joint_angle_diff, best_iou, best_angle_diff = vcot_grasp_success(
            pred_grasp,
            target_labels,
            vcot_iou_threshold,
            vcot_angle_threshold,
        )
        top1_success = False
        if target_labels:
            top1_iou = rotated_rect_iou(pred_grasp, target_labels[0])
            top1_angle_diff = angle_diff_180(pred_grasp[4], target_labels[0][4])
            top1_success = top1_iou >= vcot_iou_threshold and top1_angle_diff <= vcot_angle_threshold
        else:
            top1_iou = 0.0
            top1_angle_diff = 180.0
        out["vcot_success"] = success
        out["vcot_top1_success"] = top1_success
        out["vcot_top1_iou"] = top1_iou
        out["vcot_top1_angle_diff"] = top1_angle_diff
        out["vcot_joint_iou"] = joint_iou
        out["vcot_joint_angle_diff"] = joint_angle_diff
        out["vcot_best_iou"] = best_iou
        out["vcot_best_angle_diff"] = best_angle_diff
        out["target_label_count"] = len(target_labels)
        vcot_success += int(success)
        vcot_top1_success += int(top1_success)
        vcot_ious.append(joint_iou)
        vcot_angle_diffs.append(joint_angle_diff)
        vcot_best_ious.append(best_iou)
        vcot_best_angle_diffs.append(best_angle_diff)

    return {
        "total": total,
        "parsed": parsed,
        "parse_rate": parsed / total,
        "mean_abs_norm": abs_norm.mean(0).tolist(),
        "mean_abs_pixel_angle": abs_px.mean(0).tolist(),
        "xy_dist_mean": xy_dist.mean().item(),
        "wh_dist_mean": wh_dist.mean().item(),
        "angle_err_raw_mean": angle_err_raw.mean().item(),
        "angle_err_circular_mean": angle_err_circular.mean().item(),
        "strict_10px_10deg": strict.float().mean().item(),
        "loose_20px_20deg": loose.float().mean().item(),
        "vcot_iou_threshold": vcot_iou_threshold,
        "vcot_angle_threshold": vcot_angle_threshold,
        "vcot_success": vcot_success,
        "vcot_success_rate_all": vcot_success / total,
        "vcot_success_rate_valid": vcot_success / parsed,
        "vcot_top1_success": vcot_top1_success,
        "vcot_top1_success_rate_all": vcot_top1_success / total,
        "vcot_top1_success_rate_valid": vcot_top1_success / parsed,
        "target_label_count_mean": sum(target_label_counts) / len(target_label_counts),
        "target_label_count_max": max(target_label_counts),
        "vcot_joint_iou_mean": sum(vcot_ious) / len(vcot_ious),
        "vcot_joint_angle_diff_mean": sum(vcot_angle_diffs) / len(vcot_angle_diffs),
        "vcot_best_iou_mean": sum(vcot_best_ious) / len(vcot_best_ious),
        "vcot_best_angle_diff_mean": sum(vcot_best_angle_diffs) / len(vcot_best_angle_diffs),
    }


def evaluate_dataset(args, model, tokenizer, name: str, manifest: Path, image_size: int, use_thumbnail: bool):
    dataset = DirectGraspDataset(
        manifest=manifest,
        image_size=image_size,
        dynamic_image_size=True,
        use_thumbnail=use_thumbnail,
        max_num=args.max_num,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    outputs = []
    for pixel_values, question, num_patches, target, target_labels, record in tqdm(loader, disable=rank() != 0):
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        generation_config = {
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": 1,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            generation_config["temperature"] = args.temperature
        answer = chat_preserve_loc_tokens(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
            num_patches_list=[num_patches],
        )
        pred = decode_loc_tokens(answer)
        outputs.append({
            "answer": answer,
            "pred_norm": pred,
            "target_norm": target.tolist(),
            "target_labels": target_labels,
            "grasp_id": record.get("grasp_id"),
            "obj_name": record.get("obj_name"),
            "split": record.get("split"),
        })

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        gathered = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, outputs)
        outputs = list(itertools.chain.from_iterable(gathered))

    if rank() == 0:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%y%m%d%H%M%S", time.localtime())
        output_path = Path(args.out_dir) / f"{name}_{stamp}.json"
        summary = summarize(
            outputs,
            vcot_iou_threshold=args.vcot_iou_threshold,
            vcot_angle_threshold=args.vcot_angle_threshold,
        )
        summary.update({
            "evaluation_mode": "direct_grasp",
            "dataset_name": name,
            "split": name,
            "data_index": str(manifest.resolve()),
            "data_index_root": str(Path(args.data_index_root).resolve()),
            "checkpoint": args.checkpoint,
            "loaded_vcot_config": args.loaded_vcot_config,
        })
        for output in outputs:
            output.pop("target_labels", None)
        output_path.write_text(json.dumps({"summary": summary, "outputs": outputs}, indent=2), encoding="utf-8")
        print(f"{name}: {json.dumps(summary, ensure_ascii=False)}")
        print(f"saved: {output_path}")


def rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def main():
    args = parse_args()
    assert args.batch_size == 1, "Only batch size 1 is supported"
    args.checkpoint = str(Path(args.checkpoint).resolve())
    config_path = find_vcot_config(args.checkpoint)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args.loaded_vcot_config = str(config_path)
    if args.data_index_root is None:
        if not config.get("data_index_root"):
            raise ValueError(f"{config_path} is missing data_index_root.")
        args.data_index_root = str(Path(config["data_index_root"]).expanduser().resolve())
    else:
        args.data_index_root = str(Path(args.data_index_root).expanduser().resolve())
    if args.max_num is None:
        args.max_num = int(config.get("max_dynamic_patch") or 6)
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=int(os.getenv("WORLD_SIZE", "1")),
            rank=int(os.getenv("RANK", "0")),
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))

    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    if rank() == 0:
        print(f"checkpoint: {args.checkpoint}")
        print(f"image_size: {image_size}, use_thumbnail: {use_thumbnail}, max_num: {args.max_num}")

    data_index_root = Path(args.data_index_root)
    for dataset_name in args.datasets.split(","):
        dataset_name = dataset_name.strip()
        if not dataset_name:
            continue
        data_index = Path(DEFAULT_DATASETS.get(dataset_name, dataset_name))
        if not data_index.is_absolute():
            data_index = data_index_root / data_index
        evaluate_dataset(args, model, tokenizer, dataset_name, data_index, image_size, use_thumbnail)


if __name__ == "__main__":
    main()
