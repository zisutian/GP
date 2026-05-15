from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parent
INTERNVL_CHAT_ROOT = REPO_ROOT / "InternVL/internvl_chat"
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(INTERNVL_CHAT_ROOT))

import torch
from data_tools.vcot_bbox_lmdb import normalize_bbox_xyxy
from data_tools.vcot_crop_lmdb import (
    DEFAULT_BBOX_EDGE_EXPAND,
    DEFAULT_MIN_BBOX_HALF_SIZE,
    TARGET_FRAME_CROP_IMAGE,
    TARGET_FRAME_FULL_IMAGE,
    _load_mask,
    crop_box_from_bbox,
    mask_to_bbox_position,
    transform_grasp_from_crop_norm,
    transform_grasp_to_crop_norm,
)
from data_tools.vcot_direct_lmdb import _load_grasps, _load_image, _normalize_grasp
from evaluate_direct_grasp import (
    LOC_RE,
    VCOT_ANGLE_THRESHOLD,
    VCOT_IOU_THRESHOLD,
    InferenceSampler,
    angle_diff_180,
    chat_preserve_loc_tokens,
    decode_loc_tokens,
    denormalize_grasp,
    denormalize_grasp_like_vcot,
    load_model_and_tokenizer,
    rank,
    rotated_rect_iou,
    vcot_grasp_success,
)
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_DATASETS = {
    "test_seen": "data/vcot_grasp/vcot/test_seen.jsonl",
    "test_unseen": "data/vcot_grasp/vcot/test_unseen.jsonl",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predicted-bbox VCoT-style grasp pipeline.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", default="test_seen,test_unseen")
    parser.add_argument("--manifest-root", default=".")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--bbox-max-new-tokens", type=int, default=24)
    parser.add_argument("--grasp-max-new-tokens", type=int, default=32)
    parser.add_argument("--max-num", type=int, default=None)
    parser.add_argument("--vcot-image-size", type=int, default=416)
    parser.add_argument("--bbox-edge-expand", type=int, default=None)
    parser.add_argument("--min-bbox-half-size", type=int, default=None)
    parser.add_argument("--target-grasp-index", type=int, default=None)
    parser.add_argument(
        "--target-coordinate-frame",
        choices=[TARGET_FRAME_FULL_IMAGE, TARGET_FRAME_CROP_IMAGE],
        default=None,
        help="Stage-2 grasp output frame. Defaults to the checkpoint vcot_config.json when available.",
    )
    parser.add_argument(
        "--vcot-config",
        default=None,
        help="Optional explicit VCoT config JSON. Otherwise eval searches the checkpoint and its parent.",
    )
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "result/vcot_grasp_vcot"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vcot-iou-threshold", type=float, default=VCOT_IOU_THRESHOLD)
    parser.add_argument("--vcot-angle-threshold", type=float, default=VCOT_ANGLE_THRESHOLD)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true")
    return parser.parse_args()


def decode_loc_tokens_n(text: str, expected: int, bins: int = 1024) -> list[float] | None:
    matches = LOC_RE.findall(text)
    if len(matches) < expected:
        return None
    return [max(0, min(bins - 1, int(token))) / float(bins - 1) for token in matches[:expected]]


def normalize_pred_bbox(values: list[float]) -> list[float]:
    x0, y0, x1, y1 = [max(0.0, min(1.0, float(value))) for value in values[:4]]
    x_min, x_max = sorted([x0, x1])
    y_min, y_max = sorted([y0, y1])
    return [x_min, y_min, x_max, y_max]


def denormalize_bbox(values: list[float], image_size: int = 416) -> list[int]:
    x0, y0, x1, y1 = normalize_pred_bbox(values)
    return [
        int(x0 * image_size),
        int(y0 * image_size),
        int(x1 * image_size),
        int(y1 * image_size),
    ]


def normalize_full_grasp(values: list[float], image_size: int = 416) -> list[float]:
    x, y, w, h, angle = [float(value) for value in values[:5]]
    return [
        max(0.0, min(1.0, x / image_size)),
        max(0.0, min(1.0, y / image_size)),
        max(0.0, min(1.0, w / image_size)),
        max(0.0, min(1.0, h / image_size)),
        max(0.0, min(1.0, angle / 180.0)),
    ]


def _first_present(config: dict, *keys, default=None):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _required_config_value(args, config: dict, config_path: Path, attr: str, *keys, cast=None):
    explicit_value = getattr(args, attr)
    if explicit_value is not None:
        return explicit_value

    value = _first_present(config, *keys, default=None)
    if value is None:
        option = "--" + attr.replace("_", "-")
        source = str(config_path)
        raise ValueError(
            f"{option} is not set and no matching value was found in {source}. "
            f"Write vcot_config.json for this experiment or pass {option} explicitly."
        )
    return cast(value) if cast is not None else value


def find_vcot_config(checkpoint: str | Path, explicit_config: str | None = None) -> Path:
    if explicit_config:
        path = Path(explicit_config).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"VCoT config not found: {path}")
        return path

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


def apply_vcot_config(args):
    config_path = find_vcot_config(args.checkpoint, args.vcot_config)
    config = {}
    if config_path is not None:
        config = json.loads(config_path.read_text(encoding="utf-8"))

    args.target_coordinate_frame = _required_config_value(
        args,
        config,
        config_path,
        "target_coordinate_frame",
        "target_coordinate_frame",
    )
    args.bbox_edge_expand = _required_config_value(
        args,
        config,
        config_path,
        "bbox_edge_expand",
        "bbox_edge_expand",
        cast=int,
    )
    args.min_bbox_half_size = _required_config_value(
        args,
        config,
        config_path,
        "min_bbox_half_size",
        "min_bbox_half_size",
        cast=int,
    )
    args.target_grasp_index = _required_config_value(
        args,
        config,
        config_path,
        "target_grasp_index",
        "target_grasp_index",
        cast=int,
    )
    if args.max_num is None:
        args.max_num = int(config.get("max_dynamic_patch") or 6)

    if args.target_coordinate_frame not in {TARGET_FRAME_FULL_IMAGE, TARGET_FRAME_CROP_IMAGE}:
        raise ValueError(
            f"Unsupported target_coordinate_frame={args.target_coordinate_frame}; "
            f"expected {TARGET_FRAME_FULL_IMAGE} or {TARGET_FRAME_CROP_IMAGE}"
        )
    args.loaded_vcot_config = str(config_path)
    return args


def bbox_iou_xyxy(box_a: list[int] | tuple[int, int, int, int], box_b: list[int] | tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in box_a]
    bx0, by0, bx1, by1 = [float(value) for value in box_b]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def scale_box_for_image(crop_box: list[int], label_image_size: int, image_size: tuple[int, int]) -> list[int]:
    image_width, image_height = int(image_size[0]), int(image_size[1])
    if image_width == label_image_size and image_height == label_image_size:
        return [int(value) for value in crop_box]

    x_scale = image_width / float(label_image_size)
    y_scale = image_height / float(label_image_size)
    x0, y0, x1, y1 = crop_box
    return [
        max(0, min(image_width, math.floor(x0 * x_scale))),
        max(0, min(image_height, math.floor(y0 * y_scale))),
        max(0, min(image_width, math.ceil(x1 * x_scale))),
        max(0, min(image_height, math.ceil(y1 * y_scale))),
    ]


class VCotPipelineDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 448,
        vcot_image_size: int = 416,
        dynamic_image_size: bool = True,
        use_thumbnail: bool = True,
        max_num: int = 6,
        bbox_edge_expand: int = DEFAULT_BBOX_EDGE_EXPAND,
        min_bbox_half_size: int = DEFAULT_MIN_BBOX_HALF_SIZE,
        target_grasp_index: int = 0,
        limit: int | None = None,
    ):
        self.records = []
        with Path(manifest).open("r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))
                if limit is not None and len(self.records) >= limit:
                    break
        self.vcot_image_size = vcot_image_size
        self.image_size = image_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.bbox_edge_expand = bbox_edge_expand
        self.min_bbox_half_size = min_bbox_half_size
        self.target_grasp_index = target_grasp_index
        self.transform = build_transform(is_train=False, input_size=image_size)

    def __len__(self):
        return len(self.records)

    def preprocess_images(self, source_images: list[Image.Image]) -> tuple[torch.Tensor, list[int]]:
        patch_images = []
        num_patches = []
        per_image_max_num = max(1, self.max_num // len(source_images))
        for image in source_images:
            images = (
                dynamic_preprocess(
                    image,
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                    max_num=per_image_max_num,
                )
                if self.dynamic_image_size
                else [image]
            )
            patch_images.extend(images)
            num_patches.append(len(images))
        return torch.stack([self.transform(image) for image in patch_images]), num_patches

    def preprocess_full_image(self, image: Image.Image) -> tuple[torch.Tensor, int]:
        images = (
            dynamic_preprocess(
                image,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
                max_num=self.max_num,
            )
            if self.dynamic_image_size
            else [image]
        )
        return torch.stack([self.transform(image) for image in images]), len(images)

    def __getitem__(self, idx):
        record = self.records[idx]
        source_root = Path(record["source_root"])
        full_image = _load_image(source_root / "lmdb/image", record["image_key"])
        grasp_key = record["grasp_key"]
        target_labels = _load_grasps(source_root / "lmdb/grasp_label_positive", grasp_key)
        target_index = max(0, min(len(target_labels) - 1, int(self.target_grasp_index)))
        target_grasp = target_labels[target_index]
        target_norm = _normalize_grasp(target_grasp, self.vcot_image_size)

        mask_key = record["mask_key"]
        mask = _load_mask(source_root / "lmdb/mask", mask_key)
        gt_bbox = mask_to_bbox_position(mask)
        if gt_bbox is None:
            raise ValueError(f"Empty mask for record: {record.get('grasp_id')}")
        gt_crop_box = crop_box_from_bbox(
            gt_bbox,
            image_size=self.vcot_image_size,
            min_half_size=self.min_bbox_half_size,
            edge_expand=self.bbox_edge_expand,
        )

        full_pixel_values, full_num_patches = self.preprocess_full_image(full_image)
        return {
            "full_image": full_image,
            "full_pixel_values": full_pixel_values,
            "full_num_patches": full_num_patches,
            "target_norm": torch.tensor(target_norm, dtype=torch.float32),
            "target_grasp": target_grasp,
            "target_labels": target_labels,
            "gt_bbox": gt_bbox,
            "gt_bbox_norm": normalize_bbox_xyxy(gt_bbox, self.vcot_image_size),
            "gt_crop_box": gt_crop_box,
            "record": record,
        }


def collate_fn(batch):
    assert len(batch) == 1, "Only batch size 1 is supported for variable dynamic image patches."
    return batch[0]


def preprocess_stage2(
    dataset: VCotPipelineDataset,
    full_image: Image.Image,
    crop_box: list[int],
) -> tuple[torch.Tensor, list[int], Image.Image]:
    pil_crop_box = scale_box_for_image(crop_box, dataset.vcot_image_size, full_image.size)
    crop_image = full_image.crop(tuple(pil_crop_box))
    pixel_values, num_patches = dataset.preprocess_images([full_image, crop_image])
    return pixel_values, num_patches, crop_image


def generation_config(args, max_new_tokens: int) -> dict:
    config = {
        "num_beams": args.num_beams,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 1,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        config["temperature"] = args.temperature
    return config


def summarize(
    outputs,
    vcot_image_size: int = 416,
    vcot_iou_threshold: float = VCOT_IOU_THRESHOLD,
    vcot_angle_threshold: float = VCOT_ANGLE_THRESHOLD,
):
    valid = [out for out in outputs if out["pred_norm"] is not None]
    total = len(outputs)
    parsed = len(valid)

    bbox_valid = [out for out in outputs if out["pred_bbox_norm"] is not None]
    bbox_ious = [out["pred_bbox_iou"] for out in bbox_valid if out.get("pred_bbox_iou") is not None]
    summary = {
        "total": total,
        "bbox_parsed": len(bbox_valid),
        "bbox_parse_rate": len(bbox_valid) / total if total else 0.0,
        "bbox_iou_mean": sum(bbox_ious) / len(bbox_ious) if bbox_ious else 0.0,
        "bbox_iou_025": sum(iou >= 0.25 for iou in bbox_ious) / len(bbox_ious) if bbox_ious else 0.0,
        "bbox_iou_050": sum(iou >= 0.50 for iou in bbox_ious) / len(bbox_ious) if bbox_ious else 0.0,
        "parsed": parsed,
        "parse_rate": parsed / total if total else 0.0,
    }
    if not valid:
        return summary

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
        if target_labels:
            top1_iou = rotated_rect_iou(pred_grasp, target_labels[0])
            top1_angle_diff = angle_diff_180(pred_grasp[4], target_labels[0][4])
            top1_success = top1_iou >= vcot_iou_threshold and top1_angle_diff <= vcot_angle_threshold
        else:
            top1_iou = 0.0
            top1_angle_diff = 180.0
            top1_success = False
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

    summary.update({
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
    })
    return summary


def evaluate_dataset(args, model, tokenizer, name: str, manifest: Path, image_size: int, use_thumbnail: bool):
    dataset = VCotPipelineDataset(
        manifest=manifest,
        image_size=image_size,
        vcot_image_size=args.vcot_image_size,
        dynamic_image_size=True,
        use_thumbnail=use_thumbnail,
        max_num=args.max_num,
        bbox_edge_expand=args.bbox_edge_expand,
        min_bbox_half_size=args.min_bbox_half_size,
        target_grasp_index=args.target_grasp_index,
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
    for item in tqdm(loader, disable=rank() != 0):
        record = item["record"]
        full_image = item["full_image"]

        full_pixel_values = item["full_pixel_values"].to(torch.bfloat16).cuda()
        bbox_question = f"detect {record['obj_name']}"
        bbox_answer = chat_preserve_loc_tokens(
            model=model,
            tokenizer=tokenizer,
            pixel_values=full_pixel_values,
            question=bbox_question,
            generation_config=generation_config(args, args.bbox_max_new_tokens),
            num_patches_list=[item["full_num_patches"]],
        )
        pred_bbox_norm_raw = decode_loc_tokens_n(bbox_answer, expected=4)
        pred_bbox_norm = normalize_pred_bbox(pred_bbox_norm_raw) if pred_bbox_norm_raw is not None else None

        grasp_answer = None
        pred_model_grasp = None
        pred_grasp = None
        target_model_norm = None
        pred_bbox_xyxy = None
        pred_crop_box = None
        pred_bbox_iou = None
        pred_crop_image_size = None
        if pred_bbox_norm is not None:
            pred_bbox_xyxy = denormalize_bbox(pred_bbox_norm, args.vcot_image_size)
            pred_bbox_iou = bbox_iou_xyxy(pred_bbox_xyxy, item["gt_bbox"])
            pred_crop_box = crop_box_from_bbox(
                pred_bbox_norm,
                image_size=args.vcot_image_size,
                min_half_size=args.min_bbox_half_size,
                edge_expand=args.bbox_edge_expand,
            )
            stage2_pixel_values, stage2_num_patches, pred_crop_image = preprocess_stage2(
                dataset,
                full_image,
                pred_crop_box,
            )
            pred_crop_image_size = list(pred_crop_image.size)
            stage2_pixel_values = stage2_pixel_values.to(torch.bfloat16).cuda()
            grasp_question = f"<image>\n<image>\ngrasp the {record['obj_name']}"
            grasp_answer = chat_preserve_loc_tokens(
                model=model,
                tokenizer=tokenizer,
                pixel_values=stage2_pixel_values,
                question=grasp_question,
                generation_config=generation_config(args, args.grasp_max_new_tokens),
                num_patches_list=stage2_num_patches,
            )
            pred_model_grasp = decode_loc_tokens(grasp_answer)
            if pred_model_grasp is not None:
                if args.target_coordinate_frame == TARGET_FRAME_CROP_IMAGE:
                    pred_full_grasp = transform_grasp_from_crop_norm(pred_model_grasp, pred_crop_box)
                    pred_grasp = normalize_full_grasp(pred_full_grasp, args.vcot_image_size)
                    target_model_norm = transform_grasp_to_crop_norm(item["target_grasp"], pred_crop_box)
                else:
                    pred_grasp = pred_model_grasp
                    target_model_norm = item["target_norm"].tolist()

        outputs.append({
            "bbox_answer": bbox_answer,
            "pred_bbox_norm": pred_bbox_norm,
            "pred_bbox_xyxy": pred_bbox_xyxy,
            "gt_bbox_norm": item["gt_bbox_norm"],
            "gt_bbox_xyxy": item["gt_bbox"],
            "pred_bbox_iou": pred_bbox_iou,
            "pred_crop_box": pred_crop_box,
            "gt_crop_box": item["gt_crop_box"],
            "pred_crop_image_size": pred_crop_image_size,
            "answer": grasp_answer,
            "pred_norm": pred_grasp,
            "pred_model_norm": pred_model_grasp,
            "target_norm": item["target_norm"].tolist(),
            "target_model_norm": target_model_norm,
            "target_coordinate_frame": args.target_coordinate_frame,
            "target_full_grasp": item["target_grasp"],
            "target_labels": item["target_labels"],
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
            vcot_image_size=args.vcot_image_size,
            vcot_iou_threshold=args.vcot_iou_threshold,
            vcot_angle_threshold=args.vcot_angle_threshold,
        )
        summary.update({
            "evaluation_mode": "predicted_vcot",
            "checkpoint": args.checkpoint,
            "target_coordinate_frame": args.target_coordinate_frame,
            "bbox_edge_expand": args.bbox_edge_expand,
            "min_bbox_half_size": args.min_bbox_half_size,
            "target_grasp_index": args.target_grasp_index,
            "loaded_vcot_config": args.loaded_vcot_config,
        })
        for output in outputs:
            output.pop("target_labels", None)
        output_path.write_text(json.dumps({"summary": summary, "outputs": outputs}, indent=2), encoding="utf-8")
        print(f"{name}: {json.dumps(summary, ensure_ascii=False)}")
        print(f"saved: {output_path}")


def main():
    args = parse_args()
    assert args.batch_size == 1, "Only batch size 1 is supported"
    args.checkpoint = str(Path(args.checkpoint).resolve())
    args = apply_vcot_config(args)
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
        print(
            "vcot eval config: "
            f"target_coordinate_frame={args.target_coordinate_frame}, "
            f"bbox_edge_expand={args.bbox_edge_expand}, "
            f"min_bbox_half_size={args.min_bbox_half_size}, "
            f"target_grasp_index={args.target_grasp_index}, "
            f"loaded_vcot_config={args.loaded_vcot_config}"
        )
        print(
            "predicted VCoT pipeline: full image -> detect bbox -> predicted crop -> two-image grasp"
        )

    root = Path(args.manifest_root)
    for dataset_name in args.datasets.split(","):
        manifest = Path(DEFAULT_DATASETS.get(dataset_name, dataset_name))
        if not manifest.is_absolute():
            manifest = root / manifest
        evaluate_dataset(args, model, tokenizer, dataset_name, manifest, image_size, use_thumbnail)


if __name__ == "__main__":
    main()
