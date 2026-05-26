from __future__ import annotations

import argparse
import itertools
import json
import os
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
from data_tools.vcot_crop_lmdb import (
    DEFAULT_BBOX_EDGE_EXPAND,
    DEFAULT_MIN_BBOX_HALF_SIZE,
    TARGET_FRAME_CROP_IMAGE,
    TARGET_FRAME_FULL_IMAGE,
    build_crop_grasp_item,
    transform_grasp_from_crop_norm,
)
from data_tools.vcot_direct_lmdb import _normalize_grasp
from evaluate_direct_grasp import (
    VCOT_ANGLE_THRESHOLD,
    VCOT_IOU_THRESHOLD,
    InferenceSampler,
    angle_diff_180,
    chat_preserve_loc_tokens,
    decode_loc_tokens,
    denormalize_grasp,
    load_model_and_tokenizer,
    rank,
    rotated_rect_iou,
    vcot_grasp_success,
)
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from grasp_settings import build_settings


SETTINGS = build_settings()
DEFAULT_DATA_INDEX_ROOT = Path(SETTINGS["GRASP_CROP_DEFAULT_INDEX_ROOT"])
DEFAULT_OUT_DIR = Path(SETTINGS["GRASP_CROP_RESULT_ROOT"])
DEFAULT_DATASETS = {
    "test_seen": "test_seen.jsonl",
    "test_unseen": "test_unseen.jsonl",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate oracle-crop two-image grasp generation.")
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
        help="Optional explicit VCoT/crop config JSON. Otherwise eval searches the checkpoint and its parent.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vcot-iou-threshold", type=float, default=VCOT_IOU_THRESHOLD)
    parser.add_argument("--vcot-angle-threshold", type=float, default=VCOT_ANGLE_THRESHOLD)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true")
    return parser.parse_args()


def full_grasp_like_vcot(grasp: list[float]) -> list[float]:
    return [int(value) for value in grasp[:4]] + [float(grasp[4])]


def full_grasp_to_norm(grasp: list[float], image_size: int = 416) -> list[float]:
    return [
        max(0.0, min(1.0, float(grasp[0]) / image_size)),
        max(0.0, min(1.0, float(grasp[1]) / image_size)),
        max(0.0, min(1.0, float(grasp[2]) / image_size)),
        max(0.0, min(1.0, float(grasp[3]) / image_size)),
        max(0.0, min(1.0, float(grasp[4]) / 180.0)),
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
    if args.data_index_root is None:
        if not config.get("data_index_root"):
            raise ValueError(f"{config_path} is missing data_index_root.")
        args.data_index_root = str(Path(config["data_index_root"]).expanduser().resolve())
    else:
        args.data_index_root = str(Path(args.data_index_root).expanduser().resolve())

    if args.target_coordinate_frame not in {TARGET_FRAME_FULL_IMAGE, TARGET_FRAME_CROP_IMAGE}:
        raise ValueError(
            f"Unsupported target_coordinate_frame={args.target_coordinate_frame}; "
            f"expected {TARGET_FRAME_FULL_IMAGE} or {TARGET_FRAME_CROP_IMAGE}"
        )
    args.loaded_vcot_config = str(config_path)
    return args


class CropGraspDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        prompt: str = "<image>\n<image>\ngrasp the {obj_name}",
        image_size: int = 448,
        vcot_image_size: int = 416,
        dynamic_image_size: bool = True,
        use_thumbnail: bool = True,
        max_num: int = 6,
        bbox_edge_expand: int = DEFAULT_BBOX_EDGE_EXPAND,
        min_bbox_half_size: int = DEFAULT_MIN_BBOX_HALF_SIZE,
        target_coordinate_frame: str = TARGET_FRAME_FULL_IMAGE,
        target_grasp_index: int = 0,
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
        self.bbox_edge_expand = bbox_edge_expand
        self.min_bbox_half_size = min_bbox_half_size
        self.target_coordinate_frame = target_coordinate_frame
        self.target_grasp_index = target_grasp_index
        self.transform = build_transform(is_train=False, input_size=image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        item = build_crop_grasp_item(
            record,
            image_size=self.vcot_image_size,
            bbox_edge_expand=self.bbox_edge_expand,
            min_bbox_half_size=self.min_bbox_half_size,
            target_coordinate_frame=self.target_coordinate_frame,
            target_grasp_index=self.target_grasp_index,
            include_all_grasps=True,
        )
        source_images: list[Image.Image] = item["image"]
        patch_images = []
        num_patches = []
        per_image_max_num = max(1, self.max_num // len(source_images))
        for image in source_images:
            images = dynamic_preprocess(
                image,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
                max_num=per_image_max_num,
            ) if self.dynamic_image_size else [image]
            patch_images.extend(images)
            num_patches.append(len(images))
        pixel_values = torch.stack([self.transform(image) for image in patch_images])
        target = decode_loc_tokens(item["conversations"][1]["value"])
        if target is None:
            raise ValueError(f"Could not decode target loc tokens for record {idx}")
        question = self.prompt.format(obj_name=record["obj_name"])
        full_target_norm = _normalize_grasp(item["target_grasp"], self.vcot_image_size)
        return {
            "pixel_values": pixel_values,
            "question": question,
            "num_patches": num_patches,
            "target": torch.tensor(target, dtype=torch.float32),
            "full_target_norm": torch.tensor(full_target_norm, dtype=torch.float32),
            "target_labels": item["target_labels"],
            "crop_box": item["crop_box"],
            "target_grasp": item["target_grasp"],
            "target_coordinate_frame": item["target_coordinate_frame"],
            "record": record,
        }


def collate_fn(batch):
    assert len(batch) == 1, "Only batch size 1 is supported for variable dynamic image patches."
    item = batch[0]
    return (
        item["pixel_values"],
        item["question"],
        item["num_patches"],
        item["target"],
        item["full_target_norm"],
        item["target_labels"],
        item["crop_box"],
        item["target_grasp"],
        item["target_coordinate_frame"],
        item["record"],
    )


def summarize(
    outputs,
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
    pred_full = torch.tensor([out["pred_full_grasp"] for out in valid], dtype=torch.float32)
    target_full = torch.tensor([out["target_full_grasp"] for out in valid], dtype=torch.float32)
    abs_full = (pred_full - target_full).abs()

    xy_dist = torch.linalg.vector_norm(pred_full[:, :2] - target_full[:, :2], dim=1)
    wh_dist = torch.linalg.vector_norm(pred_full[:, 2:4] - target_full[:, 2:4], dim=1)
    angle_err_raw = abs_full[:, 4]
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
        pred_grasp = full_grasp_like_vcot(out["pred_full_grasp"])
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
        "mean_abs_full_pixel_angle": abs_full.mean(0).tolist(),
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
    dataset = CropGraspDataset(
        manifest=manifest,
        image_size=image_size,
        vcot_image_size=args.vcot_image_size,
        dynamic_image_size=True,
        use_thumbnail=use_thumbnail,
        max_num=args.max_num,
        bbox_edge_expand=args.bbox_edge_expand,
        min_bbox_half_size=args.min_bbox_half_size,
        target_coordinate_frame=args.target_coordinate_frame,
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
    for (
        pixel_values,
        question,
        num_patches,
        target,
        full_target_norm,
        target_labels,
        crop_box,
        target_grasp,
        target_coordinate_frame,
        record,
    ) in tqdm(
        loader,
        disable=rank() != 0,
    ):
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
            num_patches_list=num_patches,
        )
        pred = decode_loc_tokens(answer)
        if pred is None:
            pred_full = None
            pred_full_norm = None
        elif target_coordinate_frame == TARGET_FRAME_CROP_IMAGE:
            pred_full = transform_grasp_from_crop_norm(pred, crop_box)
            pred_full_norm = full_grasp_to_norm(pred_full, args.vcot_image_size)
        else:
            pred_full = denormalize_grasp(pred, args.vcot_image_size)
            pred_full_norm = pred
        outputs.append({
            "answer": answer,
            "pred_norm": pred_full_norm,
            "pred_model_norm": pred,
            "pred_full_grasp": pred_full,
            "target_norm": full_target_norm.tolist(),
            "target_model_norm": target.tolist(),
            "target_full_grasp": target_grasp,
            "target_labels": target_labels,
            "crop_box": crop_box,
            "target_coordinate_frame": target_coordinate_frame,
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
            "evaluation_mode": "oracle_crop",
            "dataset_name": name,
            "split": name,
            "data_index": str(manifest.resolve()),
            "data_index_root": str(Path(args.data_index_root).resolve()),
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
            "crop eval config: "
            f"target_coordinate_frame={args.target_coordinate_frame}, "
            f"bbox_edge_expand={args.bbox_edge_expand}, "
            f"min_bbox_half_size={args.min_bbox_half_size}, "
            f"target_grasp_index={args.target_grasp_index}, "
            f"loaded_vcot_config={args.loaded_vcot_config}"
        )

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
