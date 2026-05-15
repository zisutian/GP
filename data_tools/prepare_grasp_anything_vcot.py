from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_tools.prepare_grasp_anything_bbox import write_split as write_bbox_split
from data_tools.prepare_grasp_anything_crop import write_split as write_crop_split
from data_tools.vcot_crop_lmdb import DEFAULT_BBOX_EDGE_EXPAND, DEFAULT_MIN_BBOX_HALF_SIZE

DEFAULT_SOURCE_ROOT = (REPO_ROOT / "../VCoT-Grasp-self/data/grasp_anything").resolve()
DEFAULT_CROP_ROOT = REPO_ROOT / "data/vcot_grasp/crop"
DEFAULT_BBOX_ROOT = REPO_ROOT / "data/vcot_grasp/bbox"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/vcot_grasp/vcot"


def parse_args():
    parser = argparse.ArgumentParser(description="Create VCoT-style joint bbox+crop-grasp training meta.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--crop-root", default=str(DEFAULT_CROP_ROOT))
    parser.add_argument("--bbox-root", default=str(DEFAULT_BBOX_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--eval-splits", nargs="+", default=["test_seen", "test_unseen"])
    parser.add_argument("--bbox-ratio", type=float, default=0.5)
    parser.add_argument("--bbox-edge-expand", type=int, default=DEFAULT_BBOX_EDGE_EXPAND)
    parser.add_argument("--min-bbox-half-size", type=int, default=DEFAULT_MIN_BBOX_HALF_SIZE)
    parser.add_argument("--target-coordinate-frame", choices=["full_image", "crop_image"], default="full_image")
    parser.add_argument("--target-grasp-index", type=int, default=0)
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def manifest_matches_crop_config(
    path: Path,
    bbox_edge_expand: int,
    min_bbox_half_size: int,
    target_coordinate_frame: str,
    target_grasp_index: int,
) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
    if not first_line:
        return False
    first = json.loads(first_line)
    return (
        first.get("bbox_edge_expand") == bbox_edge_expand
        and first.get("min_bbox_half_size") == min_bbox_half_size
        and first.get("target_coordinate_frame") == target_coordinate_frame
        and first.get("target_grasp_index") == target_grasp_index
    )


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    crop_root = Path(args.crop_root).resolve()
    bbox_root = Path(args.bbox_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    crop_train = crop_root / "train.jsonl"
    bbox_train = bbox_root / "train.jsonl"
    if crop_train.exists() and not manifest_matches_crop_config(
        crop_train,
        args.bbox_edge_expand,
        args.min_bbox_half_size,
        args.target_coordinate_frame,
        args.target_grasp_index,
    ):
        raise ValueError(
            f"Existing crop manifest does not match requested VCoT crop config: {crop_train}. "
            "Use a separate --crop-root for this experiment or regenerate the crop manifest."
        )

    if not crop_train.exists():
        print(f"Missing crop manifest, creating it for VCoT joint training: {crop_train}")
        written = write_crop_split(
            source_root=source_root,
            output_path=crop_train,
            split="train",
            bbox_edge_expand=args.bbox_edge_expand,
            min_bbox_half_size=args.min_bbox_half_size,
            target_coordinate_frame=args.target_coordinate_frame,
            target_grasp_index=args.target_grasp_index,
            limit=None,
        )
        print(f"crop train: wrote {written} rows -> {crop_train}")
    if not bbox_train.exists():
        print(f"Missing auxiliary bbox manifest, creating it for VCoT joint training: {bbox_train}")
        written = write_bbox_split(source_root, bbox_train, "train", limit=None)
        print(f"bbox train: wrote {written} rows -> {bbox_train}")

    eval_lengths = {}
    for split in args.eval_splits:
        output_path = output_root / f"{split}.jsonl"
        eval_lengths[split] = write_crop_split(
            source_root=source_root,
            output_path=output_path,
            split=split,
            bbox_edge_expand=args.bbox_edge_expand,
            min_bbox_half_size=args.min_bbox_half_size,
            target_coordinate_frame=args.target_coordinate_frame,
            target_grasp_index=args.target_grasp_index,
            limit=None,
        )
        print(f"{split}: wrote {eval_lengths[split]} rows -> {output_path}")

    crop_length = count_jsonl(crop_train)
    bbox_length = count_jsonl(bbox_train)
    bbox_repeat_time = max(0.0, min(1.0, float(args.bbox_ratio)))

    train_meta = {
        "grasp_anything_crop_train": {
            "root": "",
            "annotation": str(crop_train),
            "data_augment": False,
            "repeat_time": 1,
            "length": crop_length,
            "vcot_dataset": "grasp_anything_crop",
            "vcot_image_size": 416,
            "vcot_crop_source": "gt_mask_object_bbox",
            "vcot_target_coordinate_frame": args.target_coordinate_frame,
            "vcot_bbox_edge_expand": args.bbox_edge_expand,
            "vcot_min_bbox_half_size": args.min_bbox_half_size,
            "vcot_target_grasp_index": args.target_grasp_index,
        },
        "grasp_anything_bbox_train": {
            "root": "",
            "annotation": str(bbox_train),
            "data_augment": False,
            "repeat_time": bbox_repeat_time,
            "length": int(bbox_length * bbox_repeat_time),
            "vcot_dataset": "grasp_anything_bbox",
            "vcot_image_size": 416,
            "vcot_target_coordinate_frame": "full_image",
            "vcot_target_bbox_order": "xyxy",
        },
    }
    meta_path = output_root / "internvl_meta_train.json"
    meta_path.write_text(json.dumps(train_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"crop train: {crop_length} rows")
    print(
        "crop config: "
        f"target_coordinate_frame={args.target_coordinate_frame}, "
        f"bbox_edge_expand={args.bbox_edge_expand}, "
        f"min_bbox_half_size={args.min_bbox_half_size}"
    )
    print(f"bbox train: {bbox_length} rows, ratio={bbox_repeat_time}, effective={int(bbox_length * bbox_repeat_time)}")
    print(f"meta: {meta_path}")


if __name__ == "__main__":
    main()
