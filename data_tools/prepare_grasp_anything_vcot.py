from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_tools.prepare_grasp_anything_bbox import write_split as write_bbox_split

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
    parser.add_argument("--bbox-ratio", type=float, default=0.5)
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    crop_root = Path(args.crop_root).resolve()
    bbox_root = Path(args.bbox_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    crop_train = crop_root / "train.jsonl"
    bbox_train = bbox_root / "train.jsonl"
    if not crop_train.exists():
        raise FileNotFoundError(f"Missing crop train manifest: {crop_train}")
    if not bbox_train.exists():
        print(f"Missing auxiliary bbox manifest, creating it for VCoT joint training: {bbox_train}")
        written = write_bbox_split(source_root, bbox_train, "train", limit=None)
        print(f"bbox train: wrote {written} rows -> {bbox_train}")

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
            "vcot_target_coordinate_frame": "full_image",
            "vcot_bbox_edge_expand": 15,
            "vcot_min_bbox_half_size": 50,
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
    print(f"bbox train: {bbox_length} rows, ratio={bbox_repeat_time}, effective={int(bbox_length * bbox_repeat_time)}")
    print(f"meta: {meta_path}")


if __name__ == "__main__":
    main()
