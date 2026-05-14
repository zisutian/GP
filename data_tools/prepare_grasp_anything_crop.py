from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_tools.vcot_crop_lmdb import DEFAULT_BBOX_EDGE_EXPAND, DEFAULT_MIN_BBOX_HALF_SIZE

DEFAULT_SOURCE_ROOT = (REPO_ROOT / "../VCoT-Grasp-self/data/grasp_anything").resolve()
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/vcot_grasp/crop"


def parse_args():
    parser = argparse.ArgumentParser(description="Create oracle-crop grasp manifests for Grasp-Anything LMDB.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--splits", nargs="+", default=["train", "test_seen", "test_unseen"])
    parser.add_argument("--bbox-edge-expand", type=int, default=DEFAULT_BBOX_EDGE_EXPAND)
    parser.add_argument("--min-bbox-half-size", type=int, default=DEFAULT_MIN_BBOX_HALF_SIZE)
    parser.add_argument("--target-coordinate-frame", choices=["full_image", "crop_image"], default="full_image")
    parser.add_argument("--target-grasp-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit per split.")
    return parser.parse_args()


def iter_split(
    csv_path: Path,
    bbox_edge_expand: int,
    min_bbox_half_size: int,
    target_coordinate_frame: str,
    target_grasp_index: int,
):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            grasp_id, obj_name, scene_description = row[:3]
            scene_id = grasp_id.split("_")[0]
            yield {
                "dataset": "grasp_anything_crop",
                "source_root": None,
                "grasp_id": grasp_id,
                "scene_id": scene_id,
                "obj_name": obj_name,
                "scene_description": scene_description,
                "image_key": f"{scene_id}.jpg",
                "grasp_key": f"{grasp_id}.pt",
                "mask_key": f"{grasp_id}.npy",
                "bbox_edge_expand": bbox_edge_expand,
                "min_bbox_half_size": min_bbox_half_size,
                "target_coordinate_frame": target_coordinate_frame,
                "target_grasp_index": target_grasp_index,
            }


def write_split(
    source_root: Path,
    output_path: Path,
    split: str,
    bbox_edge_expand: int,
    min_bbox_half_size: int,
    target_coordinate_frame: str,
    target_grasp_index: int,
    limit: int | None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = source_root / "origin_split" / f"{split}.csv"
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for item in iter_split(
            csv_path,
            bbox_edge_expand,
            min_bbox_half_size,
            target_coordinate_frame,
            target_grasp_index,
        ):
            item["source_root"] = str(source_root)
            item["split"] = split
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    lengths = {}
    for split in args.splits:
        output_path = output_root / f"{split}.jsonl"
        lengths[split] = write_split(
            source_root=source_root,
            output_path=output_path,
            split=split,
            bbox_edge_expand=args.bbox_edge_expand,
            min_bbox_half_size=args.min_bbox_half_size,
            target_coordinate_frame=args.target_coordinate_frame,
            target_grasp_index=args.target_grasp_index,
            limit=args.limit,
        )
        print(f"{split}: wrote {lengths[split]} rows -> {output_path}")

    train_meta = {
        "grasp_anything_crop_train": {
            "root": "",
            "annotation": str(output_root / "train.jsonl"),
            "data_augment": False,
            "repeat_time": 1,
            "length": lengths.get("train", 0),
            "vcot_dataset": "grasp_anything_crop",
            "vcot_image_size": 416,
            "vcot_crop_source": "gt_mask_object_bbox",
            "vcot_target_coordinate_frame": args.target_coordinate_frame,
            "vcot_bbox_edge_expand": args.bbox_edge_expand,
            "vcot_min_bbox_half_size": args.min_bbox_half_size,
        }
    }
    meta_path = output_root / "internvl_meta_train.json"
    meta_path.write_text(json.dumps(train_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"meta: {meta_path}")


if __name__ == "__main__":
    main()
