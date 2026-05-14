from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNVL_CHAT = REPO_ROOT / "InternVL/internvl_chat"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(INTERNVL_CHAT))

from data_tools.vcot_crop_lmdb import build_crop_grasp_item


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect one oracle-crop grasp LMDB manifest sample.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=416)
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.manifest).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == args.index:
                record = json.loads(line)
                break
        else:
            raise IndexError(args.index)

    item = build_crop_grasp_item(record, image_size=args.image_size, include_all_grasps=True)
    full_image, crop_image = item["image"]
    print(json.dumps({
        "full_image_size": full_image.size,
        "crop_image_size": crop_image.size,
        "crop_source": item["meta"]["crop_source"],
        "object_bbox": item["object_bbox"],
        "crop_box": item["crop_box"],
        "target_grasp": item["target_grasp"],
        "target_norm_full_image": item["target_norm"],
        "target_label_count": len(item["target_labels"]),
        "conversations": item["conversations"],
        "meta": item["meta"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
