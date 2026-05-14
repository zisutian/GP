from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNVL_CHAT = REPO_ROOT / "InternVL/internvl_chat"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(INTERNVL_CHAT))

from data_tools.vcot_bbox_lmdb import build_bbox_item


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect one bbox-detection LMDB manifest sample.")
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

    item = build_bbox_item(record, image_size=args.image_size)
    print(json.dumps({
        "image_size": item["image"].size,
        "target_bbox_xyxy": item["target_bbox"],
        "target_norm_xyxy": item["target_norm"],
        "conversations": item["conversations"],
        "meta": item["meta"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
