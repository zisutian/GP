from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNVL_CHAT = REPO_ROOT / "InternVL/internvl_chat"
sys.path.insert(0, str(INTERNVL_CHAT))

from internvl.train.vcot_direct_lmdb import build_direct_grasp_item


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect one direct-grasp LMDB manifest sample.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, default=0)
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

    item = build_direct_grasp_item(record)
    print(json.dumps({
        "image_size": item["image"].size,
        "conversations": item["conversations"],
        "meta": item["meta"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
