from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print result JSON paths from an explicit list.")
    parser.add_argument("--list", required=True, dest="list_path")
    parser.add_argument("--mode", choices=["direct_grasp", "oracle_crop", "predicted_vcot"], default=None)
    return parser.parse_args()


def iter_paths(list_path: Path):
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            yield Path(line).expanduser()


def result_mode(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    return str(summary.get("evaluation_mode", ""))


def main() -> None:
    args = parse_args()
    for path in iter_paths(Path(args.list_path)):
        resolved = path.resolve()
        if args.mode is not None and result_mode(resolved) != args.mode:
            continue
        print(resolved)


if __name__ == "__main__":
    main()
