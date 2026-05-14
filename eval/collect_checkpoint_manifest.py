from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT / "InternVL/internvl_chat/work_dirs/internvl_chat_v2_5"


def parse_args():
    parser = argparse.ArgumentParser(description="Map result JSON files to their expected checkpoint directories.")
    parser.add_argument("results", nargs="+")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def split_name(path: Path) -> str:
    if "test_unseen" in path.name:
        return "test_unseen"
    if "test_seen" in path.name:
        return "test_seen"
    return "unknown"


def latest_checkpoint(work_dir: Path) -> Path | None:
    checkpoints = sorted(work_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    if checkpoints:
        return checkpoints[-1]
    if (work_dir / "model.safetensors").exists():
        return work_dir
    return None


def infer(path: Path) -> dict[str, str]:
    parts = path.parts
    method = "unknown"
    experiment = path.parent.name
    work_dir = ""

    if "vcot_grasp_direct" in parts:
        method = "direct"
        if "hparams" in parts:
            experiment = parts[parts.index("hparams") + 1]
            work_dir = str(WORK_ROOT / "grasp_direct_hparams" / experiment)
        else:
            experiment = "internvl2_5_1b_grasp_direct_lmdb_lora"
            work_dir = str(WORK_ROOT / experiment)
    elif "vcot_grasp_crop" in parts:
        method = "oracle_crop"
        if "hparams" in parts:
            experiment = parts[parts.index("hparams") + 1]
            work_dir = str(WORK_ROOT / "grasp_crop_hparams" / experiment)
        else:
            experiment = "crop_object_lora16_lr8e-5_ep1_patch6_edge15_half50"
            work_dir = str(WORK_ROOT / "grasp_crop_hparams" / experiment)
    elif "vcot_grasp_vcot" in parts:
        method = "pred_vcot"
        if "hparams" in parts:
            experiment = parts[parts.index("hparams") + 1]
        else:
            experiment = "vcot_lora16_lr8e-5_ep1_patch6_bbox0.5"
        work_dir = str(WORK_ROOT / "grasp_vcot_hparams" / experiment)

    work_path = Path(work_dir) if work_dir else Path()
    checkpoint = latest_checkpoint(work_path) if work_dir else None
    return {
        "method": method,
        "experiment": experiment,
        "split": split_name(path),
        "result_path": str(path),
        "work_dir": work_dir,
        "latest_checkpoint": str(checkpoint) if checkpoint else "",
        "checkpoint_exists": str(checkpoint is not None),
    }


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [infer(Path(result)) for result in args.results]
    fieldnames = [
        "method",
        "experiment",
        "split",
        "result_path",
        "work_dir",
        "latest_checkpoint",
        "checkpoint_exists",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"checkpoint_manifest={out_path}")


if __name__ == "__main__":
    main()
