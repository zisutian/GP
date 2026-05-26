from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EVALUATION_MODE_TO_METHOD = {
    "direct_grasp": "direct",
    "oracle_crop": "oracle_crop",
    "predicted_vcot": "pred_vcot",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Map result JSON files to their expected checkpoint directories.")
    parser.add_argument("results", nargs="+")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_value(value: Any) -> str:
    return "" if value is None else str(value)


def config_work_dir(config_path: Path, config: dict[str, Any]) -> Path:
    output_dir = config.get("output_dir")
    if not output_dir:
        raise ValueError(f"{config_path} is missing output_dir.")
    return Path(output_dir)


def required_summary_value(summary: dict[str, Any], key: str, result_path: Path) -> Any:
    value = summary.get(key)
    if value in (None, ""):
        raise ValueError(f"{result_path} is missing summary.{key}; rerun eval with the current result schema.")
    return value


def infer(path: Path) -> dict[str, str]:
    data = load_json(path)
    summary = data.get("summary", {}) if isinstance(data.get("summary", {}), dict) else {}

    config_path = Path(str(required_summary_value(summary, "loaded_vcot_config", path)))
    if not config_path.exists():
        raise FileNotFoundError(f"Loaded vcot_config.json does not exist for {path}: {config_path}")
    config = load_json(config_path)

    work_path = config_work_dir(config_path, config)
    checkpoint = Path(str(required_summary_value(summary, "checkpoint", path)))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist for {path}: {checkpoint}")

    evaluation_mode = str(required_summary_value(summary, "evaluation_mode", path))
    if evaluation_mode not in EVALUATION_MODE_TO_METHOD:
        raise ValueError(f"Unsupported evaluation_mode for {path}: {evaluation_mode}")
    method = EVALUATION_MODE_TO_METHOD[evaluation_mode]
    experiment = str(config["experiment_name"])
    split = str(required_summary_value(summary, "split", path))
    uses_crop_contract = evaluation_mode in {"oracle_crop", "predicted_vcot"}

    def crop_summary_value(key: str) -> str:
        if not uses_crop_contract:
            return ""
        return csv_value(required_summary_value(summary, key, path))

    return {
        "method": method,
        "experiment": experiment,
        "split": split,
        "result_path": str(path),
        "evaluation_mode": evaluation_mode,
        "target_coordinate_frame": crop_summary_value("target_coordinate_frame"),
        "bbox_edge_expand": crop_summary_value("bbox_edge_expand"),
        "min_bbox_half_size": crop_summary_value("min_bbox_half_size"),
        "target_grasp_index": crop_summary_value("target_grasp_index"),
        "work_dir": str(work_path),
        "data_index_root": csv_value(config.get("data_index_root", "")),
        "latest_checkpoint": str(checkpoint),
        "checkpoint_exists": "True",
        "loaded_vcot_config": str(config_path),
        "config_exists": "True",
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
        "evaluation_mode",
        "target_coordinate_frame",
        "bbox_edge_expand",
        "min_bbox_half_size",
        "target_grasp_index",
        "work_dir",
        "data_index_root",
        "latest_checkpoint",
        "checkpoint_exists",
        "loaded_vcot_config",
        "config_exists",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"checkpoint_manifest={out_path}")


if __name__ == "__main__":
    main()
