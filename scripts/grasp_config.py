from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Write grasp pipeline vcot_config.json files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write = subparsers.add_parser("write", help="Write one vcot_config.json.")
    write.add_argument("--pipeline", required=True, choices=["direct_grasp", "oracle_crop", "predicted_vcot"])
    write.add_argument("--experiment-name", required=True)
    write.add_argument("--meta-path", required=True)
    write.add_argument("--output-dir", required=True)
    write.add_argument("--config-path", default=None)
    write.add_argument("--copy-to-checkpoints", action="store_true")
    write.add_argument("--gp-root", default=str(REPO_ROOT))
    write.add_argument("--model-path", default=None)
    write.add_argument("--crop-root", default=None)
    write.add_argument("--bbox-root", default=None)
    write.add_argument("--use-llm-lora", type=int, default=None)
    write.add_argument("--learning-rate", default=None)
    write.add_argument("--num-train-epochs", type=float, default=None)
    write.add_argument("--max-dynamic-patch", type=int, default=None)
    write.add_argument("--force-image-size", type=int, default=None)
    write.add_argument("--bbox-ratio", type=float, default=None)
    write.add_argument("--target-coordinate-frame", default=None)
    write.add_argument("--bbox-edge-expand", type=int, default=None)
    write.add_argument("--min-bbox-half-size", type=int, default=None)
    write.add_argument("--target-grasp-index", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def dataset_meta(meta: dict[str, Any], vcot_dataset: str) -> dict[str, Any]:
    return next((value for value in meta.values() if value.get("vcot_dataset") == vcot_dataset), {})


def first_not_none(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def optional_path(value: str | None) -> str | None:
    return resolve_path(value) if value else None


def required_value(value: Any, source: str, key: str) -> Any:
    if value is None:
        raise ValueError(f"Missing required {key} in {source}")
    return value


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    meta_path = Path(args.meta_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    meta = read_json(meta_path)
    crop_meta = dataset_meta(meta, "grasp_anything_crop")
    bbox_meta = dataset_meta(meta, "grasp_anything_bbox")

    config: dict[str, Any] = {
        "pipeline": args.pipeline,
        "experiment_name": args.experiment_name,
        "gp_root": resolve_path(args.gp_root),
        "meta_path": str(meta_path),
        "output_dir": str(output_dir),
    }

    for key in ["model_path", "crop_root", "bbox_root"]:
        value = optional_path(getattr(args, key, None))
        if value is not None:
            config[key] = value

    for key in ["use_llm_lora", "learning_rate", "num_train_epochs", "max_dynamic_patch", "force_image_size"]:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    if args.pipeline in {"oracle_crop", "predicted_vcot"}:
        config.update({
            "target_coordinate_frame": required_value(
                first_not_none(args.target_coordinate_frame, crop_meta.get("vcot_target_coordinate_frame")),
                str(meta_path),
                "vcot_target_coordinate_frame",
            ),
            "bbox_edge_expand": int(required_value(
                first_not_none(args.bbox_edge_expand, crop_meta.get("vcot_bbox_edge_expand")),
                str(meta_path),
                "vcot_bbox_edge_expand",
            )),
            "min_bbox_half_size": int(required_value(
                first_not_none(args.min_bbox_half_size, crop_meta.get("vcot_min_bbox_half_size")),
                str(meta_path),
                "vcot_min_bbox_half_size",
            )),
            "target_grasp_index": int(required_value(
                first_not_none(args.target_grasp_index, crop_meta.get("vcot_target_grasp_index")),
                str(meta_path),
                "vcot_target_grasp_index",
            )),
        })

    if args.pipeline == "predicted_vcot":
        config["bbox_ratio"] = float(required_value(
            first_not_none(args.bbox_ratio, bbox_meta.get("repeat_time")),
            str(meta_path),
            "bbox_ratio/repeat_time",
        ))

    return config


def write_config(config: dict[str, Any], config_path: Path, copy_to_checkpoints: bool = False) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    config_path.write_text(text, encoding="utf-8")

    if copy_to_checkpoints:
        work_dir = Path(config["output_dir"])
        for checkpoint_dir in sorted(work_dir.glob("checkpoint-*")):
            if checkpoint_dir.is_dir():
                (checkpoint_dir / "vcot_config.json").write_text(text, encoding="utf-8")
    print(f"wrote {config_path}")


def write_from_args(args: argparse.Namespace) -> None:
    config = build_config(args)
    config_path = Path(args.config_path).expanduser().resolve() if args.config_path else Path(config["output_dir"]) / "vcot_config.json"
    write_config(config, config_path, copy_to_checkpoints=args.copy_to_checkpoints)


def main() -> None:
    args = parse_args()
    if args.command == "write":
        write_from_args(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
