from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Write/read grasp experiment metadata stored as vcot_config.json.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write = subparsers.add_parser("write", help="Write one experiment metadata JSON.")
    write.add_argument("--pipeline", required=True, choices=["direct_grasp", "oracle_crop", "predicted_vcot"])
    write.add_argument("--experiment-name", required=True)
    write.add_argument("--meta-path", required=True)
    write.add_argument("--data-index-root", default=None)
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
    write.add_argument("--bbox-loss-weight", type=float, default=None)
    write.add_argument("--target-coordinate-frame", default=None)
    write.add_argument("--bbox-edge-expand", type=int, default=None)
    write.add_argument("--min-bbox-half-size", type=int, default=None)
    write.add_argument("--target-grasp-index", type=int, default=None)

    env = subparsers.add_parser("env", help="Export metadata values for one checkpoint or metadata JSON.")
    env.add_argument("--checkpoint", default=None)
    env.add_argument("--config-path", default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def config_path_for(checkpoint: str | Path | None = None, config_path: str | Path | None = None) -> Path:
    if config_path is not None:
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"vcot_config.json not found: {path}")
        return path

    if checkpoint is None:
        raise ValueError("Pass --checkpoint or --config-path.")

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    candidates = []
    if checkpoint_path.is_dir():
        candidates.append(checkpoint_path / "vcot_config.json")
        candidates.append(checkpoint_path.parent / "vcot_config.json")
    else:
        candidates.append(checkpoint_path.parent / "vcot_config.json")
        candidates.append(checkpoint_path.parent.parent / "vcot_config.json")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"vcot_config.json not found for checkpoint {checkpoint_path}. "
        "Expected it in the checkpoint directory or its parent."
    )


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
    data_index_root = Path(args.data_index_root).expanduser().resolve() if args.data_index_root else meta_path.parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    meta = read_json(meta_path)
    crop_meta = dataset_meta(meta, "grasp_anything_crop")
    bbox_meta = dataset_meta(meta, "grasp_anything_bbox")

    config: dict[str, Any] = {
        "pipeline": args.pipeline,
        "experiment_name": args.experiment_name,
        "gp_root": resolve_path(args.gp_root),
        "meta_path": str(meta_path),
        "data_index_root": str(data_index_root),
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
        config["grasp_loss_weight"] = float(first_not_none(
            crop_meta.get("vcot_loss_weight"),
            default=1.0,
        ))
        config["bbox_loss_weight"] = float(first_not_none(
            args.bbox_loss_weight,
            bbox_meta.get("vcot_loss_weight"),
            default=1.0,
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


def print_shell_env(args: argparse.Namespace) -> None:
    config_path = config_path_for(checkpoint=args.checkpoint, config_path=args.config_path)
    config = read_json(config_path)
    values = {
        "VCOT_CONFIG_PATH": str(config_path),
        "VCOT_PIPELINE": str(config.get("pipeline", "")),
        "VCOT_EXPERIMENT_NAME": str(config.get("experiment_name", "")),
        "VCOT_DATA_INDEX_ROOT": str(config.get("data_index_root", "")),
        "VCOT_OUTPUT_DIR": str(config.get("output_dir", "")),
    }
    for key, value in values.items():
        print(f"export {key}={shlex.quote(value)}")


def main() -> None:
    args = parse_args()
    if args.command == "write":
        write_from_args(args)
    elif args.command == "env":
        print_shell_env(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
