from __future__ import annotations

import argparse
import os
import shlex
from dataclasses import dataclass
from pathlib import Path


GP_ROOT = Path(__file__).resolve().parent

# Edit this section. Other scripts consume these values.
DATASET_ROOT = "/nvme1/2025201095KZJ1/VCoTGrasp/data/grasp_anything"
DATA_INDEX_ROOT = "artifacts/data_index"
RUN_ROOT = "artifacts/runs"
RESULT_ROOT = "artifacts/results"
ANALYSIS_ROOT = "artifacts/analysis"
INTERNVL_ROOT = "InternVL"
MODEL_PATH = "InternVL/pretrained/OpenGVLab/InternVL2_5-1B"

PYTHON_BIN = "python"
TRAIN_GPUS = "2"
EVAL_GPUS = "1"
BATCH_SIZE = "16"
PER_DEVICE_BATCH_SIZE = "4"
FORCE_IMAGE_SIZE = "448"
EVAL_DATASETS = "test_seen,test_unseen"
RUN_EVAL = "1"
OVERWRITE_EVAL_RESULTS = "False"
EVAL_VCOT_IOU_THRESHOLD = "0.25"
EVAL_VCOT_ANGLE_THRESHOLD = "30.0"

DIRECT_CUDA_VISIBLE_DEVICES = "0,1"
CROP_CUDA_VISIBLE_DEVICES = "2,3"
VCOT_CUDA_VISIBLE_DEVICES = "2,3"
DIRECT_EVAL_CUDA_VISIBLE_DEVICES = DIRECT_CUDA_VISIBLE_DEVICES
CROP_EVAL_CUDA_VISIBLE_DEVICES = CROP_CUDA_VISIBLE_DEVICES
VCOT_EVAL_CUDA_VISIBLE_DEVICES = VCOT_CUDA_VISIBLE_DEVICES
DIRECT_EVAL_MASTER_PORT = "63669"
CROP_EVAL_MASTER_PORT = "63679"
VCOT_EVAL_MASTER_PORT = "63689"

DIRECT_EXPERIMENTS = [
    ("baseline_lora16_lr4e-5_ep1_patch6", "16", "4e-5", "1", "6"),
    ("lora8_lr4e-5_ep1_patch6", "8", "4e-5", "1", "6"),
    ("lora32_lr4e-5_ep1_patch6", "32", "4e-5", "1", "6"),
    ("lora16_lr2e-5_ep1_patch6", "16", "2e-5", "1", "6"),
    ("lora16_lr8e-5_ep1_patch6", "16", "8e-5", "1", "6"),
    ("lora16_lr4e-5_ep2_patch6", "16", "4e-5", "2", "6"),
    ("lora16_lr4e-5_ep1_patch4", "16", "4e-5", "1", "4"),
    ("lora16_lr4e-5_ep1_patch1", "16", "4e-5", "1", "1"),
]

CROP_EXPERIMENTS = [
    ("baseline_full_lora16_lr8e-5_ep1_patch6_edge15_half50", "16", "8e-5", "1", "15", "50", "full_image", "6", "0"),
    ("crop_full_lora16_lr8e-5_ep1_patch6_edge5_half40", "16", "8e-5", "1", "5", "40", "full_image", "6", "0"),
    ("crop_full_lora16_lr8e-5_ep1_patch6_edge10_half40", "16", "8e-5", "1", "10", "40", "full_image", "6", "0"),
    ("crop_frame_lora16_lr8e-5_ep1_patch6_edge5_half40", "16", "8e-5", "1", "5", "40", "crop_image", "6", "0"),
    ("crop_frame_lora16_lr8e-5_ep1_patch6_edge10_half40", "16", "8e-5", "1", "10", "40", "crop_image", "6", "0"),
    ("crop_frame_lora16_lr8e-5_ep1_patch8_edge10_half40", "16", "8e-5", "1", "10", "40", "crop_image", "8", "0"),
]

# name, lora, lr, epochs, bbox_ratio, bbox_loss_weight, edge_expand,
# min_half, target_frame, max_dynamic_patch, target_grasp_index
VCOT_EXPERIMENTS = [
    ("baseline_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox0.5", "16", "8e-5", "1", "0.5", "1.0", "5", "40", "crop_image", "6", "0"),
    ("vcot_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox0.25", "16", "8e-5", "1", "0.25", "1.0", "5", "40", "crop_image", "6", "0"),
    ("vcot_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox1.0", "16", "8e-5", "1", "1.0", "1.0", "5", "40", "crop_image", "6", "0"),
    ("vcot_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox0.5_lambda0.5", "16", "8e-5", "1", "0.5", "0.5", "5", "40", "crop_image", "6", "0"),
    ("vcot_frame_lora16_lr8e-5_ep1_patch6_edge5_half40_bbox0.5_lambda2.0", "16", "8e-5", "1", "0.5", "2.0", "5", "40", "crop_image", "6", "0"),
    ("vcot_frame_lora16_lr8e-5_ep1_patch6_edge10_half40_bbox0.5", "16", "8e-5", "1", "0.5", "1.0", "10", "40", "crop_image", "6", "0"),
    ("vcot_frame_lora16_lr8e-5_ep1_patch8_edge10_half40_bbox0.5", "16", "8e-5", "1", "0.5", "1.0", "10", "40", "crop_image", "8", "0"),
    ("vcot_full_lora16_lr8e-5_ep1_patch6_edge15_half50_bbox0.25", "16", "8e-5", "1", "0.25", "1.0", "15", "50", "full_image", "6", "0"),
    ("vcot_full_lora16_lr8e-5_ep1_patch6_edge15_half50_bbox0.5", "16", "8e-5", "1", "0.5", "1.0", "15", "50", "full_image", "6", "0"),
    ("vcot_full_lora16_lr8e-5_ep1_patch6_edge15_half50_bbox1.0", "16", "8e-5", "1", "1.0", "1.0", "15", "50", "full_image", "6", "0"),
]


def resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute():
        path = GP_ROOT / path
    return path.resolve()


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_path(name: str, default: str | Path) -> Path:
    return resolve_path(env(name, str(default)))


@dataclass(frozen=True)
class Settings:
    values: dict[str, str]

    def __getitem__(self, key: str) -> str:
        return self.values[key]


def build_settings() -> Settings:
    dataset_root = env_path("GRASP_DATASET_ROOT", DATASET_ROOT)
    data_index_root = env_path("GRASP_DATA_INDEX_ROOT", DATA_INDEX_ROOT)
    run_root = env_path("GRASP_RUN_ROOT", RUN_ROOT)
    result_root = env_path("GRASP_RESULT_ROOT", RESULT_ROOT)
    analysis_root = env_path("GRASP_ANALYSIS_ROOT", ANALYSIS_ROOT)
    internvl_root = env_path("INTERNVL_ROOT", INTERNVL_ROOT)
    internvl_chat_root = env_path("INTERNVL_CHAT_ROOT", internvl_root / "internvl_chat")
    model_path = env_path("GRASP_MODEL_PATH", MODEL_PATH)
    worker_script_root = env_path("GRASP_WORKER_SCRIPT_ROOT", GP_ROOT / "scripts/grasp_workers")
    train_script_root = env_path("GRASP_TRAIN_SCRIPT_ROOT", worker_script_root)
    eval_script_root = env_path("GRASP_EVAL_SCRIPT_ROOT", worker_script_root)
    direct_use_lora = env("GRASP_DIRECT_USE_LLM_LORA", DIRECT_EXPERIMENTS[0][1])
    direct_lr = env("GRASP_DIRECT_LEARNING_RATE", DIRECT_EXPERIMENTS[0][2])
    direct_epochs = env("GRASP_DIRECT_NUM_TRAIN_EPOCHS", DIRECT_EXPERIMENTS[0][3])
    direct_patch = env("GRASP_DIRECT_MAX_DYNAMIC_PATCH", DIRECT_EXPERIMENTS[0][4])
    crop_use_lora = env("GRASP_CROP_USE_LLM_LORA", CROP_EXPERIMENTS[0][1])
    crop_lr = env("GRASP_CROP_LEARNING_RATE", CROP_EXPERIMENTS[0][2])
    crop_epochs = env("GRASP_CROP_NUM_TRAIN_EPOCHS", CROP_EXPERIMENTS[0][3])
    crop_target_index = env("GRASP_CROP_TARGET_GRASP_INDEX", CROP_EXPERIMENTS[0][8])
    vcot_use_lora = env("GRASP_VCOT_USE_LLM_LORA", VCOT_EXPERIMENTS[0][1])
    vcot_lr = env("GRASP_VCOT_LEARNING_RATE", VCOT_EXPERIMENTS[0][2])
    vcot_epochs = env("GRASP_VCOT_NUM_TRAIN_EPOCHS", VCOT_EXPERIMENTS[0][3])
    vcot_target_index = env("GRASP_VCOT_TARGET_GRASP_INDEX", VCOT_EXPERIMENTS[0][10])
    default_crop_index_root = data_index_root / "crop" / CROP_EXPERIMENTS[0][0]
    default_vcot_index_root = data_index_root / "vcot" / VCOT_EXPERIMENTS[0][0]

    values = {
        "GP_ROOT": str(GP_ROOT),
        "GRASP_DATASET_ROOT": str(dataset_root),
        "GRASP_IMAGE_LMDB": str(dataset_root / "lmdb/image"),
        "GRASP_GRASP_LMDB": str(dataset_root / "lmdb/grasp_label_positive"),
        "GRASP_MASK_LMDB": str(dataset_root / "lmdb/mask"),
        "GRASP_DATA_INDEX_ROOT": str(data_index_root),
        "GRASP_RUN_ROOT": str(run_root),
        "GRASP_RESULT_ROOT": str(result_root),
        "GRASP_ANALYSIS_ROOT": str(analysis_root),
        "INTERNVL_ROOT": str(internvl_root),
        "INTERNVL_CHAT_ROOT": str(internvl_chat_root),
        "GRASP_MODEL_PATH": str(model_path),
        "GRASP_MODEL_ROOT": str(model_path.parent.parent),
        "GRASP_WORKER_SCRIPT_ROOT": str(worker_script_root),
        "GRASP_TRAIN_SCRIPT_ROOT": str(train_script_root),
        "GRASP_EVAL_SCRIPT_ROOT": str(eval_script_root),
        "GRASP_DIRECT_TRAIN_SCRIPT": str(train_script_root / "train_direct_lmdb_lora.sh"),
        "GRASP_CROP_TRAIN_SCRIPT": str(train_script_root / "train_crop_lmdb_lora.sh"),
        "GRASP_VCOT_TRAIN_SCRIPT": str(train_script_root / "train_vcot_lmdb_lora.sh"),
        "GRASP_DIRECT_EVAL_SCRIPT": str(eval_script_root / "eval_direct_lmdb_lora.sh"),
        "GRASP_CROP_EVAL_SCRIPT": str(eval_script_root / "eval_crop_lmdb_lora.sh"),
        "GRASP_VCOT_EVAL_SCRIPT": str(eval_script_root / "eval_vcot_lmdb_lora.sh"),
        "GRASP_DIRECT_INDEX_ROOT": str(data_index_root / "direct"),
        "GRASP_CROP_INDEX_ROOT": str(data_index_root / "crop"),
        "GRASP_CROP_DEFAULT_INDEX_ROOT": str(default_crop_index_root),
        "GRASP_BBOX_INDEX_ROOT": str(data_index_root / "bbox"),
        "GRASP_VCOT_INDEX_ROOT": str(data_index_root / "vcot"),
        "GRASP_VCOT_DEFAULT_INDEX_ROOT": str(default_vcot_index_root),
        "GRASP_DIRECT_META_PATH": str(data_index_root / "direct/internvl_meta_train.json"),
        "GRASP_CROP_META_PATH": str(default_crop_index_root / "internvl_meta_train.json"),
        "GRASP_VCOT_META_PATH": str(default_vcot_index_root / "internvl_meta_train.json"),
        "GRASP_DIRECT_RUN_ROOT": str(run_root / "direct"),
        "GRASP_CROP_RUN_ROOT": str(run_root / "crop"),
        "GRASP_VCOT_RUN_ROOT": str(run_root / "vcot"),
        "GRASP_DIRECT_RESULT_ROOT": str(result_root / "direct"),
        "GRASP_CROP_RESULT_ROOT": str(result_root / "crop"),
        "GRASP_VCOT_RESULT_ROOT": str(result_root / "vcot"),
        "GRASP_ANALYSIS_ALL_ROOT": str(analysis_root / "all"),
        "GRASP_ANALYSIS_DIRECT_ROOT": str(analysis_root / "direct_grasp"),
        "GRASP_ANALYSIS_ORACLE_CROP_ROOT": str(analysis_root / "oracle_crop"),
        "GRASP_ANALYSIS_PREDICTED_VCOT_ROOT": str(analysis_root / "predicted_vcot"),
        "GRASP_PYTHON_BIN": env("GRASP_PYTHON_BIN", PYTHON_BIN),
        "GRASP_TRAIN_GPUS": env("GRASP_TRAIN_GPUS", TRAIN_GPUS),
        "GRASP_EVAL_GPUS": env("GRASP_EVAL_GPUS", EVAL_GPUS),
        "GRASP_BATCH_SIZE": env("GRASP_BATCH_SIZE", BATCH_SIZE),
        "GRASP_PER_DEVICE_BATCH_SIZE": env("GRASP_PER_DEVICE_BATCH_SIZE", PER_DEVICE_BATCH_SIZE),
        "GRASP_FORCE_IMAGE_SIZE": env("GRASP_FORCE_IMAGE_SIZE", FORCE_IMAGE_SIZE),
        "GRASP_EVAL_DATASETS": env("GRASP_EVAL_DATASETS", EVAL_DATASETS),
        "GRASP_RUN_EVAL": env("GRASP_RUN_EVAL", RUN_EVAL),
        "GRASP_OVERWRITE_EVAL_RESULTS": env("GRASP_OVERWRITE_EVAL_RESULTS", OVERWRITE_EVAL_RESULTS),
        "GRASP_EVAL_VCOT_IOU_THRESHOLD": env("GRASP_EVAL_VCOT_IOU_THRESHOLD", EVAL_VCOT_IOU_THRESHOLD),
        "GRASP_EVAL_VCOT_ANGLE_THRESHOLD": env("GRASP_EVAL_VCOT_ANGLE_THRESHOLD", EVAL_VCOT_ANGLE_THRESHOLD),
        "GRASP_DIRECT_CUDA_VISIBLE_DEVICES": env("GRASP_DIRECT_CUDA_VISIBLE_DEVICES", DIRECT_CUDA_VISIBLE_DEVICES),
        "GRASP_CROP_CUDA_VISIBLE_DEVICES": env("GRASP_CROP_CUDA_VISIBLE_DEVICES", CROP_CUDA_VISIBLE_DEVICES),
        "GRASP_VCOT_CUDA_VISIBLE_DEVICES": env("GRASP_VCOT_CUDA_VISIBLE_DEVICES", VCOT_CUDA_VISIBLE_DEVICES),
        "GRASP_DIRECT_EVAL_CUDA_VISIBLE_DEVICES": env("GRASP_DIRECT_EVAL_CUDA_VISIBLE_DEVICES", str(DIRECT_EVAL_CUDA_VISIBLE_DEVICES)),
        "GRASP_CROP_EVAL_CUDA_VISIBLE_DEVICES": env("GRASP_CROP_EVAL_CUDA_VISIBLE_DEVICES", str(CROP_EVAL_CUDA_VISIBLE_DEVICES)),
        "GRASP_VCOT_EVAL_CUDA_VISIBLE_DEVICES": env("GRASP_VCOT_EVAL_CUDA_VISIBLE_DEVICES", str(VCOT_EVAL_CUDA_VISIBLE_DEVICES)),
        "GRASP_DIRECT_EVAL_MASTER_PORT": env("GRASP_DIRECT_EVAL_MASTER_PORT", DIRECT_EVAL_MASTER_PORT),
        "GRASP_CROP_EVAL_MASTER_PORT": env("GRASP_CROP_EVAL_MASTER_PORT", CROP_EVAL_MASTER_PORT),
        "GRASP_VCOT_EVAL_MASTER_PORT": env("GRASP_VCOT_EVAL_MASTER_PORT", VCOT_EVAL_MASTER_PORT),
        "GRASP_DIRECT_USE_LLM_LORA": direct_use_lora,
        "GRASP_DIRECT_LEARNING_RATE": direct_lr,
        "GRASP_DIRECT_NUM_TRAIN_EPOCHS": direct_epochs,
        "GRASP_DIRECT_MAX_DYNAMIC_PATCH": direct_patch,
        "GRASP_CROP_USE_LLM_LORA": crop_use_lora,
        "GRASP_CROP_LEARNING_RATE": crop_lr,
        "GRASP_CROP_NUM_TRAIN_EPOCHS": crop_epochs,
        "GRASP_CROP_BBOX_EDGE_EXPAND": env("GRASP_CROP_BBOX_EDGE_EXPAND", CROP_EXPERIMENTS[0][4]),
        "GRASP_CROP_MIN_BBOX_HALF_SIZE": env("GRASP_CROP_MIN_BBOX_HALF_SIZE", CROP_EXPERIMENTS[0][5]),
        "GRASP_CROP_TARGET_COORDINATE_FRAME": env("GRASP_CROP_TARGET_COORDINATE_FRAME", CROP_EXPERIMENTS[0][6]),
        "GRASP_CROP_MAX_DYNAMIC_PATCH": env("GRASP_CROP_MAX_DYNAMIC_PATCH", CROP_EXPERIMENTS[0][7]),
        "GRASP_CROP_TARGET_GRASP_INDEX": crop_target_index,
        "GRASP_VCOT_USE_LLM_LORA": vcot_use_lora,
        "GRASP_VCOT_LEARNING_RATE": vcot_lr,
        "GRASP_VCOT_NUM_TRAIN_EPOCHS": vcot_epochs,
        "GRASP_VCOT_BBOX_RATIO": env("GRASP_VCOT_BBOX_RATIO", VCOT_EXPERIMENTS[0][4]),
        "GRASP_VCOT_BBOX_LOSS_WEIGHT": env("GRASP_VCOT_BBOX_LOSS_WEIGHT", VCOT_EXPERIMENTS[0][5]),
        "GRASP_VCOT_BBOX_EDGE_EXPAND": env("GRASP_VCOT_BBOX_EDGE_EXPAND", VCOT_EXPERIMENTS[0][6]),
        "GRASP_VCOT_MIN_BBOX_HALF_SIZE": env("GRASP_VCOT_MIN_BBOX_HALF_SIZE", VCOT_EXPERIMENTS[0][7]),
        "GRASP_VCOT_TARGET_COORDINATE_FRAME": env("GRASP_VCOT_TARGET_COORDINATE_FRAME", VCOT_EXPERIMENTS[0][8]),
        "GRASP_VCOT_MAX_DYNAMIC_PATCH": env("GRASP_VCOT_MAX_DYNAMIC_PATCH", VCOT_EXPERIMENTS[0][9]),
        "GRASP_VCOT_TARGET_GRASP_INDEX": vcot_target_index,
    }
    return Settings(values)


def experiment_rows(kind: str, settings: Settings) -> list[tuple[str, ...]]:
    if kind == "direct":
        return DIRECT_EXPERIMENTS
    if kind == "crop":
        return CROP_EXPERIMENTS
    if kind == "vcot":
        return VCOT_EXPERIMENTS
    raise ValueError(kind)


def print_shell_env(settings: Settings) -> None:
    for key, value in settings.values.items():
        print(f"export {key}={shlex.quote(value)}")


def validate(settings: Settings) -> None:
    split_root = Path(settings["GRASP_DATASET_ROOT"]) / "origin_split"
    if not split_root.is_dir():
        raise SystemExit(f"Invalid GRASP_DATASET_ROOT: expected {split_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Central grasp experiment settings.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("env")
    subparsers.add_parser("validate")
    for command in ["experiments", "default-name"]:
        sub = subparsers.add_parser(command)
        sub.add_argument("kind", choices=["direct", "crop", "vcot"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = build_settings()
    if args.command == "env":
        print_shell_env(settings)
    elif args.command == "experiments":
        for row in experiment_rows(args.kind, settings):
            print("\t".join(row))
    elif args.command == "default-name":
        print(experiment_rows(args.kind, settings)[0][0])
    elif args.command == "validate":
        validate(settings)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
