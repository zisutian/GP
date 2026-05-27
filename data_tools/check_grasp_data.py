from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_tools.vcot_crop_lmdb import TARGET_FRAME_CROP_IMAGE, TARGET_FRAME_FULL_IMAGE  # noqa: E402
from grasp_settings import build_settings, experiment_rows  # noqa: E402


SETTINGS = build_settings()


def existing_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {label}: {path}")


def read_json(path: Path) -> dict[str, Any]:
    existing_file(path, "JSON file")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_annotation(meta_path: Path, annotation: str) -> Path:
    path = Path(annotation).expanduser()
    if not path.is_absolute():
        path = meta_path.parent / path
    return path.resolve()


def check_meta(meta_path: Path) -> None:
    meta = read_json(meta_path)
    missing: list[str] = []
    for name, entry in meta.items():
        if not isinstance(entry, dict):
            continue
        annotation = entry.get("annotation")
        if not annotation:
            continue
        path = resolve_annotation(meta_path, str(annotation))
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(f"{name}: {path}")
    if missing:
        raise FileNotFoundError("Missing meta annotation file(s):\n" + "\n".join(missing))


def check_splits(data_index_root: Path, splits: list[str]) -> None:
    for split in splits:
        existing_file(data_index_root / f"{split}.jsonl", "data index")


def first_jsonl(path: Path) -> dict[str, Any]:
    existing_file(path, "data index")
    with path.open("r", encoding="utf-8") as f:
        line = f.readline()
    if not line:
        raise ValueError(f"Empty data index: {path}")
    return json.loads(line)


def check_crop_config(
    path: Path,
    bbox_edge_expand: int,
    min_bbox_half_size: int,
    target_coordinate_frame: str,
    target_grasp_index: int,
) -> None:
    first = first_jsonl(path)
    expected = {
        "bbox_edge_expand": bbox_edge_expand,
        "min_bbox_half_size": min_bbox_half_size,
        "target_coordinate_frame": target_coordinate_frame,
        "target_grasp_index": target_grasp_index,
    }
    errors = [
        f"{key}: expected {value!r}, got {first.get(key)!r}"
        for key, value in expected.items()
        if first.get(key) != value
    ]
    if errors:
        raise ValueError(f"Crop data index config mismatch: {path}\n" + "\n".join(errors))


def vcot_bbox_meta(meta_path: Path) -> dict[str, Any]:
    meta = read_json(meta_path)
    return next(
        (entry for entry in meta.values() if entry.get("vcot_dataset") == "grasp_anything_bbox"),
        {},
    )


def float_matches(actual: Any, expected: float) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= 1e-9
    except (TypeError, ValueError):
        return False


def check_vcot_bbox_ratio(meta_path: Path, expected_bbox_ratio: float | None) -> None:
    if expected_bbox_ratio is None:
        return
    bbox_meta = vcot_bbox_meta(meta_path)
    expected_repeat_time = max(0.0, min(1.0, float(expected_bbox_ratio)))
    actual = bbox_meta.get("repeat_time")
    if not float_matches(actual, expected_repeat_time):
        raise ValueError(
            f"VCoT bbox ratio mismatch: {meta_path}\n"
            f"expected repeat_time {expected_repeat_time!r} from bbox_ratio {expected_bbox_ratio!r}, got {actual!r}"
        )


def check_vcot_loss_weight(meta_path: Path, expected_bbox_loss_weight: float | None) -> None:
    if expected_bbox_loss_weight is None:
        return
    bbox_meta = vcot_bbox_meta(meta_path)
    actual = bbox_meta.get("vcot_loss_weight")
    if actual is None and float(expected_bbox_loss_weight) == 1.0:
        return
    if not float_matches(actual, float(expected_bbox_loss_weight)):
        raise ValueError(
            f"VCoT bbox loss weight mismatch: {meta_path}\n"
            f"expected {expected_bbox_loss_weight!r}, got {actual!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check existing Grasp-Anything data_index/meta files without generating them.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    direct = subparsers.add_parser("direct")
    direct.add_argument("--meta-path", default=SETTINGS["GRASP_DIRECT_META_PATH"])
    direct.add_argument("--data-index-root", default=SETTINGS["GRASP_DIRECT_INDEX_ROOT"])
    direct.add_argument("--splits", nargs="+", default=["train"])

    bbox = subparsers.add_parser("bbox")
    bbox.add_argument("--meta-path", default=str(Path(SETTINGS["GRASP_BBOX_INDEX_ROOT"]) / "internvl_meta_train.json"))
    bbox.add_argument("--data-index-root", default=SETTINGS["GRASP_BBOX_INDEX_ROOT"])
    bbox.add_argument("--splits", nargs="+", default=["train"])

    crop = subparsers.add_parser("crop")
    crop.add_argument("--meta-path", required=True)
    crop.add_argument("--data-index-root", required=True)
    crop.add_argument("--splits", nargs="+", default=["train"])
    crop.add_argument("--bbox-edge-expand", type=int, required=True)
    crop.add_argument("--min-bbox-half-size", type=int, required=True)
    crop.add_argument("--target-coordinate-frame", choices=[TARGET_FRAME_FULL_IMAGE, TARGET_FRAME_CROP_IMAGE], required=True)
    crop.add_argument("--target-grasp-index", type=int, required=True)

    vcot = subparsers.add_parser("vcot")
    vcot.add_argument("--meta-path", required=True)
    vcot.add_argument("--data-index-root", required=True)
    vcot.add_argument("--crop-root", required=True)
    vcot.add_argument("--bbox-root", default=SETTINGS["GRASP_BBOX_INDEX_ROOT"])
    vcot.add_argument("--eval-splits", nargs="+", default=[])
    vcot.add_argument("--bbox-ratio", type=float, default=None)
    vcot.add_argument("--bbox-loss-weight", type=float, default=None)
    vcot.add_argument("--bbox-edge-expand", type=int, required=True)
    vcot.add_argument("--min-bbox-half-size", type=int, required=True)
    vcot.add_argument("--target-coordinate-frame", choices=[TARGET_FRAME_FULL_IMAGE, TARGET_FRAME_CROP_IMAGE], required=True)
    vcot.add_argument("--target-grasp-index", type=int, required=True)

    all_data = subparsers.add_parser("all")
    all_data.add_argument("--splits", nargs="+", default=["train", "test_seen", "test_unseen"])
    all_data.add_argument("--eval-splits", nargs="+", default=["test_seen", "test_unseen"])

    return parser.parse_args()


def check_direct(meta_path: str | Path, data_index_root: str | Path, splits: list[str]) -> None:
    meta = Path(meta_path).expanduser().resolve()
    root = Path(data_index_root).expanduser().resolve()
    check_meta(meta)
    check_splits(root, splits)


def check_bbox(meta_path: str | Path, data_index_root: str | Path, splits: list[str]) -> None:
    meta = Path(meta_path).expanduser().resolve()
    root = Path(data_index_root).expanduser().resolve()
    check_meta(meta)
    check_splits(root, splits)


def check_crop(
    meta_path: str | Path,
    data_index_root: str | Path,
    splits: list[str],
    bbox_edge_expand: int,
    min_bbox_half_size: int,
    target_coordinate_frame: str,
    target_grasp_index: int,
) -> None:
    meta = Path(meta_path).expanduser().resolve()
    root = Path(data_index_root).expanduser().resolve()
    check_meta(meta)
    check_splits(root, splits)
    for split in splits:
        check_crop_config(
            root / f"{split}.jsonl",
            bbox_edge_expand,
            min_bbox_half_size,
            target_coordinate_frame,
            target_grasp_index,
        )


def check_vcot(
    meta_path: str | Path,
    data_index_root: str | Path,
    crop_root: str | Path,
    bbox_root: str | Path,
    eval_splits: list[str],
    bbox_edge_expand: int,
    min_bbox_half_size: int,
    target_coordinate_frame: str,
    target_grasp_index: int,
    bbox_ratio: float | None = None,
    bbox_loss_weight: float | None = None,
) -> None:
    meta = Path(meta_path).expanduser().resolve()
    root = Path(data_index_root).expanduser().resolve()
    crop = Path(crop_root).expanduser().resolve()
    bbox = Path(bbox_root).expanduser().resolve()
    check_meta(meta)
    check_vcot_bbox_ratio(meta, bbox_ratio)
    check_vcot_loss_weight(meta, bbox_loss_weight)
    check_splits(root, eval_splits)
    check_crop_config(
        crop / "train.jsonl",
        bbox_edge_expand,
        min_bbox_half_size,
        target_coordinate_frame,
        target_grasp_index,
    )
    existing_file(bbox / "train.jsonl", "bbox train data index")


def check_all(splits: list[str], eval_splits: list[str]) -> None:
    check_direct(SETTINGS["GRASP_DIRECT_META_PATH"], SETTINGS["GRASP_DIRECT_INDEX_ROOT"], splits)
    check_bbox(
        Path(SETTINGS["GRASP_BBOX_INDEX_ROOT"]) / "internvl_meta_train.json",
        SETTINGS["GRASP_BBOX_INDEX_ROOT"],
        splits,
    )
    for row in experiment_rows("crop", SETTINGS):
        name, _use_lora, _lr, _epochs, edge_expand, min_half, target_frame, _patch, target_grasp_index = row
        index_root = Path(SETTINGS["GRASP_CROP_INDEX_ROOT"]) / name
        check_crop(
            index_root / "internvl_meta_train.json",
            index_root,
            splits,
            int(edge_expand),
            int(min_half),
            target_frame,
            int(target_grasp_index),
        )
    for row in experiment_rows("vcot", SETTINGS):
        (
            name,
            _use_lora,
            _lr,
            _epochs,
            _bbox_ratio,
            bbox_loss_weight,
            edge_expand,
            min_half,
            target_frame,
            _patch,
            target_grasp_index,
        ) = row
        index_root = Path(SETTINGS["GRASP_VCOT_INDEX_ROOT"]) / name
        check_vcot(
            index_root / "internvl_meta_train.json",
            index_root,
            index_root / "crop",
            SETTINGS["GRASP_BBOX_INDEX_ROOT"],
            eval_splits,
            int(edge_expand),
            int(min_half),
            target_frame,
            int(target_grasp_index),
            float(_bbox_ratio),
            float(bbox_loss_weight),
        )


def main() -> None:
    args = parse_args()
    if args.command == "direct":
        check_direct(args.meta_path, args.data_index_root, args.splits)
    elif args.command == "bbox":
        check_bbox(args.meta_path, args.data_index_root, args.splits)
    elif args.command == "crop":
        check_crop(
            args.meta_path,
            args.data_index_root,
            args.splits,
            args.bbox_edge_expand,
            args.min_bbox_half_size,
            args.target_coordinate_frame,
            args.target_grasp_index,
        )
    elif args.command == "vcot":
        check_vcot(
            args.meta_path,
            args.data_index_root,
            args.crop_root,
            args.bbox_root,
            args.eval_splits,
            args.bbox_edge_expand,
            args.min_bbox_half_size,
            args.target_coordinate_frame,
            args.target_grasp_index,
            args.bbox_ratio,
            args.bbox_loss_weight,
        )
    elif args.command == "all":
        check_all(args.splits, args.eval_splits)
    else:
        raise ValueError(args.command)
    print("data_check=ok")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(
            "data_check=failed\n"
            f"{exc}\n"
            "Generate data indexes explicitly with: python data_tools/ensure_grasp_data.py all"
        ) from None
