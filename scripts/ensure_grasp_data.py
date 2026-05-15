from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_tools.prepare_grasp_anything_bbox import write_split as write_bbox_split
from data_tools.prepare_grasp_anything_crop import write_split as write_crop_split
from data_tools.prepare_grasp_anything_direct import write_split as write_direct_split
from data_tools.vcot_crop_lmdb import DEFAULT_BBOX_EDGE_EXPAND, DEFAULT_MIN_BBOX_HALF_SIZE


DEFAULT_SOURCE_ROOT = (REPO_ROOT / "../VCoT-Grasp-self/data/grasp_anything").resolve()
DEFAULT_DIRECT_META = REPO_ROOT / "data/vcot_grasp/direct/internvl_meta_train.json"
DEFAULT_CROP_META = REPO_ROOT / "data/vcot_grasp/crop/internvl_meta_train.json"
DEFAULT_VCOT_META = REPO_ROOT / "data/vcot_grasp/vcot/internvl_meta_train.json"
DEFAULT_BBOX_ROOT = REPO_ROOT / "data/vcot_grasp/bbox"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure Grasp-Anything manifest/meta files exist.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    direct = subparsers.add_parser("direct", help="Ensure direct-grasp manifests and train meta.")
    direct.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    direct.add_argument("--meta-path", default=str(DEFAULT_DIRECT_META))
    direct.add_argument("--output-root", default=None)
    direct.add_argument("--splits", nargs="+", default=["train"])

    crop = subparsers.add_parser("crop", help="Ensure oracle-crop manifests and train meta.")
    crop.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    crop.add_argument("--meta-path", default=str(DEFAULT_CROP_META))
    crop.add_argument("--output-root", default=None)
    crop.add_argument("--splits", nargs="+", default=["train"])
    crop.add_argument("--bbox-edge-expand", type=int, default=DEFAULT_BBOX_EDGE_EXPAND)
    crop.add_argument("--min-bbox-half-size", type=int, default=DEFAULT_MIN_BBOX_HALF_SIZE)
    crop.add_argument("--target-coordinate-frame", choices=["full_image", "crop_image"], default="full_image")
    crop.add_argument("--target-grasp-index", type=int, default=0)

    vcot = subparsers.add_parser("vcot", help="Ensure VCoT joint train meta and eval manifests.")
    vcot.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    vcot.add_argument("--meta-path", default=str(DEFAULT_VCOT_META))
    vcot.add_argument("--output-root", default=None)
    vcot.add_argument("--crop-root", default=None)
    vcot.add_argument("--bbox-root", default=str(DEFAULT_BBOX_ROOT))
    vcot.add_argument("--eval-splits", nargs="+", default=["test_seen", "test_unseen"])
    vcot.add_argument("--bbox-ratio", type=float, default=0.5)
    vcot.add_argument("--bbox-edge-expand", type=int, default=DEFAULT_BBOX_EDGE_EXPAND)
    vcot.add_argument("--min-bbox-half-size", type=int, default=DEFAULT_MIN_BBOX_HALF_SIZE)
    vcot.add_argument("--target-coordinate-frame", choices=["full_image", "crop_image"], default="full_image")
    vcot.add_argument("--target-grasp-index", type=int, default=0)

    return parser.parse_args()


def normalize_names(values: Iterable[str]) -> list[str]:
    names: list[str] = []
    for value in values:
        for name in value.split(","):
            name = name.strip()
            if name and name not in names:
                names.append(name)
    return names


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def exists_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def read_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def annotation_paths(meta_path: Path) -> list[Path]:
    paths: list[Path] = []
    for entry in read_meta(meta_path).values():
        annotation = entry.get("annotation")
        if annotation:
            paths.append(Path(annotation).expanduser().resolve())
    return paths


def missing_annotations(meta_path: Path) -> list[Path]:
    return [path for path in annotation_paths(meta_path) if not exists_nonempty(path)]


def ensure_split_name(split: str) -> None:
    if "/" in split or "\\" in split or split.endswith(".jsonl"):
        raise ValueError(
            f"Cannot auto-create non-standard dataset reference: {split}. "
            "Pass a normal split name such as train, test_seen, or test_unseen."
        )


def crop_manifest_matches_config(
    path: Path,
    bbox_edge_expand: int,
    min_bbox_half_size: int,
    target_coordinate_frame: str,
    target_grasp_index: int,
) -> bool:
    if not exists_nonempty(path):
        return False
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
    if not first_line:
        return False
    first = json.loads(first_line)
    return (
        first.get("bbox_edge_expand") == bbox_edge_expand
        and first.get("min_bbox_half_size") == min_bbox_half_size
        and first.get("target_coordinate_frame") == target_coordinate_frame
        and first.get("target_grasp_index") == target_grasp_index
    )


def require_crop_manifest_config(
    path: Path,
    bbox_edge_expand: int,
    min_bbox_half_size: int,
    target_coordinate_frame: str,
    target_grasp_index: int,
) -> None:
    if exists_nonempty(path) and not crop_manifest_matches_config(
        path,
        bbox_edge_expand,
        min_bbox_half_size,
        target_coordinate_frame,
        target_grasp_index,
    ):
        raise ValueError(
            f"Existing crop manifest does not match requested config: {path}. "
            "Use a separate output root or regenerate it explicitly."
        )


def write_direct_meta(meta_path: Path, output_root: Path, train_length: int) -> None:
    train_meta = {
        "grasp_anything_direct_train": {
            "root": "",
            "annotation": str(output_root / "train.jsonl"),
            "data_augment": False,
            "repeat_time": 1,
            "length": train_length,
            "vcot_dataset": "grasp_anything_direct",
            "vcot_image_size": 416,
        }
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(train_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"meta: {meta_path}")


def write_crop_meta(
    meta_path: Path,
    output_root: Path,
    train_length: int,
    bbox_edge_expand: int,
    min_bbox_half_size: int,
    target_coordinate_frame: str,
    target_grasp_index: int,
) -> None:
    train_meta = {
        "grasp_anything_crop_train": {
            "root": "",
            "annotation": str(output_root / "train.jsonl"),
            "data_augment": False,
            "repeat_time": 1,
            "length": train_length,
            "vcot_dataset": "grasp_anything_crop",
            "vcot_image_size": 416,
            "vcot_crop_source": "gt_mask_object_bbox",
            "vcot_target_coordinate_frame": target_coordinate_frame,
            "vcot_bbox_edge_expand": bbox_edge_expand,
            "vcot_min_bbox_half_size": min_bbox_half_size,
            "vcot_target_grasp_index": target_grasp_index,
        }
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(train_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"meta: {meta_path}")


def ensure_direct(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).expanduser().resolve()
    meta_path = Path(args.meta_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else meta_path.parent
    splits = normalize_names(["train", *args.splits])

    missing_from_meta = {path.stem for path in missing_annotations(meta_path)}
    for split in normalize_names([*splits, *missing_from_meta]):
        ensure_split_name(split)
        output_path = output_root / f"{split}.jsonl"
        if not exists_nonempty(output_path):
            written = write_direct_split(source_root, output_path, split, limit=None)
            print(f"{split}: wrote {written} rows -> {output_path}")

    train_length = count_jsonl(output_root / "train.jsonl")
    write_direct_meta(meta_path, output_root, train_length)


def ensure_crop(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).expanduser().resolve()
    meta_path = Path(args.meta_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else meta_path.parent
    splits = normalize_names(["train", *args.splits])

    missing_from_meta = {path.stem for path in missing_annotations(meta_path)}
    for split in normalize_names([*splits, *missing_from_meta]):
        ensure_split_name(split)
        output_path = output_root / f"{split}.jsonl"
        require_crop_manifest_config(
            output_path,
            args.bbox_edge_expand,
            args.min_bbox_half_size,
            args.target_coordinate_frame,
            args.target_grasp_index,
        )
        if not exists_nonempty(output_path):
            written = write_crop_split(
                source_root=source_root,
                output_path=output_path,
                split=split,
                bbox_edge_expand=args.bbox_edge_expand,
                min_bbox_half_size=args.min_bbox_half_size,
                target_coordinate_frame=args.target_coordinate_frame,
                target_grasp_index=args.target_grasp_index,
                limit=None,
            )
            print(f"{split}: wrote {written} rows -> {output_path}")

    train_length = count_jsonl(output_root / "train.jsonl")
    write_crop_meta(
        meta_path,
        output_root,
        train_length,
        args.bbox_edge_expand,
        args.min_bbox_half_size,
        args.target_coordinate_frame,
        args.target_grasp_index,
    )


def write_vcot_meta(
    meta_path: Path,
    crop_train: Path,
    bbox_train: Path,
    crop_length: int,
    bbox_length: int,
    bbox_ratio: float,
    bbox_edge_expand: int,
    min_bbox_half_size: int,
    target_coordinate_frame: str,
    target_grasp_index: int,
) -> None:
    bbox_repeat_time = max(0.0, min(1.0, float(bbox_ratio)))
    train_meta = {
        "grasp_anything_crop_train": {
            "root": "",
            "annotation": str(crop_train),
            "data_augment": False,
            "repeat_time": 1,
            "length": crop_length,
            "vcot_dataset": "grasp_anything_crop",
            "vcot_image_size": 416,
            "vcot_crop_source": "gt_mask_object_bbox",
            "vcot_target_coordinate_frame": target_coordinate_frame,
            "vcot_bbox_edge_expand": bbox_edge_expand,
            "vcot_min_bbox_half_size": min_bbox_half_size,
            "vcot_target_grasp_index": target_grasp_index,
        },
        "grasp_anything_bbox_train": {
            "root": "",
            "annotation": str(bbox_train),
            "data_augment": False,
            "repeat_time": bbox_repeat_time,
            "length": int(bbox_length * bbox_repeat_time),
            "vcot_dataset": "grasp_anything_bbox",
            "vcot_image_size": 416,
            "vcot_target_coordinate_frame": "full_image",
            "vcot_target_bbox_order": "xyxy",
        },
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(train_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"meta: {meta_path}")


def ensure_vcot(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).expanduser().resolve()
    meta_path = Path(args.meta_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else meta_path.parent
    crop_root = Path(args.crop_root).expanduser().resolve() if args.crop_root else output_root / "crop"
    bbox_root = Path(args.bbox_root).expanduser().resolve()
    eval_splits = normalize_names(args.eval_splits)

    crop_train = crop_root / "train.jsonl"
    bbox_train = bbox_root / "train.jsonl"

    require_crop_manifest_config(
        crop_train,
        args.bbox_edge_expand,
        args.min_bbox_half_size,
        args.target_coordinate_frame,
        args.target_grasp_index,
    )
    if not exists_nonempty(crop_train):
        written = write_crop_split(
            source_root=source_root,
            output_path=crop_train,
            split="train",
            bbox_edge_expand=args.bbox_edge_expand,
            min_bbox_half_size=args.min_bbox_half_size,
            target_coordinate_frame=args.target_coordinate_frame,
            target_grasp_index=args.target_grasp_index,
            limit=None,
        )
        print(f"crop train: wrote {written} rows -> {crop_train}")

    if not exists_nonempty(bbox_train):
        written = write_bbox_split(source_root, bbox_train, "train", limit=None)
        print(f"bbox train: wrote {written} rows -> {bbox_train}")

    for split in eval_splits:
        ensure_split_name(split)
        output_path = output_root / f"{split}.jsonl"
        require_crop_manifest_config(
            output_path,
            args.bbox_edge_expand,
            args.min_bbox_half_size,
            args.target_coordinate_frame,
            args.target_grasp_index,
        )
        if not exists_nonempty(output_path):
            written = write_crop_split(
                source_root=source_root,
                output_path=output_path,
                split=split,
                bbox_edge_expand=args.bbox_edge_expand,
                min_bbox_half_size=args.min_bbox_half_size,
                target_coordinate_frame=args.target_coordinate_frame,
                target_grasp_index=args.target_grasp_index,
                limit=None,
            )
            print(f"{split}: wrote {written} rows -> {output_path}")

    crop_length = count_jsonl(crop_train)
    bbox_length = count_jsonl(bbox_train)
    write_vcot_meta(
        meta_path,
        crop_train,
        bbox_train,
        crop_length,
        bbox_length,
        args.bbox_ratio,
        args.bbox_edge_expand,
        args.min_bbox_half_size,
        args.target_coordinate_frame,
        args.target_grasp_index,
    )


def main() -> None:
    args = parse_args()
    if args.command == "direct":
        ensure_direct(args)
    elif args.command == "crop":
        ensure_crop(args)
    elif args.command == "vcot":
        ensure_vcot(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
