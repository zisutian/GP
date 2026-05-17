from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import lmdb
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_tools.vcot_crop_lmdb import (  # noqa: E402
    crop_box_from_bbox,
    mask_to_bbox_position,
    transform_grasp_from_crop_norm,
    transform_grasp_to_crop_norm,
)


DEFAULT_SUMMARY_CSV = REPO_ROOT / "rescore_result/all_methods_direct_grasp_oracle_crop_predicted_vcot/summary.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "rescore_result/all_methods_direct_grasp_oracle_crop_predicted_vcot/analysis"
DEFAULT_GRASP_LMDB = (REPO_ROOT / "../VCoT-Grasp-self/data/grasp_anything/lmdb/grasp_label_positive").resolve()
DEFAULT_MASK_LMDB = (REPO_ROOT / "../VCoT-Grasp-self/data/grasp_anything/lmdb/mask").resolve()
IMAGE_SIZE = 416
COORDS = ["x", "y", "w", "h", "angle"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write an oracle-crop summary diagnosis for the crop-frame constant-prior baseline."
    )
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--grasp-lmdb", default=str(DEFAULT_GRASP_LMDB))
    parser.add_argument("--mask-lmdb", default=str(DEFAULT_MASK_LMDB))
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--angle-threshold", type=float, default=30.0)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5000,
        help="Number of training manifest rows used to estimate the constant crop-frame prior. Use 0 for full train.",
    )
    return parser.parse_args()


def split_from_path(path: str) -> str:
    match = re.search(r"(test_seen|test_unseen)_", path)
    return match.group(1) if match else ""


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def csv_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def raw_crop_norm(grasp: list[float], crop_box: list[int]) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in crop_box]
    crop_w = max(1.0, x1 - x0)
    crop_h = max(1.0, y1 - y0)
    x, y, w, h, angle = [float(value) for value in grasp[:5]]
    return [
        (x - x0) / crop_w,
        (y - y0) / crop_h,
        w / crop_w,
        h / crop_h,
        angle / 180.0,
    ]


def clamp_flags(raw: list[float]) -> dict[str, bool]:
    return {
        "x_clamped": raw[0] < 0.0 or raw[0] > 1.0,
        "y_clamped": raw[1] < 0.0 or raw[1] > 1.0,
        "w_clamped": raw[2] < 0.0 or raw[2] > 1.0,
        "h_clamped": raw[3] < 0.0 or raw[3] > 1.0,
        "angle_clamped": raw[4] < 0.0 or raw[4] > 1.0,
    }


def any_xywh_clamped(flags: dict[str, bool]) -> bool:
    return flags["x_clamped"] or flags["y_clamped"] or flags["w_clamped"] or flags["h_clamped"]


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }
    array = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def prefixed_stats(prefix: str, values: list[float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in stats(values).items()}


def angle_diff_180(angle1: float, angle2: float) -> float:
    diff = abs(float(angle1) - float(angle2))
    return min(diff, 180.0 - diff)


def rotated_rect_iou(grasp1: list[float], grasp2: list[float]) -> float:
    rect1 = ((grasp1[0], grasp1[1]), (grasp1[2], grasp1[3]), grasp1[4])
    rect2 = ((grasp2[0], grasp2[1]), (grasp2[2], grasp2[3]), grasp2[4])
    box1 = cv2.boxPoints(rect1)
    box2 = cv2.boxPoints(rect2)
    intersection, _ = cv2.intersectConvexConvex(box1, box2)
    area1 = grasp1[2] * grasp1[3]
    area2 = grasp2[2] * grasp2[3]
    union = area1 + area2 - intersection
    return float(intersection / union) if union > 0 else 0.0


def official_success(pred: list[float], labels: list[list[float]], iou_t: float, angle_t: float) -> bool:
    for label in labels:
        if rotated_rect_iou(pred, label) >= iou_t and angle_diff_180(pred[4], label[4]) <= angle_t:
            return True
    return False


def top1_success(pred: list[float], labels: list[list[float]], iou_t: float, angle_t: float) -> bool:
    if not labels:
        return False
    label = labels[0]
    return rotated_rect_iou(pred, label) >= iou_t and angle_diff_180(pred[4], label[4]) <= angle_t


def vcot_like_grasp(grasp: list[float]) -> list[float]:
    return [int(value) for value in grasp[:4]] + [float(grasp[4])]


def load_labels(txn, grasp_id: str) -> list[list[float]]:
    value = txn.get(f"{grasp_id}.pt".encode("utf-8"))
    if value is None:
        raise FileNotFoundError(f"LMDB key not found: {grasp_id}.pt")
    labels = torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
    return [[float(value) for value in row[1:]] for row in labels]


def load_mask(txn, mask_key: str) -> np.ndarray:
    value = txn.get(mask_key.encode("utf-8"))
    if value is None and not mask_key.endswith(".npy"):
        value = txn.get(f"{mask_key}.npy".encode("utf-8"))
    if value is None:
        raise FileNotFoundError(f"LMDB key not found: {mask_key}")
    return np.load(io.BytesIO(value))


def crop_train_annotation(config: dict[str, Any]) -> Path:
    meta_path = Path(config["meta_path"])
    meta = read_json(meta_path)
    for entry in meta.values():
        if entry.get("vcot_dataset") == "grasp_anything_crop":
            return Path(entry["annotation"]).expanduser().resolve()
    raise ValueError(f"No grasp_anything_crop entry in {meta_path}")


def iter_manifest(path: Path, limit: int | None = None):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                return
            yield json.loads(line)


def compute_train_prior(
    config: dict[str, Any],
    mask_txn,
    grasp_txn,
    image_size: int,
    sample_limit: int | None = None,
) -> tuple[list[float], dict[str, Any]]:
    edge = int(config["bbox_edge_expand"])
    half = int(config["min_bbox_half_size"])
    target_index = int(config.get("target_grasp_index", 0))
    annotation = crop_train_annotation(config)

    values_by_coord: dict[str, list[float]] = {coord: [] for coord in COORDS}
    raw_by_coord: dict[str, list[float]] = {coord: [] for coord in COORDS}
    clamp_counts = {f"{coord}_clamped": 0 for coord in COORDS}
    any_clamped = 0
    count = 0

    limit = None if sample_limit == 0 else sample_limit
    for record in iter_manifest(annotation, limit=limit):
        labels = load_labels(grasp_txn, Path(record["grasp_key"]).stem)
        label_index = max(0, min(len(labels) - 1, target_index))
        grasp = labels[label_index]
        mask = load_mask(mask_txn, record["mask_key"])
        object_bbox = mask_to_bbox_position(mask)
        if object_bbox is None:
            continue
        crop_box = crop_box_from_bbox(object_bbox, image_size=image_size, min_half_size=half, edge_expand=edge)
        raw = raw_crop_norm(grasp, crop_box)
        crop_norm = transform_grasp_to_crop_norm(grasp, crop_box)
        flags = clamp_flags(raw)

        for coord, value, raw_value in zip(COORDS, crop_norm, raw):
            values_by_coord[coord].append(float(value))
            raw_by_coord[coord].append(float(raw_value))
            clamp_counts[f"{coord}_clamped"] += int(flags[f"{coord}_clamped"])
        any_clamped += int(any_xywh_clamped(flags))
        count += 1

    mean_vector = [float(np.mean(values_by_coord[coord])) for coord in COORDS]
    summary: dict[str, Any] = {
        "train_annotation": str(annotation),
        "train_count": count,
        "train_crop_any_xywh_clamped_rate": any_clamped / count if count else 0.0,
    }
    for coord in COORDS:
        summary[f"train_{coord}_mean_for_constant_baseline"] = mean_vector[COORDS.index(coord)]
        summary[f"train_{coord}_clamped_rate"] = clamp_counts[f"{coord}_clamped"] / count if count else 0.0
        summary.update(prefixed_stats(f"train_target_crop_{coord}", values_by_coord[coord]))
        summary.update(prefixed_stats(f"train_raw_crop_{coord}", raw_by_coord[coord]))
    return mean_vector, summary


def unique_oracle_results(summary_csv: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with summary_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("method") != "oracle_crop":
                continue
            result_path = Path(row["result_path"])
            split = split_from_path(str(result_path))
            key = (row["experiment"], split)
            if key in seen:
                continue
            seen.add(key)
            row["split"] = split
            rows.append(row)
    return rows


def diagnose_result(
    row: dict[str, Any],
    train_mean: list[float],
    train_summary: dict[str, Any],
    grasp_txn,
    image_size: int,
    iou_threshold: float,
    angle_threshold: float,
) -> dict[str, Any]:
    result_path = Path(row["result_path"])
    data = read_json(result_path)
    outputs = data.get("outputs", [])

    values_by_coord: dict[str, list[float]] = {coord: [] for coord in COORDS}
    raw_by_coord: dict[str, list[float]] = {coord: [] for coord in COORDS}
    clamp_counts = {f"{coord}_clamped": 0 for coord in COORDS}
    any_clamped = 0
    constant_official = 0
    constant_top1 = 0

    for output in outputs:
        crop_box = output["crop_box"]
        grasp = output["target_full_grasp"]
        raw = raw_crop_norm(grasp, crop_box)
        crop_norm = transform_grasp_to_crop_norm(grasp, crop_box)
        flags = clamp_flags(raw)
        pred_full = vcot_like_grasp(transform_grasp_from_crop_norm(train_mean, crop_box))
        labels = load_labels(grasp_txn, output["grasp_id"])
        const_official_success = official_success(pred_full, labels, iou_threshold, angle_threshold)
        const_top1_success = top1_success(pred_full, labels, iou_threshold, angle_threshold)
        constant_official += int(const_official_success)
        constant_top1 += int(const_top1_success)
        for coord, value, raw_value in zip(COORDS, crop_norm, raw):
            values_by_coord[coord].append(float(value))
            raw_by_coord[coord].append(float(raw_value))
            clamp_counts[f"{coord}_clamped"] += int(flags[f"{coord}_clamped"])
        any_clamped += int(any_xywh_clamped(flags))

    total = len(outputs)
    summary: dict[str, Any] = {
        "method": row["method"],
        "experiment": row["experiment"],
        "split": row["split"],
        "result_path": str(result_path),
        "target_coordinate_frame": row.get("target_coordinate_frame", ""),
        "bbox_edge_expand": row.get("bbox_edge_expand", ""),
        "min_bbox_half_size": row.get("min_bbox_half_size", ""),
        "target_grasp_index": row.get("target_grasp_index", ""),
        "total": total,
        "model_official_rate": csv_float(row.get("vcot_success_rate_all")),
        "model_top1_rate": csv_float(row.get("vcot_top1_success_rate_all")),
        "constant_train_mean_official_rate": constant_official / total if total else 0.0,
        "constant_train_mean_top1_rate": constant_top1 / total if total else 0.0,
        "constant_gap_model_minus_official": csv_float(row.get("vcot_success_rate_all")) - (constant_official / total if total else 0.0),
        "constant_gap_model_minus_top1": csv_float(row.get("vcot_top1_success_rate_all")) - (constant_top1 / total if total else 0.0),
        "test_crop_any_xywh_clamped_rate": any_clamped / total if total else 0.0,
        **train_summary,
    }
    for coord in COORDS:
        summary[f"test_{coord}_clamped_rate"] = clamp_counts[f"{coord}_clamped"] / total if total else 0.0
        summary.update(prefixed_stats(f"test_target_crop_{coord}", values_by_coord[coord]))
        summary.update(prefixed_stats(f"test_raw_crop_{coord}", raw_by_coord[coord]))
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_manifest(out_dir: Path, output_path: Path) -> None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = read_json(manifest_path)
    outputs = set(manifest.get("outputs", []))
    outputs.add(str(output_path.relative_to(out_dir)))
    manifest["outputs"] = sorted(outputs)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    rows = unique_oracle_results(summary_csv)

    grasp_env = lmdb.open(
        str(Path(args.grasp_lmdb)),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2048,
    )
    mask_env = lmdb.open(
        str(Path(args.mask_lmdb)),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2048,
    )

    train_cache: dict[str, tuple[list[float], dict[str, Any]]] = {}
    summary_rows = []
    with grasp_env.begin() as grasp_txn, mask_env.begin() as mask_txn:
        for row in rows:
            config_path = Path(row["loaded_vcot_config"])
            config = read_json(config_path)
            cache_key = json.dumps({
                "meta_path": config["meta_path"],
                "bbox_edge_expand": config["bbox_edge_expand"],
                "min_bbox_half_size": config["min_bbox_half_size"],
                "target_grasp_index": config.get("target_grasp_index", 0),
            }, sort_keys=True)
            if cache_key not in train_cache:
                train_cache[cache_key] = compute_train_prior(
                    config=config,
                    mask_txn=mask_txn,
                    grasp_txn=grasp_txn,
                    image_size=args.image_size,
                    sample_limit=args.sample_limit,
                )
            train_mean, train_summary = train_cache[cache_key]
            result_summary = diagnose_result(
                row=row,
                train_mean=train_mean,
                train_summary=train_summary,
                grasp_txn=grasp_txn,
                image_size=args.image_size,
                iou_threshold=args.iou_threshold,
                angle_threshold=args.angle_threshold,
            )
            summary_rows.append(result_summary)

    summary_path = out_dir / "oracle_crop" / "crop_frame_prior_summary.csv"
    write_csv(summary_path, summary_rows)
    update_manifest(out_dir, summary_path)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
