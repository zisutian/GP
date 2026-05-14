from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import lmdb
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRASP_LMDB = (REPO_ROOT / "../VCoT-Grasp-self/data/grasp_anything/lmdb/grasp_label_positive").resolve()
DEFAULT_MASK_LMDB = (REPO_ROOT / "../VCoT-Grasp-self/data/grasp_anything/lmdb/mask").resolve()
IMAGE_SIZE = 416

GEOMETRY_LEVELS = {
    "strict_10": (10.0, 10.0, 10.0),
    "medium_20": (20.0, 20.0, 20.0),
    "loose_30": (30.0, 30.0, 30.0),
}
CENTER_THRESHOLDS = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
ANGLE_THRESHOLDS = [5.0, 10.0, 15.0, 20.0, 30.0]
IOU_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


@dataclass
class ResultInfo:
    result_id: str
    method: str
    experiment: str
    split: str
    path: Path
    total: int
    parsed: int


def parse_args():
    parser = argparse.ArgumentParser(description="Create enhanced direct/crop grasp evaluation tables.")
    parser.add_argument("results", nargs="+", help="Result JSON files from direct/crop evaluation or rescoring.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--grasp-lmdb", default=str(DEFAULT_GRASP_LMDB))
    parser.add_argument("--mask-lmdb", default=str(DEFAULT_MASK_LMDB))
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--angle-threshold", type=float, default=30.0)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    return parser.parse_args()


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


def denormalize_grasp_like_vcot(values: list[float], image_size: int) -> list[float]:
    return [int(float(value) * image_size) for value in values[:4]] + [float(values[4]) * 180.0]


def load_labels(txn, grasp_id: str) -> list[list[float]]:
    value = txn.get(f"{grasp_id}.pt".encode("utf-8"))
    if value is None:
        raise FileNotFoundError(f"LMDB key not found: {grasp_id}.pt")
    labels = torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
    return [[float(value) for value in row[1:]] for row in labels]


def load_mask(txn, grasp_id: str) -> np.ndarray | None:
    value = txn.get(f"{grasp_id}.npy".encode("utf-8"))
    if value is None:
        return None
    return np.load(io.BytesIO(value))


def label_metric(pred: list[float], label: list[float]) -> dict[str, float]:
    center = float(np.linalg.norm(np.array(pred[:2], dtype=np.float32) - np.array(label[:2], dtype=np.float32)))
    wh = float(np.linalg.norm(np.array(pred[2:4], dtype=np.float32) - np.array(label[2:4], dtype=np.float32)))
    angle = angle_diff_180(pred[4], label[4])
    iou = rotated_rect_iou(pred, label)
    return {
        "center_xy_error_px": center,
        "width_height_error_px": wh,
        "circular_angle_error_deg": angle,
        "iou": iou,
    }


def geometry_success(metric: dict[str, float], center_t: float, wh_t: float, angle_t: float) -> bool:
    return (
        metric["center_xy_error_px"] <= center_t
        and metric["width_height_error_px"] <= wh_t
        and metric["circular_angle_error_deg"] <= angle_t
    )


def official_success(metric: dict[str, float], iou_t: float, angle_t: float) -> bool:
    return metric["iou"] >= iou_t and metric["circular_angle_error_deg"] <= angle_t


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0}
    array = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def infer_info(path: Path, total: int, parsed: int) -> ResultInfo:
    parts = path.parts
    if "vcot_grasp_direct" in parts:
        method = "direct"
        if "hparams" in parts:
            experiment = parts[parts.index("hparams") + 1]
        else:
            experiment = path.parent.name
    elif "vcot_grasp_vcot" in parts:
        method = "pred_vcot"
        experiment = path.parent.name if path.parent.name != "vcot_grasp_vcot" else "predicted_bbox_crop"
    elif "vcot_grasp_crop" in parts:
        method = "oracle_crop"
        experiment = path.parent.name if path.parent.name != "vcot_grasp_crop" else "oracle_object_crop"
    else:
        method = "unknown"
        experiment = path.parent.name

    if "test_unseen" in path.name:
        split = "test_unseen"
    elif "test_seen" in path.name:
        split = "test_seen"
    else:
        split = "unknown"
    return ResultInfo(
        result_id=f"{method}:{experiment}:{split}:{path.name}",
        method=method,
        experiment=experiment,
        split=split,
        path=path,
        total=total,
        parsed=parsed,
    )


def evaluate_output(
    output: dict[str, Any],
    labels: list[list[float]],
    iou_t: float,
    angle_t: float,
    image_size: int,
) -> dict[str, Any] | None:
    if output.get("pred_norm") is None:
        return None
    pred = denormalize_grasp_like_vcot(output["pred_norm"], image_size)
    metrics = [label_metric(pred, label) for label in labels]
    if not metrics:
        return None

    top1 = metrics[0]
    best_iou_index = max(range(len(metrics)), key=lambda idx: metrics[idx]["iou"])
    best_iou = metrics[best_iou_index]

    row = {
        "grasp_id": output["grasp_id"],
        "obj_name": output.get("obj_name", ""),
        "split": output.get("split", ""),
        "target_label_count": len(labels),
        "official_success": any(official_success(metric, iou_t, angle_t) for metric in metrics),
        "top1_success": official_success(top1, iou_t, angle_t),
        "top1_center_xy_error_px": top1["center_xy_error_px"],
        "top1_width_height_error_px": top1["width_height_error_px"],
        "top1_circular_angle_error_deg": top1["circular_angle_error_deg"],
        "top1_iou": top1["iou"],
        "best_iou_center_xy_error_px": best_iou["center_xy_error_px"],
        "best_iou_width_height_error_px": best_iou["width_height_error_px"],
        "best_iou_circular_angle_error_deg": best_iou["circular_angle_error_deg"],
        "max_iou_with_all_labels": best_iou["iou"],
        "min_center_xy_error_px": min(metric["center_xy_error_px"] for metric in metrics),
        "min_width_height_error_px": min(metric["width_height_error_px"] for metric in metrics),
        "min_circular_angle_error_deg": min(metric["circular_angle_error_deg"] for metric in metrics),
        "_label_metrics": metrics,
    }
    for name, (center_t, wh_t, angle_t_level) in GEOMETRY_LEVELS.items():
        row[f"top1_{name}"] = geometry_success(top1, center_t, wh_t, angle_t_level)
        row[f"all_labels_{name}"] = any(
            geometry_success(metric, center_t, wh_t, angle_t_level) for metric in metrics
        )
    return row


def read_result_file(
    path: Path,
    grasp_txn,
    args,
) -> tuple[ResultInfo, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    outputs = data.get("outputs", [])
    records = []
    for output in outputs:
        if output.get("pred_norm") is None:
            continue
        labels = load_labels(grasp_txn, output["grasp_id"])
        row = evaluate_output(output, labels, args.iou_threshold, args.angle_threshold, args.image_size)
        if row is None:
            continue
        row["crop_box"] = output.get("crop_box")
        records.append(row)
    return infer_info(path, total=len(outputs), parsed=len(records)), records


def rate(records: list[dict[str, Any]], key: str, total: int) -> float:
    if total <= 0:
        return 0.0
    return sum(int(bool(row.get(key))) for row in records) / total


def mean(records: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in records if key in row]
    return float(np.mean(values)) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_main_summary(out_dir: Path, grouped: dict[str, tuple[ResultInfo, list[dict[str, Any]]]]):
    rows = []
    for result_id, (info, records) in grouped.items():
        row = {
            "result_id": result_id,
            "method": info.method,
            "experiment": info.experiment,
            "split": info.split,
            "result_path": str(info.path),
            "total": info.total,
            "parsed": info.parsed,
            "parse_rate": info.parsed / info.total if info.total else 0.0,
            "official_success_rate": rate(records, "official_success", info.total),
            "top1_success_rate": rate(records, "top1_success", info.total),
            "best_iou_center_xy_error_mean": mean(records, "best_iou_center_xy_error_px"),
            "best_iou_width_height_error_mean": mean(records, "best_iou_width_height_error_px"),
            "best_iou_angle_error_mean": mean(records, "best_iou_circular_angle_error_deg"),
            "max_iou_with_all_labels_mean": mean(records, "max_iou_with_all_labels"),
            "top1_center_xy_error_mean": mean(records, "top1_center_xy_error_px"),
            "top1_angle_error_mean": mean(records, "top1_circular_angle_error_deg"),
            "top1_iou_mean": mean(records, "top1_iou"),
        }
        for level in GEOMETRY_LEVELS:
            row[f"top1_{level}_rate"] = rate(records, f"top1_{level}", info.total)
            row[f"all_labels_{level}_rate"] = rate(records, f"all_labels_{level}", info.total)
        rows.append(row)

    fieldnames = [
        "method",
        "experiment",
        "split",
        "total",
        "parsed",
        "parse_rate",
        "official_success_rate",
        "top1_success_rate",
        "top1_strict_10_rate",
        "all_labels_strict_10_rate",
        "top1_medium_20_rate",
        "all_labels_medium_20_rate",
        "top1_loose_30_rate",
        "all_labels_loose_30_rate",
        "best_iou_center_xy_error_mean",
        "best_iou_angle_error_mean",
        "max_iou_with_all_labels_mean",
        "top1_center_xy_error_mean",
        "top1_angle_error_mean",
        "top1_iou_mean",
        "best_iou_width_height_error_mean",
        "result_id",
        "result_path",
    ]
    write_csv(out_dir / "main_summary.csv", rows, fieldnames)


def write_error_stats(out_dir: Path, grouped: dict[str, tuple[ResultInfo, list[dict[str, Any]]]]):
    metrics = [
        "top1_center_xy_error_px",
        "top1_width_height_error_px",
        "top1_circular_angle_error_deg",
        "top1_iou",
        "best_iou_center_xy_error_px",
        "best_iou_width_height_error_px",
        "best_iou_circular_angle_error_deg",
        "max_iou_with_all_labels",
        "min_center_xy_error_px",
        "min_width_height_error_px",
        "min_circular_angle_error_deg",
    ]
    rows = []
    for result_id, (info, records) in grouped.items():
        for metric in metrics:
            values = [float(row[metric]) for row in records]
            metric_stats = stats(values)
            rows.append({
                "method": info.method,
                "experiment": info.experiment,
                "split": info.split,
                "metric": metric,
                **metric_stats,
                "result_id": result_id,
            })
    write_csv(out_dir / "error_stats.csv", rows)


def write_sample_metrics(out_dir: Path, grouped: dict[str, tuple[ResultInfo, list[dict[str, Any]]]]):
    rows = []
    for result_id, (info, records) in grouped.items():
        for record in records:
            row = {
                "method": info.method,
                "experiment": info.experiment,
                "split": info.split,
                "result_id": result_id,
                "grasp_id": record["grasp_id"],
                "obj_name": record["obj_name"],
                "official_success": int(record["official_success"]),
                "top1_success": int(record["top1_success"]),
                "target_label_count": record["target_label_count"],
                "top1_center_xy_error_px": record["top1_center_xy_error_px"],
                "top1_width_height_error_px": record["top1_width_height_error_px"],
                "top1_circular_angle_error_deg": record["top1_circular_angle_error_deg"],
                "top1_iou": record["top1_iou"],
                "best_iou_center_xy_error_px": record["best_iou_center_xy_error_px"],
                "best_iou_width_height_error_px": record["best_iou_width_height_error_px"],
                "best_iou_circular_angle_error_deg": record["best_iou_circular_angle_error_deg"],
                "max_iou_with_all_labels": record["max_iou_with_all_labels"],
            }
            for level in GEOMETRY_LEVELS:
                row[f"top1_{level}"] = int(record[f"top1_{level}"])
                row[f"all_labels_{level}"] = int(record[f"all_labels_{level}"])
            rows.append(row)
    write_csv(out_dir / "sample_metrics.csv", rows)


def write_geometry_sweep(out_dir: Path, grouped: dict[str, tuple[ResultInfo, list[dict[str, Any]]]]):
    rows = []
    for result_id, (info, records) in grouped.items():
        for center_t in CENTER_THRESHOLDS:
            for angle_t in ANGLE_THRESHOLDS:
                for scope in ["top1", "all_labels"]:
                    success_count = 0
                    for record in records:
                        if scope == "top1":
                            metric = record["_label_metrics"][0]
                            success = geometry_success(metric, center_t, center_t, angle_t)
                        else:
                            success = any(
                                geometry_success(metric, center_t, center_t, angle_t)
                                for metric in record["_label_metrics"]
                            )
                        success_count += int(success)
                    rows.append({
                        "method": info.method,
                        "experiment": info.experiment,
                        "split": info.split,
                        "scope": scope,
                        "center_threshold_px": center_t,
                        "width_height_threshold_px": center_t,
                        "angle_threshold_deg": angle_t,
                        "success_count": success_count,
                        "success_rate": success_count / info.total if info.total else 0.0,
                        "result_id": result_id,
                    })
    write_csv(out_dir / "threshold_sweep_geometry.csv", rows)


def write_iou_sweep(out_dir: Path, grouped: dict[str, tuple[ResultInfo, list[dict[str, Any]]]]):
    rows = []
    for result_id, (info, records) in grouped.items():
        for iou_t in IOU_THRESHOLDS:
            for angle_t in ANGLE_THRESHOLDS:
                for scope in ["top1", "all_labels"]:
                    iou_only_count = 0
                    iou_angle_count = 0
                    for record in records:
                        metrics = record["_label_metrics"]
                        if scope == "top1":
                            selected = [metrics[0]]
                        else:
                            selected = metrics
                        iou_only = any(metric["iou"] >= iou_t for metric in selected)
                        iou_angle = any(official_success(metric, iou_t, angle_t) for metric in selected)
                        iou_only_count += int(iou_only)
                        iou_angle_count += int(iou_angle)
                    rows.append({
                        "method": info.method,
                        "experiment": info.experiment,
                        "split": info.split,
                        "scope": scope,
                        "iou_threshold": iou_t,
                        "angle_threshold_deg": angle_t,
                        "iou_only_success_count": iou_only_count,
                        "iou_only_success_rate": iou_only_count / info.total if info.total else 0.0,
                        "iou_angle_success_count": iou_angle_count,
                        "iou_angle_success_rate": iou_angle_count / info.total if info.total else 0.0,
                        "result_id": result_id,
                    })
    write_csv(out_dir / "threshold_sweep_iou.csv", rows)


def write_direct_crop_cross(out_dir: Path, grouped: dict[str, tuple[ResultInfo, list[dict[str, Any]]]]):
    direct_items = [(rid, info, rows) for rid, (info, rows) in grouped.items() if info.method == "direct"]
    crop_items = [(rid, info, rows) for rid, (info, rows) in grouped.items() if info.method == "oracle_crop"]
    conditions = ["official_success", "top1_success"]
    for level in GEOMETRY_LEVELS:
        conditions.append(f"top1_{level}")
        conditions.append(f"all_labels_{level}")

    rows = []
    for direct_id, direct_info, direct_records in direct_items:
        direct_by_id = {record["grasp_id"]: record for record in direct_records}
        for crop_id, crop_info, crop_records in crop_items:
            if direct_info.split != crop_info.split:
                continue
            crop_by_id = {record["grasp_id"]: record for record in crop_records}
            shared_ids = sorted(set(direct_by_id) & set(crop_by_id))
            for condition in conditions:
                both_success = both_fail = direct_success_crop_fail = direct_fail_crop_success = 0
                for grasp_id in shared_ids:
                    direct_success = bool(direct_by_id[grasp_id][condition])
                    crop_success = bool(crop_by_id[grasp_id][condition])
                    both_success += int(direct_success and crop_success)
                    both_fail += int(not direct_success and not crop_success)
                    direct_success_crop_fail += int(direct_success and not crop_success)
                    direct_fail_crop_success += int(not direct_success and crop_success)
                total = len(shared_ids)
                rows.append({
                    "split": direct_info.split,
                    "condition": condition,
                    "direct_experiment": direct_info.experiment,
                    "crop_experiment": crop_info.experiment,
                    "matched_total": total,
                    "both_success": both_success,
                    "both_fail": both_fail,
                    "direct_success_crop_fail": direct_success_crop_fail,
                    "direct_fail_crop_success": direct_fail_crop_success,
                    "rescued_by_crop": direct_fail_crop_success,
                    "broken_by_crop": direct_success_crop_fail,
                    "net_gain": direct_fail_crop_success - direct_success_crop_fail,
                    "rescued_rate": direct_fail_crop_success / total if total else 0.0,
                    "broken_rate": direct_success_crop_fail / total if total else 0.0,
                    "net_gain_rate": (direct_fail_crop_success - direct_success_crop_fail) / total if total else 0.0,
                    "direct_result_id": direct_id,
                    "crop_result_id": crop_id,
                })
    write_csv(out_dir / "direct_vs_crop_cross.csv", rows)


def crop_quality_for_record(record: dict[str, Any], mask_txn, image_size: int) -> dict[str, float] | None:
    crop_box = record.get("crop_box")
    if not crop_box:
        return None
    mask = load_mask(mask_txn, record["grasp_id"])
    if mask is None:
        return None

    height, width = mask.shape[:2]
    x_scale = width / float(image_size)
    y_scale = height / float(image_size)
    x0, y0, x1, y1 = crop_box
    x0 = max(0, min(width, int(np.floor(x0 * x_scale))))
    y0 = max(0, min(height, int(np.floor(y0 * y_scale))))
    x1 = max(0, min(width, int(np.ceil(x1 * x_scale))))
    y1 = max(0, min(height, int(np.ceil(y1 * y_scale))))
    if x1 <= x0 or y1 <= y0:
        return None

    mask_bool = mask.astype(bool)
    object_pixels = int(mask_bool.sum())
    crop_mask = mask_bool[y0:y1, x0:x1]
    object_inside = int(crop_mask.sum())
    crop_area = int((x1 - x0) * (y1 - y0))
    return {
        "object_coverage": object_inside / object_pixels if object_pixels else 0.0,
        "background_ratio": (crop_area - object_inside) / crop_area if crop_area else 0.0,
        "crop_area_ratio": crop_area / float(width * height) if width > 0 and height > 0 else 0.0,
    }


def write_crop_quality(
    out_dir: Path,
    grouped: dict[str, tuple[ResultInfo, list[dict[str, Any]]]],
    mask_env,
    image_size: int,
):
    sample_rows = []
    with mask_env.begin() as mask_txn:
        for result_id, (info, records) in grouped.items():
            if info.method != "oracle_crop":
                continue
            for record in records:
                quality = crop_quality_for_record(record, mask_txn, image_size)
                if quality is None:
                    continue
                sample_rows.append({
                    "method": info.method,
                    "experiment": info.experiment,
                    "split": info.split,
                    "result_id": result_id,
                    "grasp_id": record["grasp_id"],
                    "obj_name": record["obj_name"],
                    "official_success": int(record["official_success"]),
                    "top1_success": int(record["top1_success"]),
                    **quality,
                })
    write_csv(out_dir / "crop_quality_samples.csv", sample_rows)

    summary_rows = []
    for result_id in sorted({row["result_id"] for row in sample_rows}):
        subset = [row for row in sample_rows if row["result_id"] == result_id]
        for group_name, group_rows in [
            ("all", subset),
            ("official_success", [row for row in subset if row["official_success"]]),
            ("official_fail", [row for row in subset if not row["official_success"]]),
        ]:
            if not group_rows:
                continue
            info_row = group_rows[0]
            for metric in ["object_coverage", "background_ratio", "crop_area_ratio"]:
                metric_stats = stats([float(row[metric]) for row in group_rows])
                summary_rows.append({
                    "method": info_row["method"],
                    "experiment": info_row["experiment"],
                    "split": info_row["split"],
                    "group": group_name,
                    "metric": metric,
                    "count": len(group_rows),
                    **metric_stats,
                    "result_id": result_id,
                })
    write_csv(out_dir / "crop_quality_summary.csv", summary_rows)


def run_analysis(
    results: list[str | Path],
    out_dir: str | Path,
    grasp_lmdb: str | Path = DEFAULT_GRASP_LMDB,
    mask_lmdb: str | Path = DEFAULT_MASK_LMDB,
    iou_threshold: float = 0.25,
    angle_threshold: float = 30.0,
    image_size: int = IMAGE_SIZE,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grasp_env = lmdb.open(
        str(Path(grasp_lmdb)),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2048,
    )
    mask_env = lmdb.open(
        str(Path(mask_lmdb)),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2048,
    )

    grouped: dict[str, tuple[ResultInfo, list[dict[str, Any]]]] = {}
    with grasp_env.begin() as grasp_txn:
        analysis_args = argparse.Namespace(
            iou_threshold=iou_threshold,
            angle_threshold=angle_threshold,
            image_size=image_size,
        )
        for result in results:
            info, records = read_result_file(Path(result), grasp_txn, analysis_args)
            grouped[info.result_id] = (info, records)
            print(f"{info.method} {info.experiment} {info.split}: parsed {info.parsed}/{info.total}")

    write_main_summary(out_dir, grouped)
    write_error_stats(out_dir, grouped)
    write_sample_metrics(out_dir, grouped)
    write_geometry_sweep(out_dir, grouped)
    write_iou_sweep(out_dir, grouped)
    write_direct_crop_cross(out_dir, grouped)
    write_crop_quality(out_dir, grouped, mask_env, image_size)

    manifest = {
        "iou_threshold": iou_threshold,
        "angle_threshold": angle_threshold,
        "image_size": image_size,
        "result_count": len(grouped),
        "outputs": [
            "main_summary.csv",
            "error_stats.csv",
            "sample_metrics.csv",
            "threshold_sweep_geometry.csv",
            "threshold_sweep_iou.csv",
            "direct_vs_crop_cross.csv",
            "crop_quality_samples.csv",
            "crop_quality_summary.csv",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"analysis_dir={out_dir}")


def main():
    args = parse_args()
    run_analysis(
        results=args.results,
        out_dir=args.out_dir,
        grasp_lmdb=args.grasp_lmdb,
        mask_lmdb=args.mask_lmdb,
        iou_threshold=args.iou_threshold,
        angle_threshold=args.angle_threshold,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
