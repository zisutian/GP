from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import lmdb
import numpy as np
import torch

from collect_checkpoint_manifest import infer as infer_checkpoint_manifest


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
METHOD_DIRS = {
    "direct": "direct_grasp",
    "oracle_crop": "oracle_crop",
    "pred_vcot": "predicted_vcot",
}
COMPARISON_DIR = "comparisons"


@dataclass
class ResultInfo:
    result_id: str
    method: str
    experiment: str
    split: str
    path: Path
    total: int
    parsed: int
    evaluation_mode: str = ""
    target_coordinate_frame: str = ""
    bbox_edge_expand: str = ""
    min_bbox_half_size: str = ""
    target_grasp_index: str = ""
    loaded_vcot_config: str = ""


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


def csv_value(value: Any) -> str:
    return "" if value is None else str(value)


def infer_info(path: Path, total: int, parsed: int) -> ResultInfo:
    manifest_info = infer_checkpoint_manifest(path)
    method = manifest_info["method"]
    experiment = manifest_info["experiment"]
    split = manifest_info["split"]
    evaluation_mode = manifest_info["evaluation_mode"]
    return ResultInfo(
        result_id=f"{method}:{experiment}:{split}:{path.name}",
        method=method,
        experiment=experiment,
        split=split,
        path=path,
        total=total,
        parsed=parsed,
        evaluation_mode=evaluation_mode,
        target_coordinate_frame=csv_value(manifest_info.get("target_coordinate_frame", "")),
        bbox_edge_expand=csv_value(manifest_info.get("bbox_edge_expand", "")),
        min_bbox_half_size=csv_value(manifest_info.get("min_bbox_half_size", "")),
        target_grasp_index=csv_value(manifest_info.get("target_grasp_index", "")),
        loaded_vcot_config=csv_value(manifest_info.get("loaded_vcot_config", "")),
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


def value_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def value_percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.array(values, dtype=np.float64), percentile)) if values else 0.0


def valid_xyxy(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    x0, y0, x1, y1 = [float(value) for value in box]
    return x1 > x0 and y1 > y0


def xyxy_iou(box_a: Any, box_b: Any) -> float | None:
    if not valid_xyxy(box_a) or not valid_xyxy(box_b):
        return None
    ax0, ay0, ax1, ay1 = [float(value) for value in box_a]
    bx0, by0, bx1, by1 = [float(value) for value in box_b]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def xyxy_center_error(box_a: Any, box_b: Any) -> float | None:
    if not valid_xyxy(box_a) or not valid_xyxy(box_b):
        return None
    ax0, ay0, ax1, ay1 = [float(value) for value in box_a]
    bx0, by0, bx1, by1 = [float(value) for value in box_b]
    center_a = np.array([(ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0], dtype=np.float32)
    center_b = np.array([(bx0 + bx1) / 2.0, (by0 + by1) / 2.0], dtype=np.float32)
    return float(np.linalg.norm(center_a - center_b))


def target_crop_frame_flags(target_full_grasp: Any, crop_box: Any) -> dict[str, bool]:
    if not valid_xyxy(crop_box) or not isinstance(target_full_grasp, (list, tuple)) or len(target_full_grasp) < 4:
        return {
            "grasp_center_outside_crop": True,
            "grasp_size_gt_crop": True,
            "grasp_not_expressible_crop_frame": True,
        }
    x0, y0, x1, y1 = [float(value) for value in crop_box]
    x, y, w, h = [float(value) for value in target_full_grasp[:4]]
    crop_w = x1 - x0
    crop_h = y1 - y0
    center_outside = x < x0 or x > x1 or y < y0 or y > y1
    size_gt_crop = w > crop_w or h > crop_h
    return {
        "grasp_center_outside_crop": center_outside,
        "grasp_size_gt_crop": size_gt_crop,
        "grasp_not_expressible_crop_frame": center_outside or size_gt_crop,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_section_csv(
    out_dir: Path,
    section_name: str,
    filename: str,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
):
    write_csv(out_dir / section_name / filename, rows, fieldnames)


def reset_analysis_output_dir(out_dir: Path):
    resolved = out_dir.resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), REPO_ROOT.resolve(), REPO_ROOT.parent.resolve()}
    if resolved in protected:
        raise ValueError(f"Refusing to reset protected analysis output directory: {out_dir}")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)


def grouped_for_method(
    grouped: dict[str, tuple[ResultInfo, list[dict[str, Any]]]],
    method: str,
) -> dict[str, tuple[ResultInfo, list[dict[str, Any]]]]:
    return {result_id: item for result_id, item in grouped.items() if item[0].method == method}


def write_main_summary(out_dir: Path, grouped: dict[str, tuple[ResultInfo, list[dict[str, Any]]]]):
    rows = []
    for result_id, (info, records) in grouped.items():
        row = {
            "result_id": result_id,
            "method": info.method,
            "experiment": info.experiment,
            "split": info.split,
            "evaluation_mode": info.evaluation_mode,
            "target_coordinate_frame": info.target_coordinate_frame,
            "bbox_edge_expand": info.bbox_edge_expand,
            "min_bbox_half_size": info.min_bbox_half_size,
            "target_grasp_index": info.target_grasp_index,
            "loaded_vcot_config": info.loaded_vcot_config,
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
        "evaluation_mode",
        "target_coordinate_frame",
        "bbox_edge_expand",
        "min_bbox_half_size",
        "target_grasp_index",
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
        "loaded_vcot_config",
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
    if not direct_items or not crop_items:
        return
    conditions = cross_conditions()

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
    fieldnames = [
        "split",
        "condition",
        "direct_experiment",
        "crop_experiment",
        "matched_total",
        "both_success",
        "both_fail",
        "direct_success_crop_fail",
        "direct_fail_crop_success",
        "rescued_by_crop",
        "broken_by_crop",
        "net_gain",
        "rescued_rate",
        "broken_rate",
        "net_gain_rate",
        "direct_result_id",
        "crop_result_id",
    ]
    write_section_csv(out_dir, COMPARISON_DIR, "direct_vs_crop_cross.csv", rows, fieldnames)


def cross_conditions() -> list[str]:
    conditions = ["official_success", "top1_success"]
    for level in GEOMETRY_LEVELS:
        conditions.append(f"top1_{level}")
        conditions.append(f"all_labels_{level}")
    return conditions


def write_direct_pred_vcot_cross(out_dir: Path, grouped: dict[str, tuple[ResultInfo, list[dict[str, Any]]]]):
    direct_items = [(rid, info, rows) for rid, (info, rows) in grouped.items() if info.method == "direct"]
    pred_items = [(rid, info, rows) for rid, (info, rows) in grouped.items() if info.method == "pred_vcot"]
    if not direct_items or not pred_items:
        return
    conditions = cross_conditions()

    summary_rows = []
    for direct_id, direct_info, direct_records in direct_items:
        direct_by_id = {record["grasp_id"]: record for record in direct_records}
        for pred_id, pred_info, pred_records in pred_items:
            if direct_info.split != pred_info.split:
                continue
            pred_by_id = {record["grasp_id"]: record for record in pred_records}
            shared_ids = sorted(set(direct_by_id) & set(pred_by_id))
            for condition in conditions:
                both_success = both_fail = direct_success_pred_fail = direct_fail_pred_success = 0
                for grasp_id in shared_ids:
                    direct_success = bool(direct_by_id[grasp_id][condition])
                    pred_success = bool(pred_by_id[grasp_id][condition])
                    both_success += int(direct_success and pred_success)
                    both_fail += int(not direct_success and not pred_success)
                    direct_success_pred_fail += int(direct_success and not pred_success)
                    direct_fail_pred_success += int(not direct_success and pred_success)

                total = len(shared_ids)
                summary_rows.append({
                    "split": direct_info.split,
                    "condition": condition,
                    "direct_experiment": direct_info.experiment,
                    "pred_vcot_experiment": pred_info.experiment,
                    "matched_total": total,
                    "both_success": both_success,
                    "both_fail": both_fail,
                    "direct_success_pred_fail": direct_success_pred_fail,
                    "direct_fail_pred_success": direct_fail_pred_success,
                    "rescued_by_pred_vcot": direct_fail_pred_success,
                    "broken_by_pred_vcot": direct_success_pred_fail,
                    "net_gain": direct_fail_pred_success - direct_success_pred_fail,
                    "rescued_rate": direct_fail_pred_success / total if total else 0.0,
                    "broken_rate": direct_success_pred_fail / total if total else 0.0,
                    "net_gain_rate": (direct_fail_pred_success - direct_success_pred_fail) / total if total else 0.0,
                    "direct_result_id": direct_id,
                    "pred_vcot_result_id": pred_id,
                })

    summary_fieldnames = [
        "split",
        "condition",
        "direct_experiment",
        "pred_vcot_experiment",
        "matched_total",
        "both_success",
        "both_fail",
        "direct_success_pred_fail",
        "direct_fail_pred_success",
        "rescued_by_pred_vcot",
        "broken_by_pred_vcot",
        "net_gain",
        "rescued_rate",
        "broken_rate",
        "net_gain_rate",
        "direct_result_id",
        "pred_vcot_result_id",
    ]
    write_section_csv(
        out_dir,
        COMPARISON_DIR,
        "direct_vs_pred_vcot_cross.csv",
        summary_rows,
        summary_fieldnames,
    )


def crop_quality_for_record(record: dict[str, Any], mask_txn, image_size: int) -> dict[str, float] | None:
    return crop_quality_for_box(record["grasp_id"], record.get("crop_box"), mask_txn, image_size)


def crop_quality_for_box(grasp_id: str, crop_box: Any, mask_txn, image_size: int) -> dict[str, float] | None:
    if not valid_xyxy(crop_box):
        return None
    mask = load_mask(mask_txn, grasp_id)
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
    if not any(info.method == "oracle_crop" for info, _records in grouped.values()):
        return

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
    summary_fieldnames = [
        "method",
        "experiment",
        "split",
        "group",
        "metric",
        "count",
        "mean",
        "median",
        "p75",
        "p90",
        "result_id",
    ]
    write_section_csv(out_dir, METHOD_DIRS["oracle_crop"], "crop_quality_summary.csv", summary_rows, summary_fieldnames)


def write_predicted_vcot_diagnostics(
    out_dir: Path,
    grouped: dict[str, tuple[ResultInfo, list[dict[str, Any]]]],
    mask_env,
    image_size: int,
):
    if not any(info.method == "pred_vcot" for info, _records in grouped.values()):
        return

    summary_rows = []

    with mask_env.begin() as mask_txn:
        for result_id, (info, records) in grouped.items():
            if info.method != "pred_vcot":
                continue
            data = json.loads(info.path.read_text(encoding="utf-8"))
            outputs = data.get("outputs", [])
            records_by_id = {record["grasp_id"]: record for record in records}

            bbox_ious = []
            bbox_center_errors = []
            crop_ious = []
            crop_center_errors = []
            object_coverages = []
            background_ratios = []
            crop_area_ratios = []
            success_bbox_ious = []
            fail_bbox_ious = []
            success_crop_ious = []
            fail_crop_ious = []
            success_object_coverages = []
            fail_object_coverages = []

            bbox_parsed = 0
            pred_crop_valid = 0
            good_crop_count = 0
            bad_crop_count = 0
            good_crop_success = 0
            bad_crop_success = 0
            bbox_iou_ge_050_count = 0
            bbox_iou_lt_050_count = 0
            bbox_iou_ge_050_success = 0
            bbox_iou_lt_050_success = 0
            grasp_center_outside = 0
            grasp_size_gt_crop = 0
            grasp_not_expressible = 0

            for output in outputs:
                grasp_id = output.get("grasp_id", "")
                record = records_by_id.get(grasp_id)
                official = bool(record and record.get("official_success"))
                pred_bbox = output.get("pred_bbox_xyxy")
                gt_bbox = output.get("gt_bbox_xyxy")
                pred_crop = output.get("pred_crop_box")
                gt_crop = output.get("gt_crop_box")

                bbox_iou = output.get("pred_bbox_iou")
                if bbox_iou is None:
                    bbox_iou = xyxy_iou(pred_bbox, gt_bbox)
                else:
                    bbox_iou = float(bbox_iou)
                bbox_center_error = xyxy_center_error(pred_bbox, gt_bbox)
                crop_iou = xyxy_iou(pred_crop, gt_crop)
                crop_center_error = xyxy_center_error(pred_crop, gt_crop)
                quality = crop_quality_for_box(grasp_id, pred_crop, mask_txn, image_size)
                express_flags = target_crop_frame_flags(output.get("target_full_grasp"), pred_crop)

                bbox_ok = bbox_iou is not None
                crop_ok = valid_xyxy(pred_crop)
                bbox_parsed += int(bbox_ok)
                pred_crop_valid += int(crop_ok)
                grasp_center_outside += int(express_flags["grasp_center_outside_crop"])
                grasp_size_gt_crop += int(express_flags["grasp_size_gt_crop"])
                grasp_not_expressible += int(express_flags["grasp_not_expressible_crop_frame"])

                object_coverage = quality["object_coverage"] if quality else None
                background_ratio = quality["background_ratio"] if quality else None
                crop_area_ratio = quality["crop_area_ratio"] if quality else None
                good_crop = bool(crop_ok and crop_iou is not None and crop_iou >= 0.50 and object_coverage is not None and object_coverage >= 0.95)
                bad_crop = not good_crop

                if bbox_iou is not None:
                    bbox_ious.append(float(bbox_iou))
                    (success_bbox_ious if official else fail_bbox_ious).append(float(bbox_iou))
                    if bbox_iou >= 0.50:
                        bbox_iou_ge_050_count += 1
                        bbox_iou_ge_050_success += int(official)
                    else:
                        bbox_iou_lt_050_count += 1
                        bbox_iou_lt_050_success += int(official)
                if bbox_center_error is not None:
                    bbox_center_errors.append(float(bbox_center_error))
                if crop_iou is not None:
                    crop_ious.append(float(crop_iou))
                    (success_crop_ious if official else fail_crop_ious).append(float(crop_iou))
                if crop_center_error is not None:
                    crop_center_errors.append(float(crop_center_error))
                if object_coverage is not None:
                    object_coverages.append(float(object_coverage))
                    (success_object_coverages if official else fail_object_coverages).append(float(object_coverage))
                if background_ratio is not None:
                    background_ratios.append(float(background_ratio))
                if crop_area_ratio is not None:
                    crop_area_ratios.append(float(crop_area_ratio))

                if good_crop:
                    good_crop_count += 1
                    good_crop_success += int(official)
                if bad_crop:
                    bad_crop_count += 1
                    bad_crop_success += int(official)

            total = len(outputs)
            parsed = len(records)
            summary_rows.append({
                "experiment": info.experiment,
                "split": info.split,
                "result_json": str(info.path),
                "target_coordinate_frame": info.target_coordinate_frame,
                "bbox_edge_expand": info.bbox_edge_expand,
                "min_bbox_half_size": info.min_bbox_half_size,
                "target_grasp_index": info.target_grasp_index,
                "total": total,
                "bbox_parse_rate": bbox_parsed / total if total else 0.0,
                "parse_rate": parsed / total if total else 0.0,
                "success_rate": rate(records, "official_success", total),
                "top1_success_rate": rate(records, "top1_success", total),
                "bbox_iou_mean": value_mean(bbox_ious),
                "bbox_iou_p10": value_percentile(bbox_ious, 10),
                "bbox_iou_lt_025_rate": sum(value < 0.25 for value in bbox_ious) / len(bbox_ious) if bbox_ious else 0.0,
                "bbox_iou_lt_050_rate": sum(value < 0.50 for value in bbox_ious) / len(bbox_ious) if bbox_ious else 0.0,
                "bbox_center_error_mean": value_mean(bbox_center_errors),
                "pred_crop_valid_rate": pred_crop_valid / total if total else 0.0,
                "crop_iou_mean": value_mean(crop_ious),
                "crop_iou_p10": value_percentile(crop_ious, 10),
                "crop_iou_lt_050_rate": sum(value < 0.50 for value in crop_ious) / len(crop_ious) if crop_ious else 0.0,
                "crop_center_error_mean": value_mean(crop_center_errors),
                "object_coverage_mean": value_mean(object_coverages),
                "object_coverage_p10": value_percentile(object_coverages, 10),
                "object_coverage_lt_095_rate": sum(value < 0.95 for value in object_coverages) / len(object_coverages) if object_coverages else 0.0,
                "object_coverage_lt_080_rate": sum(value < 0.80 for value in object_coverages) / len(object_coverages) if object_coverages else 0.0,
                "background_ratio_mean": value_mean(background_ratios),
                "background_ratio_p90": value_percentile(background_ratios, 90),
                "crop_area_ratio_mean": value_mean(crop_area_ratios),
                "crop_area_ratio_p90": value_percentile(crop_area_ratios, 90),
                "grasp_center_outside_crop_rate": grasp_center_outside / total if total else 0.0,
                "grasp_size_gt_crop_rate": grasp_size_gt_crop / total if total else 0.0,
                "grasp_not_expressible_crop_frame_rate": grasp_not_expressible / total if total else 0.0,
                "success_rate_good_crop": good_crop_success / good_crop_count if good_crop_count else 0.0,
                "success_rate_bad_crop": bad_crop_success / bad_crop_count if bad_crop_count else 0.0,
                "good_crop_count": good_crop_count,
                "bad_crop_count": bad_crop_count,
                "success_rate_bbox_iou_ge_050": bbox_iou_ge_050_success / bbox_iou_ge_050_count if bbox_iou_ge_050_count else 0.0,
                "success_rate_bbox_iou_lt_050": bbox_iou_lt_050_success / bbox_iou_lt_050_count if bbox_iou_lt_050_count else 0.0,
                "bbox_iou_ge_050_count": bbox_iou_ge_050_count,
                "bbox_iou_lt_050_count": bbox_iou_lt_050_count,
                "success_bbox_iou_mean": value_mean(success_bbox_ious),
                "fail_bbox_iou_mean": value_mean(fail_bbox_ious),
                "success_crop_iou_mean": value_mean(success_crop_ious),
                "fail_crop_iou_mean": value_mean(fail_crop_ious),
                "success_object_coverage_mean": value_mean(success_object_coverages),
                "fail_object_coverage_mean": value_mean(fail_object_coverages),
            })

    summary_fieldnames = [
        "experiment",
        "split",
        "result_json",
        "target_coordinate_frame",
        "bbox_edge_expand",
        "min_bbox_half_size",
        "target_grasp_index",
        "total",
        "bbox_parse_rate",
        "parse_rate",
        "success_rate",
        "top1_success_rate",
        "bbox_iou_mean",
        "bbox_iou_p10",
        "bbox_iou_lt_025_rate",
        "bbox_iou_lt_050_rate",
        "bbox_center_error_mean",
        "pred_crop_valid_rate",
        "crop_iou_mean",
        "crop_iou_p10",
        "crop_iou_lt_050_rate",
        "crop_center_error_mean",
        "object_coverage_mean",
        "object_coverage_p10",
        "object_coverage_lt_095_rate",
        "object_coverage_lt_080_rate",
        "background_ratio_mean",
        "background_ratio_p90",
        "crop_area_ratio_mean",
        "crop_area_ratio_p90",
        "grasp_center_outside_crop_rate",
        "grasp_size_gt_crop_rate",
        "grasp_not_expressible_crop_frame_rate",
        "success_rate_good_crop",
        "success_rate_bad_crop",
        "good_crop_count",
        "bad_crop_count",
        "success_rate_bbox_iou_ge_050",
        "success_rate_bbox_iou_lt_050",
        "bbox_iou_ge_050_count",
        "bbox_iou_lt_050_count",
        "success_bbox_iou_mean",
        "fail_bbox_iou_mean",
        "success_crop_iou_mean",
        "fail_crop_iou_mean",
        "success_object_coverage_mean",
        "fail_object_coverage_mean",
    ]
    write_section_csv(
        out_dir,
        METHOD_DIRS["pred_vcot"],
        "diagnostics_summary.csv",
        summary_rows,
        summary_fieldnames,
    )


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
    reset_analysis_output_dir(out_dir)

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

    write_main_summary(out_dir, grouped)

    for method, dirname in METHOD_DIRS.items():
        method_grouped = grouped_for_method(grouped, method)
        if not method_grouped:
            continue
        method_out_dir = out_dir / dirname
        write_main_summary(method_out_dir, method_grouped)
        write_error_stats(method_out_dir, method_grouped)
        write_geometry_sweep(method_out_dir, method_grouped)
        write_iou_sweep(method_out_dir, method_grouped)

    write_direct_crop_cross(out_dir, grouped)
    write_direct_pred_vcot_cross(out_dir, grouped)
    write_crop_quality(out_dir, grouped, mask_env, image_size)
    write_predicted_vcot_diagnostics(out_dir, grouped, mask_env, image_size)

    outputs = sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*.csv"))

    manifest = {
        "iou_threshold": iou_threshold,
        "angle_threshold": angle_threshold,
        "image_size": image_size,
        "result_count": len(grouped),
        "outputs": outputs,
        "method_dirs": {
            method: dirname
            for method, dirname in METHOD_DIRS.items()
            if grouped_for_method(grouped, method)
        },
        "comparison_dir": COMPARISON_DIR,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"analysis_dir={out_dir} result_count={len(grouped)} csv_outputs={len(outputs)}")


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
