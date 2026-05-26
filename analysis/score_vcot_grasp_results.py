from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import cv2
import lmdb
import torch

from collect_checkpoint_manifest import infer as infer_checkpoint_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ANALYSIS_ROOT))

from grasp_settings import build_settings  # noqa: E402


SETTINGS = build_settings()
DEFAULT_GRASP_LMDB = Path(SETTINGS["GRASP_GRASP_LMDB"])
DEFAULT_MASK_LMDB = Path(SETTINGS["GRASP_MASK_LMDB"])
IMAGE_SIZE = 416


def parse_args():
    parser = argparse.ArgumentParser(description="Score direct-grasp result JSON files with the VCoTGrasp metric.")
    parser.add_argument("results", nargs="+", help="Result JSON file(s) produced by evaluate_direct_grasp.py.")
    parser.add_argument("--grasp-lmdb", default=str(DEFAULT_GRASP_LMDB))
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--angle-threshold", type=float, default=30.0)
    parser.add_argument("--write", action="store_true", help="Write the VCoTGrasp metrics back into each JSON summary.")
    parser.add_argument("--out-dir", default=None, help="Write rescored JSON copies under this directory.")
    parser.add_argument(
        "--relative-root",
        default=str(REPO_ROOT),
        help="Root used to preserve relative paths under --out-dir.",
    )
    parser.add_argument("--summary-csv", default=None, help="Optional path for a CSV metrics summary.")
    parser.add_argument(
        "--analysis-out-dir",
        default=None,
        help="Optional directory for enhanced analysis CSVs.",
    )
    parser.add_argument("--mask-lmdb", default=str(DEFAULT_MASK_LMDB), help="Mask LMDB for crop quality analysis.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    return parser.parse_args()


def denormalize_grasp_like_vcot(values: list[float], image_size: int = IMAGE_SIZE) -> list[float]:
    return [int(value * image_size) for value in values[:4]] + [float(values[4] * 180.0)]


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


def is_success(pred: list[float], labels: list[list[float]], iou_threshold: float, angle_threshold: float):
    best_iou = 0.0
    best_angle_diff = 180.0
    best_joint_iou = 0.0
    best_joint_angle_diff = 180.0
    for label in labels:
        iou = rotated_rect_iou(pred, label)
        angle_diff = angle_diff_180(pred[4], label[4])
        if iou > best_iou:
            best_iou = iou
        if angle_diff < best_angle_diff:
            best_angle_diff = angle_diff
        if iou > best_joint_iou:
            best_joint_iou = iou
            best_joint_angle_diff = angle_diff
        if iou >= iou_threshold and angle_diff <= angle_threshold:
            return True, iou, angle_diff, best_iou, best_angle_diff
    return False, best_joint_iou, best_joint_angle_diff, best_iou, best_angle_diff


def load_labels(txn, grasp_id: str):
    value = txn.get(f"{grasp_id}.pt".encode("utf-8"))
    if value is None:
        raise FileNotFoundError(f"LMDB key not found: {grasp_id}.pt")
    labels = torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
    return [[float(value) for value in row[1:]] for row in labels]


def output_path_for(path: Path, args) -> Path:
    out_dir = Path(args.out_dir)
    try:
        relative = path.resolve().relative_to(Path(args.relative_root).resolve())
    except ValueError:
        relative = Path(path.name)

    parts = list(relative.parts)
    if "result" in parts:
        relative = Path(*parts[parts.index("result") + 1 :])
    return out_dir / relative


def score_file(path: Path, env, args):
    data = json.loads(path.read_text(encoding="utf-8"))
    outputs = data.get("outputs", [])
    total = len(outputs)
    valid = [output for output in outputs if output.get("pred_norm") is not None]
    success_count = 0
    top1_success_count = 0
    ious = []
    angle_diffs = []
    best_ious = []
    best_angle_diffs = []
    label_counts = []

    with env.begin() as txn:
        for output in valid:
            pred = denormalize_grasp_like_vcot(output["pred_norm"])
            labels = load_labels(txn, output["grasp_id"])
            label_counts.append(len(labels))
            success, joint_iou, joint_angle_diff, best_iou, best_angle_diff = is_success(
                pred,
                labels,
                args.iou_threshold,
                args.angle_threshold,
            )
            if labels:
                top1_iou = rotated_rect_iou(pred, labels[0])
                top1_angle_diff = angle_diff_180(pred[4], labels[0][4])
                top1_success = top1_iou >= args.iou_threshold and top1_angle_diff <= args.angle_threshold
            else:
                top1_iou = 0.0
                top1_angle_diff = 180.0
                top1_success = False
            output["vcot_success"] = success
            output["vcot_top1_success"] = top1_success
            output["vcot_top1_iou"] = top1_iou
            output["vcot_top1_angle_diff"] = top1_angle_diff
            output["vcot_joint_iou"] = joint_iou
            output["vcot_joint_angle_diff"] = joint_angle_diff
            output["vcot_best_iou"] = best_iou
            output["vcot_best_angle_diff"] = best_angle_diff
            output["target_label_count"] = len(labels)
            success_count += int(success)
            top1_success_count += int(top1_success)
            ious.append(joint_iou)
            angle_diffs.append(joint_angle_diff)
            best_ious.append(best_iou)
            best_angle_diffs.append(best_angle_diff)

    metrics = {
        "vcot_iou_threshold": args.iou_threshold,
        "vcot_angle_threshold": args.angle_threshold,
        "vcot_success": success_count,
        "vcot_success_rate_all": success_count / total if total else 0.0,
        "vcot_success_rate_valid": success_count / len(valid) if valid else 0.0,
        "vcot_top1_success": top1_success_count,
        "vcot_top1_success_rate_all": top1_success_count / total if total else 0.0,
        "vcot_top1_success_rate_valid": top1_success_count / len(valid) if valid else 0.0,
        "target_label_count_mean": sum(label_counts) / len(label_counts) if label_counts else 0.0,
        "target_label_count_max": max(label_counts) if label_counts else 0,
        "vcot_joint_iou_mean": sum(ious) / len(ious) if ious else 0.0,
        "vcot_joint_angle_diff_mean": sum(angle_diffs) / len(angle_diffs) if angle_diffs else 0.0,
        "vcot_best_iou_mean": sum(best_ious) / len(best_ious) if best_ious else 0.0,
        "vcot_best_angle_diff_mean": sum(best_angle_diffs) / len(best_angle_diffs) if best_angle_diffs else 0.0,
    }
    data.setdefault("summary", {}).update(metrics)
    written_path = None
    if args.out_dir:
        written_path = output_path_for(path, args)
        written_path.parent.mkdir(parents=True, exist_ok=True)
        written_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    elif args.write:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        written_path = path
    return total, len(valid), metrics, written_path


def write_summary_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "result_path",
        "output_path",
        "method",
        "experiment",
        "split",
        "evaluation_mode",
        "checkpoint",
        "loaded_vcot_config",
        "data_index_root",
        "target_coordinate_frame",
        "bbox_edge_expand",
        "min_bbox_half_size",
        "target_grasp_index",
        "bbox_ratio",
        "grasp_loss_weight",
        "bbox_loss_weight",
        "total",
        "valid",
        "vcot_iou_threshold",
        "vcot_angle_threshold",
        "vcot_success",
        "vcot_success_rate_all",
        "vcot_success_rate_valid",
        "vcot_top1_success",
        "vcot_top1_success_rate_all",
        "vcot_top1_success_rate_valid",
        "target_label_count_mean",
        "target_label_count_max",
        "vcot_joint_iou_mean",
        "vcot_joint_angle_diff_mean",
        "vcot_best_iou_mean",
        "vcot_best_angle_diff_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def print_final_summary(
    rows: list[dict],
    summary_csv: Path | None,
    analysis_out_dir: str | None,
):
    result_count = len(rows)
    total = sum(int(row.get("total", 0)) for row in rows)
    valid = sum(int(row.get("valid", 0)) for row in rows)
    success = sum(int(row.get("vcot_success", 0)) for row in rows)
    top1_success = sum(int(row.get("vcot_top1_success", 0)) for row in rows)

    print(
        "rescore_summary: "
        f"results={result_count} total={total} valid={valid} "
        f"official={success / total * 100 if total else 0.0:.2f}% "
        f"top1={top1_success / total * 100 if total else 0.0:.2f}%"
    )

    for method, split in sorted({(row.get("method", ""), row.get("split", "")) for row in rows}):
        subset = [row for row in rows if row.get("method", "") == method and row.get("split", "") == split]
        subset_total = sum(int(row.get("total", 0)) for row in subset)
        best_official = max((float(row.get("vcot_success_rate_all", 0.0)) for row in subset), default=0.0)
        best_top1 = max((float(row.get("vcot_top1_success_rate_all", 0.0)) for row in subset), default=0.0)
        print(
            "rescore_summary_group: "
            f"method={method} split={split} results={len(subset)} total={subset_total} "
            f"best_official={best_official * 100:.2f}% best_top1={best_top1 * 100:.2f}%"
        )

    if summary_csv is not None:
        print(f"summary_csv={summary_csv}")
    if analysis_out_dir:
        print(f"analysis_out_dir={analysis_out_dir}")


def main():
    args = parse_args()
    if args.write and args.out_dir:
        raise ValueError("Use either --write for in-place updates or --out-dir for separate copies, not both.")
    env = lmdb.open(
        str(Path(args.grasp_lmdb)),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2048,
    )
    summary_rows = []
    analysis_paths = []
    for result in args.results:
        path = Path(result)
        manifest_info = infer_checkpoint_manifest(path)
        total, valid, metrics, written_path = score_file(path, env, args)
        analysis_paths.append(written_path if written_path else path)
        summary_rows.append({
            "result_path": str(path),
            "output_path": str(written_path) if written_path else "",
            "method": manifest_info.get("method", ""),
            "experiment": manifest_info.get("experiment", ""),
            "split": manifest_info.get("split", ""),
            "evaluation_mode": manifest_info.get("evaluation_mode", ""),
            "checkpoint": manifest_info.get("latest_checkpoint", ""),
            "loaded_vcot_config": manifest_info.get("loaded_vcot_config", ""),
            "data_index_root": manifest_info.get("data_index_root", ""),
            "target_coordinate_frame": manifest_info.get("target_coordinate_frame", ""),
            "bbox_edge_expand": manifest_info.get("bbox_edge_expand", ""),
            "min_bbox_half_size": manifest_info.get("min_bbox_half_size", ""),
            "target_grasp_index": manifest_info.get("target_grasp_index", ""),
            "bbox_ratio": manifest_info.get("bbox_ratio", ""),
            "grasp_loss_weight": manifest_info.get("grasp_loss_weight", ""),
            "bbox_loss_weight": manifest_info.get("bbox_loss_weight", ""),
            "total": total,
            "valid": valid,
            **metrics,
        })

    summary_csv = Path(args.summary_csv) if args.summary_csv else None
    if summary_csv is None and args.out_dir:
        summary_csv = Path(args.out_dir) / "summary.csv"
    if summary_csv is not None:
        write_summary_csv(summary_rows, summary_csv)

    if args.analysis_out_dir:
        env.close()
        from analyze_grasp_results import run_analysis

        run_analysis(
            results=analysis_paths,
            out_dir=args.analysis_out_dir,
            grasp_lmdb=args.grasp_lmdb,
            mask_lmdb=args.mask_lmdb,
            iou_threshold=args.iou_threshold,
            angle_threshold=args.angle_threshold,
            image_size=args.image_size,
        )

    print_final_summary(summary_rows, summary_csv, args.analysis_out_dir)


if __name__ == "__main__":
    main()
