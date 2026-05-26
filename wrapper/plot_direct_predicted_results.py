from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import lmdb

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from grasp_settings import build_settings  # noqa: E402


SETTINGS = build_settings()
DEFAULT_ANALYSIS_DIR = Path(SETTINGS["GRASP_ANALYSIS_ALL_ROOT"])
DEFAULT_SUMMARY_CSV = Path(SETTINGS["GRASP_ANALYSIS_ALL_ROOT"]) / "summary.csv"
DEFAULT_PIPELINE_CSV = DEFAULT_ANALYSIS_DIR / "methods/predicted_vcot/diagnostics/pipeline_summary.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "wrapper/direct_predicted_figures"
DEFAULT_IMAGE_LMDB = Path(SETTINGS["GRASP_IMAGE_LMDB"])
IMAGE_SIZE = 416
DIRECT_GRASP_PROMPT_TEMPLATE = "<image>\ngrasp the {obj_name}"
PRED_VCOT_BBOX_PROMPT_TEMPLATE = "<image>\ndetect {obj_name}"
PRED_VCOT_GRASP_PROMPT_TEMPLATE = "<image>\n<image>\ngrasp the {obj_name}"
PAIR_CATEGORIES = ("random", "pred_rescue", "direct_only", "both_success", "both_fail")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create figures for direct vs predicted VCoT grasp results.")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--pipeline-csv", default=str(DEFAULT_PIPELINE_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--image-lmdb", default=str(DEFAULT_IMAGE_LMDB))
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--examples-per-group", type=int, default=6)
    parser.add_argument(
        "--paired-per-category",
        type=int,
        default=4,
        help="Default number of shown paired examples per split/category when creating a new pair config.",
    )
    parser.add_argument(
        "--paired-candidate-limit",
        type=int,
        default=20,
        help="Number of configurable pair candidates written per split/category.",
    )
    parser.add_argument(
        "--paired-categories",
        default="pred_rescue,direct_only,both_success,both_fail",
        help="Comma-separated pair categories to include in the configurable pair candidate file.",
    )
    parser.add_argument(
        "--pair-random-seed",
        type=int,
        default=0,
        help="Seed used when generating random pair candidates.",
    )
    parser.add_argument(
        "--pair-config",
        default=None,
        help="CSV that controls paired examples. Defaults to <out-dir>/examples/paired_example_config.csv.",
    )
    parser.add_argument(
        "--refresh-pair-config",
        action="store_true",
        help="Regenerate the pair config even if it already exists.",
    )
    parser.add_argument(
        "--skip-examples",
        action="store_true",
        help="Only write metric plots; skip qualitative prediction overlays.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def short_name(name: str) -> str:
    replacements = {
        "lora": "L",
        "lr": "lr",
        "patch": "p",
        "edge": "e",
        "half": "h",
        "bbox": "b",
        "vcot_": "V-",
    }
    text = name
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("_", "\n")


def pct_axis(ax) -> None:
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value * 100:.0f}%")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["method"].isin(["direct", "pred_vcot"])].copy()
    numeric_columns = [
        "vcot_success_rate_all",
        "vcot_top1_success_rate_all",
        "vcot_joint_iou_mean",
        "vcot_best_angle_diff_mean",
        "total",
        "valid",
    ]
    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def best_rows(df: pd.DataFrame, metric: str = "vcot_success_rate_all") -> pd.DataFrame:
    rows = []
    for (_method, _split), group in df.groupby(["method", "split"]):
        rows.append(group.sort_values(metric, ascending=False).iloc[0])
    return pd.DataFrame(rows).sort_values(["split", "method"])


def plot_best_method_comparison(df: pd.DataFrame, out_dir: Path) -> Path:
    best = best_rows(df)
    labels = [f"{row.method}\n{row.split}" for row in best.itertuples()]
    x = np.arange(len(best))

    plt.figure(figsize=(8.2, 4.5))
    colors = ["#3B6EA8" if method == "direct" else "#D55E00" for method in best["method"]]
    bars = plt.bar(x, best["vcot_success_rate_all"], color=colors, width=0.65)
    for bar, row in zip(bars, best.itertuples()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{row.vcot_success_rate_all * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(x, labels)
    plt.ylabel("Official success rate")
    plt.title("Best Direct vs Predicted VCoT Results")
    ax = plt.gca()
    pct_axis(ax)
    path = out_dir / "best_direct_vs_predicted_official_success.png"
    savefig(path)
    return path


def plot_all_experiments(df: pd.DataFrame, out_dir: Path) -> Path:
    methods = {"direct": "Direct", "pred_vcot": "Predicted VCoT"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8), sharey=False)
    axes = axes.ravel()
    for ax, ((method, split), group) in zip(axes, df.groupby(["method", "split"])):
        group = group.sort_values("vcot_success_rate_all", ascending=True)
        y = np.arange(len(group))
        ax.barh(y, group["vcot_success_rate_all"], color="#3B6EA8" if method == "direct" else "#D55E00")
        ax.set_yticks(y)
        ax.set_yticklabels([short_name(value) for value in group["experiment"]], fontsize=8)
        ax.set_xlim(0, 0.85)
        ax.xaxis.set_major_formatter(lambda value, _pos: f"{value * 100:.0f}%")
        ax.grid(axis="x", alpha=0.22)
        ax.set_title(f"{methods.get(method, method)} / {split}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for idx, value in enumerate(group["vcot_success_rate_all"]):
            ax.text(value + 0.008, idx, f"{value * 100:.1f}%", va="center", fontsize=8)
    fig.suptitle("Official Success Rate by Experiment", y=1.02, fontsize=14)
    path = out_dir / "all_direct_predicted_experiments.png"
    savefig(path)
    return path


def plot_top1_vs_official(df: pd.DataFrame, out_dir: Path) -> Path:
    best = best_rows(df)
    labels = [f"{row.method}\n{row.split}" for row in best.itertuples()]
    x = np.arange(len(best))
    width = 0.35

    plt.figure(figsize=(8.6, 4.6))
    plt.bar(x - width / 2, best["vcot_success_rate_all"], width, label="Official", color="#3B6EA8")
    plt.bar(x + width / 2, best["vcot_top1_success_rate_all"], width, label="Top-1", color="#E69F00")
    plt.xticks(x, labels)
    plt.ylabel("Success rate")
    plt.title("Official vs Top-1 Success for Best Runs")
    plt.legend(frameon=False)
    pct_axis(plt.gca())
    path = out_dir / "best_runs_official_vs_top1.png"
    savefig(path)
    return path


def plot_pipeline_diagnostics(path: Path, out_dir: Path) -> Path | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    metrics = [
        ("bbox_iou_mean", "BBox IoU"),
        ("crop_iou_mean", "Crop IoU"),
        ("object_coverage_mean", "Object coverage"),
        ("success_rate_good_crop", "Success good crop"),
        ("success_rate_bad_crop", "Success bad crop"),
    ]
    for column, _label in metrics:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, split in zip(axes, ["test_seen", "test_unseen"]):
        group = df[df["split"] == split].sort_values("success_rate", ascending=True)
        y = np.arange(len(group))
        for idx, (column, label) in enumerate(metrics):
            offset = (idx - 2) * 0.12
            ax.scatter(group[column], y + offset, label=label, s=34)
        ax.set_yticks(y)
        ax.set_yticklabels([short_name(value) for value in group["experiment"]], fontsize=8)
        ax.set_xlim(0, 1.02)
        ax.xaxis.set_major_formatter(lambda value, _pos: f"{value * 100:.0f}%")
        ax.grid(axis="x", alpha=0.22)
        ax.set_title(split)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Predicted VCoT Crop/BBox Diagnostics", y=1.02, fontsize=14)
    out_path = out_dir / "predicted_vcot_pipeline_diagnostics.png"
    savefig(out_path)
    return out_path


def read_lmdb_bytes(txn, key: str) -> bytes:
    value = txn.get(key.encode("utf-8"))
    if value is None:
        raise FileNotFoundError(f"LMDB key not found: {key}")
    return value


def load_image(txn, scene_id: str, image_size: int) -> Image.Image:
    data = read_lmdb_bytes(txn, f"{scene_id}.jpg")
    return Image.open(io.BytesIO(data)).convert("RGB").resize((image_size, image_size))


def norm_to_grasp(values: list[float], image_size: int) -> list[float]:
    return [
        float(values[0]) * image_size,
        float(values[1]) * image_size,
        float(values[2]) * image_size,
        float(values[3]) * image_size,
        float(values[4]) * 180.0,
    ]


def draw_rotated_grasp(draw: ImageDraw.ImageDraw, grasp: list[float], color: str, width: int = 3) -> None:
    cx, cy, grasp_w, grasp_h, angle = [float(value) for value in grasp[:5]]
    theta = np.deg2rad(angle)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    half_w = grasp_w / 2.0
    half_h = grasp_h / 2.0
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    polygon = [
        (cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t)
        for dx, dy in corners
    ]
    draw.line(polygon + [polygon[0]], fill=color, width=width)
    center = (cx, cy)
    draw.ellipse((center[0] - 3, center[1] - 3, center[0] + 3, center[1] + 3), fill=color)


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: str, width: int = 2) -> None:
    x0, y0, x1, y1 = [float(value) for value in box]
    for offset in range(width):
        draw.rectangle((x0 + offset, y0 + offset, x1 - offset, y1 - offset), outline=color)


def prompt_template_for_method(method: str) -> str:
    if method == "pred_vcot":
        return PRED_VCOT_GRASP_PROMPT_TEMPLATE
    return DIRECT_GRASP_PROMPT_TEMPLATE


def bbox_prompt_template_for_method(method: str) -> str:
    if method == "pred_vcot":
        return PRED_VCOT_BBOX_PROMPT_TEMPLATE
    return ""


def prompt_for_output(method: str, output: dict[str, Any]) -> str:
    obj_name = output.get("obj_name", "{obj_name}")
    return prompt_template_for_method(method).format(obj_name=obj_name)


def csv_prompt(value: str) -> str:
    return value.replace("\n", "\\n")


def annotate(draw: ImageDraw.ImageDraw, text: str) -> None:
    font = ImageFont.load_default()
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=3)
    draw.rectangle((0, 0, bbox[2] + 8, bbox[3] + 8), fill=(255, 255, 255))
    draw.multiline_text((4, 4), text, fill=(0, 0, 0), font=font, spacing=3)


def target_grasp_from_output(output: dict[str, Any], image_size: int) -> list[float] | None:
    if output.get("target_full_grasp") is not None:
        return [float(value) for value in output["target_full_grasp"][:5]]
    if output.get("target_norm") is not None:
        return norm_to_grasp(output["target_norm"], image_size)
    return None


def render_output(output: dict[str, Any], image_txn, method: str, image_size: int) -> Image.Image:
    grasp_id = output["grasp_id"]
    scene_id = grasp_id.split("_")[0]
    image = load_image(image_txn, scene_id, image_size)
    draw = ImageDraw.Draw(image)

    target_grasp = target_grasp_from_output(output, image_size)
    if target_grasp is not None:
        draw_rotated_grasp(draw, target_grasp, "#2CA02C", width=3)

    pred_norm = output.get("pred_norm")
    if pred_norm is not None:
        draw_rotated_grasp(draw, norm_to_grasp(pred_norm, image_size), "#D62728", width=3)

    if method == "pred_vcot":
        pred_crop = output.get("pred_crop_box")
        gt_crop = output.get("gt_crop_box")
        if gt_crop:
            draw_box(draw, gt_crop, "#17BECF", width=2)
        if pred_crop:
            draw_box(draw, pred_crop, "#1F77B4", width=2)

    ok = "ok" if output.get("vcot_success") else "fail"
    prompt = prompt_for_output(method, output).replace("<image>", "[image]")
    annotate(draw, f"{method} | {output.get('split', '')} | {ok}\n{prompt}")
    return image


def select_examples(outputs: list[dict[str, Any]], per_group: int) -> list[dict[str, Any]]:
    valid = [item for item in outputs if item.get("pred_norm") is not None]
    success = [item for item in valid if item.get("vcot_success")]
    fail = [item for item in valid if not item.get("vcot_success")]
    return success[:per_group] + fail[:per_group]


def make_contact_sheet(images: list[Image.Image], columns: int = 4, pad: int = 10) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), "white")
    width, height = images[0].size
    rows = int(np.ceil(len(images) / columns))
    sheet = Image.new("RGB", (columns * width + (columns + 1) * pad, rows * height + (rows + 1) * pad), "white")
    for idx, image in enumerate(images):
        x = pad + (idx % columns) * (width + pad)
        y = pad + (idx // columns) * (height + pad)
        sheet.paste(image, (x, y))
    return sheet


def outputs_by_id(outputs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        item["grasp_id"]: item
        for item in outputs
        if item.get("grasp_id") and item.get("pred_norm") is not None
    }


def parse_categories(value: str) -> list[str]:
    categories = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(categories) - set(PAIR_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown pair categories: {unknown}. Expected one of: {', '.join(PAIR_CATEGORIES)}")
    return categories


def pair_category(direct: dict[str, Any], pred: dict[str, Any]) -> str:
    direct_success = bool(direct.get("vcot_success"))
    pred_success = bool(pred.get("vcot_success"))
    if pred_success and not direct_success:
        return "pred_rescue"
    if direct_success and not pred_success:
        return "direct_only"
    if direct_success and pred_success:
        return "both_success"
    return "both_fail"


def bool_for_csv(value: bool) -> int:
    return 1 if value else 0


def csv_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "show"}


def metric_value(output: dict[str, Any], key: str) -> str:
    value = output.get(key)
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def output_float(output: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = output.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rescue_score(direct: dict[str, Any], pred: dict[str, Any]) -> float:
    pred_iou = output_float(pred, "vcot_joint_iou")
    direct_iou = output_float(direct, "vcot_joint_iou")
    pred_angle = min(output_float(pred, "vcot_joint_angle_diff", 180.0), 180.0)
    direct_angle = min(output_float(direct, "vcot_joint_angle_diff", 180.0), 180.0)
    bbox_iou = output_float(pred, "pred_bbox_iou")
    angle_gain = (direct_angle - pred_angle) / 180.0
    return (2.0 * pred_iou) - direct_iou + angle_gain + (0.25 * bbox_iou)


def pair_row(
    split: str,
    category: str,
    grasp_id: str,
    direct_output: dict[str, Any],
    pred_output: dict[str, Any],
    direct_experiment: str,
    pred_experiment: str,
    show: bool,
) -> dict[str, Any]:
    obj_name = direct_output.get("obj_name") or pred_output.get("obj_name", "")
    return {
        "show": bool_for_csv(show),
        "split": split,
        "category": category,
        "outcome_category": pair_category(direct_output, pred_output),
        "grasp_id": grasp_id,
        "obj_name": obj_name,
        "direct_success": bool(direct_output.get("vcot_success")),
        "pred_vcot_success": bool(pred_output.get("vcot_success")),
        "rescue_score": f"{rescue_score(direct_output, pred_output):.4f}",
        "direct_joint_iou": metric_value(direct_output, "vcot_joint_iou"),
        "pred_vcot_joint_iou": metric_value(pred_output, "vcot_joint_iou"),
        "direct_best_iou": metric_value(direct_output, "vcot_best_iou"),
        "pred_vcot_best_iou": metric_value(pred_output, "vcot_best_iou"),
        "direct_angle_diff": metric_value(direct_output, "vcot_joint_angle_diff"),
        "pred_vcot_angle_diff": metric_value(pred_output, "vcot_joint_angle_diff"),
        "pred_bbox_iou": metric_value(pred_output, "pred_bbox_iou"),
        "direct_experiment": direct_experiment,
        "pred_vcot_experiment": pred_experiment,
        "direct_prompt": csv_prompt(prompt_for_output("direct", direct_output)),
        "pred_vcot_bbox_prompt": csv_prompt(
            PRED_VCOT_BBOX_PROMPT_TEMPLATE.format(obj_name=pred_output.get("obj_name", "{obj_name}"))
        ),
        "pred_vcot_grasp_prompt": csv_prompt(prompt_for_output("pred_vcot", pred_output)),
    }


def build_pair_candidates(
    split: str,
    direct_outputs: dict[str, dict[str, Any]],
    pred_outputs: dict[str, dict[str, Any]],
    direct_experiment: str,
    pred_experiment: str,
    categories: list[str],
    shown_per_category: int,
    candidate_limit: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    common_ids = [grasp_id for grasp_id in direct_outputs if grasp_id in pred_outputs]
    rng = random.Random(f"{random_seed}:{split}")
    rows: list[dict[str, Any]] = []
    if "random" in categories:
        random_ids = common_ids[:]
        rng.shuffle(random_ids)
        for count, grasp_id in enumerate(random_ids[:candidate_limit]):
            rows.append(
                pair_row(
                    split=split,
                    category="random",
                    grasp_id=grasp_id,
                    direct_output=direct_outputs[grasp_id],
                    pred_output=pred_outputs[grasp_id],
                    direct_experiment=direct_experiment,
                    pred_experiment=pred_experiment,
                    show=count < shown_per_category,
                )
            )

    for category in [item for item in categories if item != "random"]:
        category_ids = [
            grasp_id
            for grasp_id in common_ids
            if pair_category(direct_outputs[grasp_id], pred_outputs[grasp_id]) == category
        ]
        if category == "pred_rescue":
            category_ids.sort(
                key=lambda grasp_id: (
                    rescue_score(direct_outputs[grasp_id], pred_outputs[grasp_id]),
                    output_float(pred_outputs[grasp_id], "vcot_joint_iou"),
                    output_float(pred_outputs[grasp_id], "pred_bbox_iou"),
                    -output_float(pred_outputs[grasp_id], "vcot_joint_angle_diff", 180.0),
                ),
                reverse=True,
            )
        else:
            rng.shuffle(category_ids)
        for count, grasp_id in enumerate(category_ids[:candidate_limit]):
            rows.append(
                pair_row(
                    split=split,
                    category=category,
                    grasp_id=grasp_id,
                    direct_output=direct_outputs[grasp_id],
                    pred_output=pred_outputs[grasp_id],
                    direct_experiment=direct_experiment,
                    pred_experiment=pred_experiment,
                    show=count < shown_per_category,
                )
            )
    return rows


def make_pair_tile(
    direct_image: Image.Image,
    pred_image: Image.Image,
    category: str,
    grasp_id: str,
    obj_name: str,
    pad: int = 10,
) -> Image.Image:
    font = ImageFont.load_default()
    header_h = 26
    width = direct_image.width + pred_image.width + pad
    height = header_h + direct_image.height
    tile = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(tile)
    label = f"{category} | same grasp_id | {obj_name} | {grasp_id[:12]}"
    draw.text((4, 6), label, fill=(0, 0, 0), font=font)
    tile.paste(direct_image, (0, header_h))
    tile.paste(pred_image, (direct_image.width + pad, header_h))
    return tile


def pair_config_path(out_dir: Path, args: argparse.Namespace) -> Path:
    if args.pair_config:
        return Path(args.pair_config)
    return out_dir / "examples" / "paired_example_config.csv"


def write_paired_examples(df: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    categories = parse_categories(args.paired_categories)
    config_path = pair_config_path(out_dir, args)
    best = best_rows(df)
    pair_context: dict[str, dict[str, Any]] = {}
    candidate_rows: list[dict[str, Any]] = []

    for split in ["test_seen", "test_unseen"]:
        direct_match = best[(best["method"] == "direct") & (best["split"] == split)]
        pred_match = best[(best["method"] == "pred_vcot") & (best["split"] == split)]
        if direct_match.empty or pred_match.empty:
            continue

        direct_row = direct_match.iloc[0]
        pred_row = pred_match.iloc[0]
        direct_data = json.loads(Path(direct_row["result_path"]).read_text(encoding="utf-8"))
        pred_data = json.loads(Path(pred_row["result_path"]).read_text(encoding="utf-8"))
        direct_outputs = outputs_by_id(direct_data.get("outputs", []))
        pred_outputs = outputs_by_id(pred_data.get("outputs", []))
        pair_context[split] = {
            "direct_row": direct_row,
            "pred_row": pred_row,
            "direct_outputs": direct_outputs,
            "pred_outputs": pred_outputs,
        }
        candidate_rows.extend(
            build_pair_candidates(
                split=split,
                direct_outputs=direct_outputs,
                pred_outputs=pred_outputs,
                direct_experiment=str(direct_row["experiment"]),
                pred_experiment=str(pred_row["experiment"]),
                categories=categories,
                shown_per_category=max(0, args.paired_per_category),
                candidate_limit=max(1, args.paired_candidate_limit),
                random_seed=args.pair_random_seed,
            )
        )

    if args.refresh_pair_config or not config_path.exists():
        ensure_dir(config_path.parent)
        candidate_df = pd.DataFrame(candidate_rows)
        if not candidate_df.empty:
            split_order = {"test_seen": 0, "test_unseen": 1}
            category_order = {category: idx for idx, category in enumerate(PAIR_CATEGORIES)}
            candidate_df["_split_order"] = candidate_df["split"].map(split_order).fillna(99)
            candidate_df["_category_order"] = candidate_df["category"].map(category_order).fillna(99)
            candidate_df = candidate_df.sort_values(
                ["_split_order", "_category_order", "show"],
                ascending=[True, True, False],
            ).drop(columns=["_split_order", "_category_order"])
        candidate_df.to_csv(config_path, index=False)
    paths.append(config_path)

    config_df = pd.read_csv(config_path, dtype=str).fillna("")
    required = {"split", "grasp_id", "show"}
    missing = required - set(config_df.columns)
    if missing:
        raise ValueError(f"Pair config is missing required columns: {sorted(missing)}")
    selected_df = config_df[config_df["show"].map(csv_truthy)].copy()

    image_env = lmdb.open(str(Path(args.image_lmdb)), readonly=True, lock=False, readahead=False, meminit=False)
    with image_env.begin() as image_txn:
        for split in ["test_seen", "test_unseen"]:
            if split not in pair_context:
                continue
            context = pair_context[split]
            direct_row = context["direct_row"]
            pred_row = context["pred_row"]
            direct_outputs = context["direct_outputs"]
            pred_outputs = context["pred_outputs"]
            split_rows = selected_df[selected_df["split"] == split]

            tiles = []
            for _idx, selected in split_rows.iterrows():
                grasp_id = str(selected["grasp_id"])
                if grasp_id not in direct_outputs or grasp_id not in pred_outputs:
                    continue
                direct_output = direct_outputs[grasp_id]
                pred_output = pred_outputs[grasp_id]
                category = str(selected.get("category") or pair_category(direct_output, pred_output))
                direct_image = render_output(direct_output, image_txn, "direct", args.image_size)
                pred_image = render_output(pred_output, image_txn, "pred_vcot", args.image_size)
                obj_name = direct_output.get("obj_name") or pred_output.get("obj_name", "")
                tiles.append(make_pair_tile(direct_image, pred_image, category, grasp_id, obj_name))
                manifest_rows.append(
                    pair_row(
                        split=split,
                        category=category,
                        grasp_id=grasp_id,
                        direct_output=direct_output,
                        pred_output=pred_output,
                        direct_experiment=str(direct_row["experiment"]),
                        pred_experiment=str(pred_row["experiment"]),
                        show=True,
                    )
                )

            if tiles:
                sheet = make_contact_sheet(tiles, columns=2)
                path = out_dir / "examples" / f"paired_{split}_direct_vs_predicted_same_samples.png"
                ensure_dir(path.parent)
                sheet.save(path)
                paths.append(path)
    image_env.close()

    if manifest_rows:
        manifest_path = out_dir / "examples" / "paired_example_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        paths.append(manifest_path)
    return paths


def write_qualitative_examples(df: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> list[Path]:
    paths = []
    best = best_rows(df)
    image_env = lmdb.open(str(Path(args.image_lmdb)), readonly=True, lock=False, readahead=False, meminit=False)
    with image_env.begin() as image_txn:
        for row in best.itertuples():
            data = json.loads(Path(row.result_path).read_text(encoding="utf-8"))
            selected = select_examples(data.get("outputs", []), args.examples_per_group)
            images = [render_output(item, image_txn, row.method, args.image_size) for item in selected]
            sheet = make_contact_sheet(images)
            path = out_dir / "examples" / f"{row.method}_{row.split}_{row.experiment}.png"
            ensure_dir(path.parent)
            sheet.save(path)
            paths.append(path)
    image_env.close()
    return paths


def write_best_table(df: pd.DataFrame, out_dir: Path) -> Path:
    best = best_rows(df)
    columns = [
        "method",
        "split",
        "experiment",
        "vcot_success_rate_all",
        "vcot_top1_success_rate_all",
        "vcot_joint_iou_mean",
        "vcot_best_angle_diff_mean",
    ]
    table = best[columns].copy()
    table.insert(3, "bbox_prompt_template", [csv_prompt(bbox_prompt_template_for_method(method)) for method in table["method"]])
    table.insert(4, "grasp_prompt_template", [csv_prompt(prompt_template_for_method(method)) for method in table["method"]])
    for column in columns[3:]:
        table[column] = table[column].map(lambda value: f"{float(value):.4f}")
    path = out_dir / "best_direct_predicted_table.csv"
    table.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = load_summary(Path(args.summary_csv))
    paths: list[Path] = [
        plot_best_method_comparison(df, out_dir),
        plot_all_experiments(df, out_dir),
        plot_top1_vs_official(df, out_dir),
        write_best_table(df, out_dir),
    ]
    pipeline_path = plot_pipeline_diagnostics(Path(args.pipeline_csv), out_dir)
    if pipeline_path is not None:
        paths.append(pipeline_path)
    if not args.skip_examples:
        paths.extend(write_qualitative_examples(df, out_dir, args))
        paths.extend(write_paired_examples(df, out_dir, args))

    manifest = out_dir / "manifest.txt"
    manifest.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
    print(f"wrote {len(paths)} files")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
