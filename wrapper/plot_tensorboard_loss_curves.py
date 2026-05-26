from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from grasp_settings import build_settings  # noqa: E402


SETTINGS = build_settings()
DEFAULT_OUT_DIR = REPO_ROOT / "wrapper/tensorboard_loss_figures"
DEFAULT_TAG = "train/loss"

DEFAULT_GROUPS = {
    "direct": Path(SETTINGS["GRASP_DIRECT_RUN_ROOT"]),
    "crop": Path(SETTINGS["GRASP_CROP_RUN_ROOT"]),
    "predict": Path(SETTINGS["GRASP_VCOT_RUN_ROOT"]),
}

GROUP_TITLES = {
    "direct": "Direct",
    "crop": "Crop",
    "predict": "Predicted VCoT",
}


@dataclass(frozen=True)
class RunScalars:
    group: str
    experiment: str
    run_name: str
    run_dir: Path
    last_step: int
    num_points: int
    mtime: float
    data: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/loss curves from TensorBoard event logs.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tag", default=DEFAULT_TAG, help="TensorBoard scalar tag to plot.")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=50,
        help="Rolling mean window for plotted curves. Use 1 to plot raw values.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override or add a group directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract TensorBoard scalar data to CSV. Does not require matplotlib.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only plot existing CSV data. Does not require tensorboard.",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="CSV used by --plot-only. Defaults to <out-dir>/train_loss_curves.csv.",
    )
    return parser.parse_args()


def parse_groups(values: list[str]) -> dict[str, Path]:
    groups = dict(DEFAULT_GROUPS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --group value: {value}. Expected NAME=PATH.")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --group value: {value}. Group name is empty.")
        groups[name] = Path(path).expanduser().resolve()
    return groups


def run_dirs_with_events(group_dir: Path) -> list[Path]:
    if not group_dir.exists():
        return []
    return sorted({path.parent for path in group_dir.rglob("events.out.tfevents*")})


def load_run_scalars(group: str, group_dir: Path, run_dir: Path, tag: str) -> RunScalars | None:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The tensorboard package is required for extraction. "
            "Run this step inside the 260513-internvl conda environment."
        ) from exc

    rel_parts = run_dir.relative_to(group_dir).parts
    if not rel_parts:
        return None
    experiment = rel_parts[0]
    run_name = "/".join(rel_parts[2:]) if len(rel_parts) > 2 and rel_parts[1] == "runs" else "/".join(rel_parts[1:])
    if not run_name:
        run_name = run_dir.name

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return None

    scalars = accumulator.Scalars(tag)
    if not scalars:
        return None

    data = pd.DataFrame(
        {
            "group": group,
            "experiment": experiment,
            "run_name": run_name,
            "step": [item.step for item in scalars],
            "wall_time": [item.wall_time for item in scalars],
            "value": [item.value for item in scalars],
            "run_dir": str(run_dir),
        }
    )
    event_mtime = max(path.stat().st_mtime for path in run_dir.glob("events.out.tfevents*"))
    return RunScalars(
        group=group,
        experiment=experiment,
        run_name=run_name,
        run_dir=run_dir,
        last_step=int(data["step"].max()),
        num_points=len(data),
        mtime=event_mtime,
        data=data,
    )


def select_runs(runs: list[RunScalars]) -> list[RunScalars]:
    selected = []
    for (_group, _experiment), group_runs in pd.DataFrame(
        [
            {
                "idx": idx,
                "group": run.group,
                "experiment": run.experiment,
                "last_step": run.last_step,
                "num_points": run.num_points,
                "mtime": run.mtime,
            }
            for idx, run in enumerate(runs)
        ]
    ).groupby(["group", "experiment"]):
        row = group_runs.sort_values(["last_step", "num_points", "mtime"], ascending=False).iloc[0]
        selected.append(runs[int(row["idx"])])
    return sorted(selected, key=lambda item: (item.group, item.experiment))


def short_label(name: str) -> str:
    replacements = {
        "baseline_": "base_",
        "lora": "L",
        "patch": "p",
        "edge": "e",
        "half": "h",
        "bbox": "b",
        "crop_": "C-",
        "vcot_": "P-",
    }
    label = name
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def save_group_plot(group: str, data: pd.DataFrame, out_dir: Path, tag: str, smooth_window: int) -> Path | None:
    if data.empty:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    colors = plt.get_cmap("tab20").colors
    for idx, (experiment, curve) in enumerate(data.groupby("experiment", sort=True)):
        curve = curve.sort_values("step")
        values = curve["value"]
        if smooth_window > 1:
            values = values.rolling(window=smooth_window, min_periods=1).mean()
        ax.plot(
            curve["step"],
            values,
            linewidth=1.5,
            alpha=0.9,
            color=colors[idx % len(colors)],
            label=short_label(str(experiment)),
        )

    title = GROUP_TITLES.get(group, group)
    ax.set_title(f"{title} {tag}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.24, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{group}_loss_curves.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_manifest(
    out_dir: Path,
    paths: list[Path],
    selected_runs: list[RunScalars] | pd.DataFrame | None,
    tag: str,
    smooth_window: int,
) -> Path:
    manifest = out_dir / "manifest.txt"
    lines = [
        f"tag: {tag}",
        f"smooth_window: {smooth_window}",
        "",
        "figures:",
        *[str(path.relative_to(REPO_ROOT)) for path in paths],
        "",
        "selected runs:",
    ]
    if isinstance(selected_runs, pd.DataFrame):
        for row in selected_runs.itertuples(index=False):
            lines.append(
                f"{row.group}\t{row.experiment}\t{row.run_name}\tlast_step={row.last_step}\tpoints={row.num_points}\t{row.run_dir}"
            )
    elif selected_runs:
        for run in selected_runs:
            lines.append(
                f"{run.group}\t{run.experiment}\t{run.run_name}\tlast_step={run.last_step}\tpoints={run.num_points}\t{run.run_dir}"
            )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def save_group_plots(data: pd.DataFrame, groups: dict[str, Path], out_dir: Path, tag: str, smooth_window: int) -> list[Path]:
    paths = []
    for group in groups:
        group_data = data[data["group"] == group]
        path = save_group_plot(group, group_data, out_dir, tag, smooth_window)
        if path is not None:
            paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    if args.extract_only and args.plot_only:
        raise SystemExit("--extract-only and --plot-only cannot be used together.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    groups = parse_groups(args.group)

    if args.plot_only:
        input_csv = Path(args.input_csv).expanduser().resolve() if args.input_csv else out_dir / "train_loss_curves.csv"
        if not input_csv.exists():
            raise SystemExit(f"Input CSV not found: {input_csv}")
        out_dir.mkdir(parents=True, exist_ok=True)
        selected_df = pd.read_csv(input_csv)
        paths = save_group_plots(selected_df, groups, out_dir, args.tag, args.smooth_window)
        summary_path = out_dir / "train_loss_run_summary.csv"
        selected_runs = pd.read_csv(summary_path) if summary_path.exists() else None
        manifest = write_manifest(out_dir, paths, selected_runs, args.tag, args.smooth_window)
        for path in paths:
            print(path)
        print(manifest)
        return

    all_runs: list[RunScalars] = []
    for group, group_dir in groups.items():
        for run_dir in run_dirs_with_events(group_dir):
            run = load_run_scalars(group, group_dir, run_dir, args.tag)
            if run is not None:
                all_runs.append(run)

    if not all_runs:
        group_text = ", ".join(f"{name}={path}" for name, path in groups.items())
        raise SystemExit(f"No TensorBoard scalar data found for tag {args.tag!r}. Groups: {group_text}")

    selected_runs = select_runs(all_runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_df = pd.concat([run.data for run in selected_runs], ignore_index=True)
    selected_df.to_csv(out_dir / "train_loss_curves.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "group": run.group,
                "experiment": run.experiment,
                "run_name": run.run_name,
                "last_step": run.last_step,
                "num_points": run.num_points,
                "run_dir": str(run.run_dir),
            }
            for run in selected_runs
        ]
    )
    summary_df.to_csv(out_dir / "train_loss_run_summary.csv", index=False)

    if args.extract_only:
        manifest = write_manifest(out_dir, [], selected_runs, args.tag, args.smooth_window)
        print(out_dir / "train_loss_curves.csv")
        print(out_dir / "train_loss_run_summary.csv")
        print(manifest)
        return

    paths = save_group_plots(selected_df, groups, out_dir, args.tag, args.smooth_window)

    manifest = write_manifest(out_dir, paths, selected_runs, args.tag, args.smooth_window)
    for path in paths:
        print(path)
    print(manifest)


if __name__ == "__main__":
    main()
