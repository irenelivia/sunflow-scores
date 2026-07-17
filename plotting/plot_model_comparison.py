#!/usr/bin/env python3
"""Compare validation scores of two (or more) nowcast model versions.

Each model version has its own directory of daily validation CSVs named:
    scores_YYYYMMDD.csv

with the schema written by ``run_validation.py``:
    initialization_time, ensemble, valid_time, lead_time_minutes,
    mae_by_init, rmse_by_init

This script overlays several labelled model directories on a single figure. It
supports two modes matching the two requested comparison figures:

  * ``leadtime-line``  — mean score across forecast horizons 0 -> 360 min, one
    line per model (e.g. v1.0.0 vs v1.0.1).
  * ``monthly-bars``   — mean score per calendar month for a chosen lead time
    (e.g. 15 and 30 min), grouped bars with one bar per model per month.

Examples
--------
Lead-time line graph (0-360 min), MAE and RMSE, two models:
    uv run python plot_model_comparison.py \
        --inputs results/v1.0.0 results/v1.0.1 \
        --labels v1.0.0 v1.0.1 \
        --mode leadtime-line \
        --metric both \
        --output-dir plots

Monthly bars at the 15- and 30-minute horizons:
    uv run python plot_model_comparison.py \
        --inputs results/v1.0.0 results/v1.0.1 \
        --labels v1.0.0 v1.0.1 \
        --mode monthly-bars \
        --lead-time 15,30 \
        --metric both \
        --output-dir plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path to allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sunflow_scores import (
    _load_daily_csv,
    _metric_columns,
    _month_from_path,
    _month_label_iso,
)

METRIC_LABELS = {"mae_by_init": "MAE", "rmse_by_init": "RMSE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare validation scores of two or more model versions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One directory of scores_*.csv per model version.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="A label per --inputs directory (e.g. v1.0.0 v1.0.1).",
    )
    parser.add_argument(
        "--mode",
        choices=["leadtime-line", "monthly-bars"],
        required=True,
        help="leadtime-line: mean score per lead time (0-360). "
             "monthly-bars: mean score per calendar month for a chosen lead time.",
    )
    parser.add_argument(
        "--metric",
        choices=["mae", "rmse", "both"],
        default="both",
        help="Metric(s) to plot.",
    )
    parser.add_argument(
        "--lead-time",
        default="15,30",
        help="For monthly-bars: comma-separated lead time(s) in minutes, e.g. 15,30. "
             "Ignored for leadtime-line.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where the plot(s) will be written.",
    )
    return parser.parse_args()


# Local helpers for this script
def _month_label(month: str) -> str:
    """Turn a YYYYMM tag into a readable 'YYYY-MM' label."""
    return _month_label_iso(month)


def _parse_lead_times(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError(f"Could not parse any lead time from '{raw}'.")
    return values


# -----------------------------------------------------------------------------
# Aggregation core
# -----------------------------------------------------------------------------
def collect_leadtime_curve(input_dir: Path, metric: str) -> pd.DataFrame:
    """Mean score per lead_time_minutes across all daily CSVs.

    Returns a DataFrame indexed by lead_time_minutes with one column per metric.
    """
    csv_files = sorted(input_dir.rglob("scores_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No scores_*.csv files found in {input_dir}")

    metric_cols = _metric_columns(metric)
    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = _load_daily_csv(path)
        daily = df.groupby("lead_time_minutes", as_index=False)[metric_cols].mean()
        frames.append(daily)

    summary = pd.concat(frames, ignore_index=True)
    average = (
        summary.groupby("lead_time_minutes", as_index=True)[metric_cols]
        .mean()
        .sort_index()
    )
    return average


def collect_monthly_for_leadtime(
    input_dir: Path, lead_time: int, metric: str
) -> pd.DataFrame:
    """Mean score per calendar month for a chosen lead time.

    Returns a DataFrame indexed by month (YYYYMM) with one column per metric.
    Raises ValueError if the requested lead time is absent from all CSVs.
    """
    csv_files = sorted(input_dir.rglob("scores_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No scores_*.csv files found in {input_dir}")

    metric_cols = _metric_columns(metric)
    rows: list[pd.DataFrame] = []
    for path in csv_files:
        df = _load_daily_csv(path)
        selected = df[df["lead_time_minutes"] == lead_time]
        if selected.empty:
            continue
        month = _month_from_path(path)
        monthly = selected[metric_cols].mean().to_frame().T
        monthly.insert(0, "month", month)
        rows.append(monthly)

    if not rows:
        raise ValueError(
            f"No rows with lead_time_minutes == {lead_time} found in any CSV "
            f"under {input_dir}."
        )

    combined = pd.concat(rows, ignore_index=True)
    monthly_mean = (
        combined.groupby("month", as_index=True)[metric_cols].mean().sort_index()
    )
    return monthly_mean


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_leadtime_line(
    curves_by_label: dict[str, pd.DataFrame],
    metric: str,
    output_dir: Path,
) -> Path:
    """One line per (model, metric) across lead time (0-360 min)."""
    metric_cols = _metric_columns(metric)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    line_styles = {"mae_by_init": "-", "rmse_by_init": "--"}
    markers = {"mae_by_init": "o", "rmse_by_init": "s"}

    for model_idx, (label, curve) in enumerate(curves_by_label.items()):
        color = color_cycle[model_idx % len(color_cycle)] if color_cycle else None
        for col in metric_cols:
            if len(metric_cols) > 1:
                series_label = f"{label} — {METRIC_LABELS[col]}"
            else:
                series_label = label
            ax.plot(
                curve.index,
                curve[col].values,
                marker=markers[col],
                linestyle=line_styles[col],
                color=color,
                label=series_label,
            )

    metric_title = {
        "mae": "MAE",
        "rmse": "RMSE",
        "both": "MAE & RMSE",
    }[metric]
    ax.set_title(f"Mean {metric_title} per lead time — model comparison")
    ax.set_xlabel("Lead time (minutes)")
    ax.set_ylabel("Score (W/m²)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"comparison_leadtime_line_{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_monthly_bars(
    monthly_by_label: dict[str, pd.DataFrame],
    lead_time: int,
    metric: str,
    output_dir: Path,
) -> Path:
    """Grouped bars: x = calendar months, one bar per model (and per metric)."""
    metric_cols = _metric_columns(metric)

    # Union of all months across models, sorted chronologically.
    months: list[str] = sorted(
        {m for df in monthly_by_label.values() for m in df.index}
    )
    x = np.arange(len(months))

    labels = list(monthly_by_label.keys())
    # One bar group per (model, metric) combination.
    series = [(label, col) for label in labels for col in metric_cols]
    n_series = len(series)
    bar_width = 0.8 / max(n_series, 1)

    fig, ax = plt.subplots(
        1, 1, figsize=(max(10, len(months) * 1.1), 6), constrained_layout=True
    )

    for i, (label, col) in enumerate(series):
        df = monthly_by_label[label]
        # Align to the shared month axis; missing months become NaN (no bar).
        values = [df.loc[m, col] if m in df.index else np.nan for m in months]
        offset = (i - (n_series - 1) / 2) * bar_width
        if len(metric_cols) > 1:
            series_label = f"{label} — {METRIC_LABELS[col]}"
        else:
            series_label = label
        ax.bar(x + offset, values, width=bar_width, label=series_label)

    ax.set_title(f"Monthly mean score at {lead_time}-minute lead time — model comparison")
    ax.set_xlabel("Month")
    ax.set_ylabel("Score (W/m²)")
    ax.set_xticks(x)
    ax.set_xticklabels([_month_label(m) for m in months], rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"comparison_monthly_leadtime_{lead_time}min_{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError(
            f"--inputs and --labels must have the same number of entries: "
            f"got {len(args.inputs)} inputs and {len(args.labels)} labels."
        )

    input_dirs = [Path(p) for p in args.inputs]
    for d in input_dirs:
        if not d.is_dir():
            raise NotADirectoryError(
                f"--inputs entry is not a directory of scores_*.csv files: {d}"
            )

    output_dir = Path(args.output_dir)

    if args.mode == "leadtime-line":
        curves_by_label: dict[str, pd.DataFrame] = {}
        for label, d in zip(args.labels, input_dirs):
            curves_by_label[label] = collect_leadtime_curve(d, args.metric)
        out_path = plot_leadtime_line(curves_by_label, args.metric, output_dir)
        print(f"Wrote {out_path}")
        return

    # monthly-bars
    lead_times = _parse_lead_times(args.lead_time)
    for lead_time in lead_times:
        monthly_by_label: dict[str, pd.DataFrame] = {}
        for label, d in zip(args.labels, input_dirs):
            monthly_by_label[label] = collect_monthly_for_leadtime(
                d, lead_time, args.metric
            )
        out_path = plot_monthly_bars(
            monthly_by_label, lead_time, args.metric, output_dir
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
