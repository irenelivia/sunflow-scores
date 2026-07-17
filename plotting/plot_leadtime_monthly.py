#!/usr/bin/env python3
"""Plot per-month validation scores for a single chosen lead time.

The validation pipeline writes one CSV per day named like:
    scores_YYYYMMDD.csv

Each file contains by-init scores and a ``lead_time_minutes`` column. This
script extracts only the lead time of choice (e.g. 60 minutes) from every daily
CSV, aggregates the score across all days within each calendar month, and draws
one bar per month.

Example
-------
    uv run python plot_leadtime_monthly.py \
        --input results \
        --lead-time 60 \
        --metric both \
        --output-dir results/plots
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-month scores for a single chosen lead time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing scores_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where the plot will be written.",
    )
    parser.add_argument(
        "--lead-time",
        type=int,
        required=True,
        help="Lead time in minutes to extract from each daily CSV (e.g. 60).",
    )
    parser.add_argument(
        "--metric",
        choices=["mae", "rmse", "both"],
        default="both",
        help="Metric to plot.",
    )
    return parser.parse_args()


def _month_label(month: str) -> str:
    """Turn a YYYYMM tag into a readable 'YYYY-MM' label."""
    return _month_label_iso(month)


def collect_monthly_scores(input_dir: Path, lead_time: int, metric: str) -> pd.DataFrame:
    """Extract the chosen lead time from every CSV and average per month.

    Returns a DataFrame indexed by month (YYYYMM) with one column per metric.
    """
    csv_files = sorted(input_dir.rglob("scores_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No scores_*.csv files found in {input_dir}")

    metric_cols = _metric_columns(metric)
    rows: list[pd.DataFrame] = []
    for path in csv_files:
        df = _load_daily_csv(path)
        df = _filter_nowcasts_with_nan_at_leadtime_0(df)
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
    monthly_mean = combined.groupby("month", as_index=True)[metric_cols].mean().sort_index()
    return monthly_mean


def plot_leadtime_monthly(
    input_dir: Path,
    output_dir: Path,
    lead_time: int,
    metric: str,
) -> Path:
    monthly = collect_monthly_scores(input_dir, lead_time, metric)

    metric_cols = _metric_columns(metric)
    labels = {"mae_by_init": "MAE", "rmse_by_init": "RMSE"}

    months = list(monthly.index)
    x = np.arange(len(months))
    n_metrics = len(metric_cols)
    bar_width = 0.8 / n_metrics

    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(months) * 0.8), 6), constrained_layout=True)
    for i, col in enumerate(metric_cols):
        offset = (i - (n_metrics - 1) / 2) * bar_width
        ax.bar(x + offset, monthly[col].values, width=bar_width, label=labels[col])

    ax.set_title(f"Monthly mean score at {lead_time}-minute lead time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Score (W/m²)")
    ax.set_xticks(x)
    ax.set_xticklabels([_month_label(m) for m in months], rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0, top=100)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"monthly_scores_leadtime_{lead_time}min_{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input must be a directory of scores_*.csv files: {input_dir}")

    out_path = plot_leadtime_monthly(input_dir, output_dir, args.lead_time, args.metric)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
