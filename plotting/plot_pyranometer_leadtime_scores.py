#!/usr/bin/env python3
"""Plot MAE/RMSE vs lead time for DINI and the satellite nowcast, and mark
the lead time at which DINI overtakes the nowcast.

Reads dini_by_init.csv and nowcast_by_init.csv written by
run_validation_pyranometer.py, averages each metric over all initialization
times (and stations, already averaged in the *_by_init files) per lead time,
and overlays both curves. The nowcast is only available out to its own max
lead time (e.g. 6h); the crossover is only searched for within the lead-time
range common to both sources.

Both input CSVs already exclude lead times shorter than each forecast's
real-world availability latency (DINI: 3h, nowcast: 30min -- see
sunflow_scores.time_alignment.filter_usable_lead_times), so the DINI curve
here starts at 3h and the nowcast curve at 30min; a "crossover" below 3h is
not possible by construction.

Example
-------
    uv run python plotting/plot_pyranometer_leadtime_scores.py \
        --input-dir results/pyranometer \
        --metric both \
        --output-dir plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MAE/RMSE vs lead time for DINI vs the satellite nowcast.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory with dini_by_init.csv and nowcast_by_init.csv (from run_validation_pyranometer.py).",
    )
    parser.add_argument("--output-dir", default="plots", help="Directory where the plot will be written.")
    parser.add_argument("--metric", choices=["mae", "rmse", "both"], default="both", help="Metric to plot.")
    return parser.parse_args()


def _mean_by_leadtime(csv_path: Path, metric: str) -> pd.Series:
    """Average a metric over all initialization times, indexed by lead_time_minutes."""
    df = pd.read_csv(csv_path)
    return df.groupby("lead_time_minutes")[metric].mean().sort_index()


def _find_crossover(nowcast: pd.Series, dini: pd.Series) -> tuple[float | None, bool]:
    """
    Lead time (minutes) within the shared range where DINI's error first drops
    to or below the nowcast's, linearly interpolated between the surrounding
    grid points.

    Returns (crossover_lead_time, already_ahead_at_start):
      - (lead_time, False): DINI starts out worse and overtakes at lead_time.
      - (None, True):       DINI is at or below the nowcast already at the
                             shortest shared lead time (no "overtake" event).
      - (None, False):      DINI never catches up within the shared range.
    """
    common_index = nowcast.index.intersection(dini.index)
    if len(common_index) < 2:
        return None, False
    diff = (dini.loc[common_index] - nowcast.loc[common_index]).sort_index()

    if diff.iloc[0] <= 0:
        return None, True

    for i in range(1, len(diff)):
        prev_lt, prev_diff = diff.index[i - 1], diff.iloc[i - 1]
        curr_lt, curr_diff = diff.index[i], diff.iloc[i]
        if prev_diff > 0 and curr_diff <= 0:
            frac = prev_diff / (prev_diff - curr_diff)
            return prev_lt + frac * (curr_lt - prev_lt), False
    return None, False


def plot_metric(metric: str, nowcast: pd.Series, dini: pd.Series, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.plot(nowcast.index, nowcast.values, marker="o", color="C0", label="Satellite nowcast")
    ax.plot(dini.index, dini.values, marker="o", color="C1", label="DINI")

    crossover, already_ahead = _find_crossover(nowcast, dini)
    if crossover is not None:
        ax.axvline(crossover, color="grey", linestyle="--", linewidth=1)
        ax.annotate(
            f"DINI overtakes\nnowcast at {crossover:.0f} min",
            xy=(crossover, ax.get_ylim()[1] * 0.9),
            xytext=(crossover + 10, ax.get_ylim()[1] * 0.9),
            fontsize=9, color="grey",
        )
    elif already_ahead:
        ax.text(
            0.02, 0.95, "DINI is already at or below the nowcast's error at the shortest shared lead time",
            transform=ax.transAxes, fontsize=9, color="grey", va="top",
        )
    else:
        ax.text(
            0.02, 0.95, "DINI does not overtake the nowcast within the shared lead-time range",
            transform=ax.transAxes, fontsize=9, color="grey", va="top",
        )

    ax.set_xlabel("Lead time (minutes)")
    ax.set_ylabel(f"{metric.upper()} (W/m$^2$)")
    ax.set_title(f"{metric.upper()} vs lead time: DINI vs satellite nowcast")
    ax.legend()
    ax.grid(alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"leadtime_scores_dini_vs_nowcast_{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_path}")
    if crossover is not None:
        print(f"  {metric.upper()} crossover lead time: {crossover:.1f} min")
    elif already_ahead:
        print(f"  {metric.upper()}: DINI already ahead at the shortest shared lead time.")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)

    dini_csv = input_dir / "dini_by_init.csv"
    nowcast_csv = input_dir / "nowcast_by_init.csv"
    if not dini_csv.exists() or not nowcast_csv.exists():
        raise FileNotFoundError(
            f"Both dini_by_init.csv and nowcast_by_init.csv are required in {input_dir}."
        )

    metrics = ["mae", "rmse"] if args.metric == "both" else [args.metric]
    for metric in metrics:
        nowcast = _mean_by_leadtime(nowcast_csv, metric)
        dini = _mean_by_leadtime(dini_csv, metric)
        plot_metric(metric, nowcast, dini, Path(args.output_dir))


if __name__ == "__main__":
    main()
