#!/usr/bin/env python3
"""Plot lead-time curves for daily/monthly/yearly aggregations.

This script computes scores across lead time horizons (0, 15min, 30min, ..., 360min)
for a given day, month, or year by averaging over all initialization times.

For single dates: plots overlaid individual curves.
For multiple dates (month/year): plots median with 25th-75th and 5th-95th percentile bands.

Example
-------
    # Single day
    uv run python plot_leadtime_curves.py \
        --input results \
        --date 2025-01-15 \
        --output-dir plots

    # Entire month (shows median + percentiles)
    uv run python plot_leadtime_curves.py \
        --input results \
        --month 2025-01 \
        --output-dir plots

    # Entire year (shows median + percentiles)
    uv run python plot_leadtime_curves.py \
        --input results \
        --year 2025 \
        --output-dir plots

    # Multiple days overlaid
    uv run python plot_leadtime_curves.py \
        --input results \
        --date 2025-01-15 2025-01-16 2025-01-17 \
        --output-dir plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path to allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sunflow_scores import (
    _load_daily_csv,
    _metric_columns,
    _month_label_iso,
    _extract_date,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot lead-time curves for daily/monthly/yearly aggregations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing scores_*.csv files (supports monthly subfolders).",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where the plot will be written.",
    )
    parser.add_argument(
        "--metric",
        choices=["mae", "rmse", "both"],
        default="both",
        help="Metric to plot.",
    )

    # Date/month/year selection (mutually exclusive)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--date",
        nargs="+",
        help="One or more dates in YYYY-MM-DD format (e.g. 2025-01-15 or 2025-01-15 2025-01-16).",
    )
    date_group.add_argument(
        "--month",
        help="Month in YYYY-MM format (e.g. 2025-01).",
    )
    date_group.add_argument(
        "--year",
        help="Year in YYYY format (e.g. 2025).",
    )

    return parser.parse_args()




def _dates_from_month(month_str: str) -> list[str]:
    """Generate list of YYYYMMDD strings for all days in a month."""
    month = pd.Timestamp(month_str)
    days_in_month = pd.Period(month, freq='M').days_in_month
    dates = []
    for day in range(1, days_in_month + 1):
        date = month.replace(day=day)
        dates.append(date.strftime("%Y%m%d"))
    return dates


def _dates_from_year(year_str: str) -> list[str]:
    """Generate list of YYYYMMDD strings for all days in a year."""
    year = int(year_str)
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    date_range = pd.date_range(start, end, freq="D")
    return [d.strftime("%Y%m%d") for d in date_range]


def collect_leadtime_scores(
    input_dir: Path,
    dates: list[str],
    metric: str,
) -> dict[str, pd.DataFrame]:
    """
    Collect lead-time averaged scores for each requested date.

    Parameters
    ----------
    input_dir : Path
        Directory containing scores_*.csv files (may have monthly subfolders).
    dates : list[str]
        List of dates in YYYYMMDD format.
    metric : str
        "mae", "rmse", or "both".

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of date label to DataFrame with columns (lead_time_minutes, metric).
    """
    csv_files = sorted(input_dir.rglob("scores_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No scores_*.csv files found in {input_dir}")

    # Build a dict mapping YYYYMMDD -> path for quick lookup
    date_to_path = {}
    for path in csv_files:
        date = _extract_date(path)
        date_to_path[date] = path

    metric_cols = _metric_columns(metric)

    results = {}
    missing_dates = []

    for date in dates:
        if date not in date_to_path:
            missing_dates.append(date)
            continue

        df = _load_daily_csv(date_to_path[date])
        # Average over all initialization times for each lead time
        daily_avg = df.groupby("lead_time_minutes")[metric_cols].mean()
        results[date] = daily_avg

    if not results:
        if missing_dates:
            raise ValueError(
                f"No CSV files found for any of the requested dates: {dates}. "
                f"Missing: {missing_dates}"
            )
        raise ValueError(f"Could not process any dates from {dates}")

    if missing_dates:
        print(f"  WARNING: {len(missing_dates)} date(s) not found: {missing_dates}")

    return results


def plot_leadtime_curves(
    input_dir: Path,
    output_dir: Path,
    dates: list[str],
    metric: str,
    label_dates: bool = True,
) -> Path:
    """
    Plot lead-time curves for one or more dates.

    For single dates: overlaid line curves.
    For multiple dates (month/year): median line with percentile bands.

    Parameters
    ----------
    input_dir : Path
        Directory containing scores_*.csv files.
    output_dir : Path
        Directory where the plot will be written.
    dates : list[str]
        List of dates in YYYYMMDD format.
    metric : str
        "mae", "rmse", or "both".
    label_dates : bool
        Whether to label each curve with its date (single dates only).

    Returns
    -------
    Path
        Path to the saved figure.
    """
    data = collect_leadtime_scores(input_dir, dates, metric)

    metric_cols = _metric_columns(metric)
    labels = {"mae_by_init": "MAE", "rmse_by_init": "RMSE"}

    n_metrics = len(metric_cols)
    n_dates = len(data)

    # Determine if we should show percentiles (multiple days) or individual curves
    show_percentiles = n_dates > 1

    if metric == "both":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        axes = [ax]

    for idx, col in enumerate(metric_cols):
        ax = axes[idx] if metric == "both" else axes[0]

        if show_percentiles:
            # Multiple dates: show median and percentile bands
            # Collect all values per lead time
            lead_times = sorted(data[list(data.keys())[0]].index)
            values_by_lt = {lt: [] for lt in lead_times}

            for date in data.keys():
                df = data[date]
                for lt in lead_times:
                    if lt in df.index:
                        values_by_lt[lt].append(df.loc[lt, col])

            # Compute median and percentiles
            lead_times_list = []
            median_vals = []
            p25_vals = []
            p75_vals = []
            p5_vals = []
            p95_vals = []

            for lt in sorted(values_by_lt.keys()):
                if values_by_lt[lt]:
                    vals = pd.Series(values_by_lt[lt])
                    lead_times_list.append(lt)
                    median_vals.append(vals.median())
                    p25_vals.append(vals.quantile(0.25))
                    p75_vals.append(vals.quantile(0.75))
                    p5_vals.append(vals.quantile(0.05))
                    p95_vals.append(vals.quantile(0.95))

            # Plot percentile bands and median
            ax.fill_between(lead_times_list, p5_vals, p95_vals, alpha=0.2, label="5th-95th percentile")
            ax.fill_between(lead_times_list, p25_vals, p75_vals, alpha=0.4, label="25th-75th percentile")
            ax.plot(lead_times_list, median_vals, color="C0", linewidth=2.5, marker="o", markersize=5, label="Median")

        else:
            # Single date: show individual curve
            for date in sorted(data.keys()):
                df = data[date]
                label = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if label_dates else None
                ax.plot(
                    df.index,
                    df[col].values,
                    marker="o",
                    label=label,
                    linewidth=2,
                    markersize=4,
                )

        ax.set_title(f"{labels[col]} across lead times")
        ax.set_xlabel("Lead time (minutes)")
        ax.set_ylabel(f"{labels[col]} (W/m²)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output filename with date range
    if len(dates) == 1:
        date_label = dates[0]
    else:
        # Multiple dates: show the actual range from available data
        first_date = min(sorted(data.keys()))
        last_date = max(sorted(data.keys()))
        if first_date[:6] == last_date[:6]:
            # Same month: show YYYYMM_DDDD format
            date_label = f"{first_date[:6]}_{first_date[6:8]}to{last_date[6:8]}"
        elif first_date[:4] == last_date[:4]:
            # Same year: show YYYY_MMDD format
            date_label = f"{first_date[:4]}_{first_date[4:6]}{first_date[6:8]}to{last_date[4:6]}{last_date[6:8]}"
        else:
            # Different years: show full date range
            date_label = f"{first_date}to{last_date}"

    # Use descriptive metric label for filename
    metric_label = "mae_rmse" if metric == "both" else metric
    out_path = output_dir / f"leadtime_curves_{date_label}_{metric_label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input must be a directory: {input_dir}")

    # Determine which dates to process
    if args.date:
        dates = []
        for date_str in args.date:
            date_obj = pd.Timestamp(date_str)
            dates.append(date_obj.strftime("%Y%m%d"))
    elif args.month:
        dates = _dates_from_month(args.month)
    else:  # args.year
        dates = _dates_from_year(args.year)

    print(f"Processing {len(dates)} date(s)...")
    out_path = plot_leadtime_curves(input_dir, output_dir, dates, args.metric)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
