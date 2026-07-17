#!/usr/bin/env python3
"""Plot seasonal validation diurnal cycles from daily CSV outputs.

This script reads daily scores_YYYYMMDD.csv files, builds each month's
diurnal-cycle average, then averages those monthly curves into the four
meteorological seasons:

- DJF: December, January, February
- MAM: March, April, May
- JJA: June, July, August
- SON: September, October, November

The output is a 2x2 figure with one panel per season.

This script can use either traditional meteorological seasons (DJF, MAM, JJA, SON)
or shifted calendar windows (JFM, AMJ, JAS, OND).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Add parent directory to path to allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sunflow_scores import (
    _load_daily_csv,
    _metric_columns,
    _month_from_path,
    _month_label_short,
)


SEASON_SCHEMES = {
    "meteorological": {
        "order": ["MAM", "JJA", "SON", "DJF"],
        "month_to_season": {
            1: "DJF",
            2: "DJF",
            3: "MAM",
            4: "MAM",
            5: "MAM",
            6: "JJA",
            7: "JJA",
            8: "JJA",
            9: "SON",
            10: "SON",
            11: "SON",
            12: "DJF",
        },
        "season_month_order": {
            "DJF": [12, 1, 2],
            "MAM": [3, 4, 5],
            "JJA": [6, 7, 8],
            "SON": [9, 10, 11],
        },
    },
    "quarterly": {
        "order": ["JFM", "AMJ", "JAS", "OND"],
        "month_to_season": {
            1: "JFM",
            2: "JFM",
            3: "JFM",
            4: "AMJ",
            5: "AMJ",
            6: "AMJ",
            7: "JAS",
            8: "JAS",
            9: "JAS",
            10: "OND",
            11: "OND",
            12: "OND",
        },
        "season_month_order": {
            "JFM": [1, 2, 3],
            "AMJ": [4, 5, 6],
            "JAS": [7, 8, 9],
            "OND": [10, 11, 12],
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot seasonal diurnal-cycle averages from daily validation CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="/dmidata/projects/weather2x/Energivejr_historical_data/sunflow_validation_scores/v1.0.0/2025",
        help="Directory containing scores_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where the seasonal plot will be written.",
    )
    parser.add_argument(
        "--metric",
        choices=["mae", "rmse", "both"],
        default="mae",
        help="Metric to plot.",
    )
    parser.add_argument(
        "--season-scheme",
        choices=["meteorological", "quarterly"],
        default="meteorological",
        help="Season grouping scheme: meteorological=DJF/MAM/JJA/SON, quarterly=JFM/AMJ/JAS/OND.",
    )
    parser.add_argument(
        "--max-lead-time-minutes",
        type=int,
        default=90,
        help="Maximum lead time to include in the diurnal-cycle curves.",
    )
    return parser.parse_args()


def _month_label(month_key: str) -> str:
    return _month_label_short(month_key)


def _season_month_sort_key(season: str, month_key: str, season_month_order: dict[str, list[int]]) -> tuple[int, int]:
    month_num = int(month_key[4:6])
    season_order = season_month_order[season]
    return (season_order.index(month_num), int(month_key[:4]))


def _season_for_month(month: int, month_to_season: dict[int, str]) -> str:
    return month_to_season[month]


def _prepare_monthly_diurnal_cycle(paths: list[Path], metric_col: str) -> pd.DataFrame:
    monthly_frames = []
    for path in paths:
        df = _load_daily_csv(path)
        daily = df[["initialization_time", "lead_time_minutes", metric_col]].copy()
        daily["init_time"] = daily["initialization_time"].dt.floor("h").dt.strftime("%H:%M")
        daily = daily.groupby(["init_time", "lead_time_minutes"], as_index=False)[[metric_col]].mean()
        monthly_frames.append(daily)

    summary = pd.concat(monthly_frames, ignore_index=True)
    return summary.groupby(["init_time", "lead_time_minutes"], as_index=False)[[metric_col]].mean()


def _combine_monthly_curves(monthly_frames: list[pd.DataFrame], metric_col: str) -> pd.DataFrame:
    summary = pd.concat(monthly_frames, ignore_index=True)
    return summary.groupby(["init_time", "lead_time_minutes"], as_index=False)[[metric_col]].mean()


def _minutes_formatter(value: float, _pos: int | None = None) -> str:
    total_minutes = int(round(value))
    hour, minute = divmod(total_minutes, 60)
    return f"{hour:02d}:{minute:02d}"


def _plot_season_panels(
    season_tables: dict[str, tuple[pd.DataFrame, list[str]]],
    season_order: list[str],
    metric: str,
    output_dir: Path,
    max_lead_time_minutes: int,
    season_scheme: str,
) -> Path:
    metric_col = _metric_columns(metric)[0]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True, sharex=True, sharey=True)
    axes = axes.ravel()

    x_max = 24 * 60 + max_lead_time_minutes
    x_ticks = np.arange(0, 24 * 60 + 1, 180)

    for ax, season in zip(axes, season_order):
        table, months = season_tables[season]
        if table.empty:
            ax.set_axis_off()
            continue

        lead_times = sorted(table["lead_time_minutes"].unique())
        lead_times = [lead for lead in lead_times if lead <= max_lead_time_minutes]
        table = table[table["lead_time_minutes"] <= max_lead_time_minutes]

        init_times = sorted(table["init_time"].unique(), key=lambda value: int(value[:2]))
        for init_time in init_times:
            series = (
                table.loc[table["init_time"] == init_time]
                .set_index("lead_time_minutes")[metric_col]
                .reindex(lead_times)
            )
            hour = int(init_time[:2])
            x_values = [hour * 60 + lead for lead in lead_times]
            ax.plot(x_values, series.values, color="dodgerblue", linewidth=1.6, alpha=0.92, marker="o", markersize=3)

        month_names = ", ".join(months)
        ax.set_title(f"{season}  ({month_names})")
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 160)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Valid time of day (UTC)")
        ax.set_ylabel(metric.upper())
        ax.xaxis.set_major_locator(mticker.FixedLocator(x_ticks))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_minutes_formatter))
        ax.tick_params(axis="x", rotation=45)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if season_scheme == "meteorological" else f"_{season_scheme}"
    out_path = output_dir / f"seasonal_scores_{metric}_diurnal_cycle{suffix}.png"
    fig.suptitle(f"Seasonal diurnal-cycle average ({season_scheme})", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)

    csv_files = sorted(input_dir.rglob("scores_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No scores_*.csv files found in {input_dir}")

    month_to_files: dict[str, list[Path]] = {}
    for path in csv_files:
        month_to_files.setdefault(_month_from_path(path), []).append(path)

    season_scheme = SEASON_SCHEMES[args.season_scheme]
    season_order = season_scheme["order"]
    month_to_season = season_scheme["month_to_season"]
    season_month_order = season_scheme["season_month_order"]

    month_tables_by_metric: dict[str, dict[str, pd.DataFrame]] = {"mae": {}, "rmse": {}}
    month_labels_by_season: dict[str, list[tuple[str, str]]] = {season: [] for season in season_order}

    for month_key, paths in sorted(month_to_files.items()):
        month_num = int(month_key[4:6])
        season = _season_for_month(month_num, month_to_season)
        month_labels_by_season[season].append((month_key, _month_label(month_key)))
        print(f"Processing {month_key}: found {len(paths)} daily files -> {season}")

        for metric in ["mae", "rmse"]:
            metric_col = _metric_columns(metric)[0]
            month_tables_by_metric[metric][month_key] = _prepare_monthly_diurnal_cycle(paths, metric_col)

    metrics = [args.metric] if args.metric != "both" else ["mae", "rmse"]
    for metric in metrics:
        metric_col = _metric_columns(metric)[0]
        season_tables: dict[str, tuple[pd.DataFrame, list[str]]] = {}

        for season in season_order:
            month_keys = [
                month_key
                for month_key in sorted(month_to_files)
                if _season_for_month(int(month_key[4:6]), month_to_season) == season
            ]
            monthly_frames = [month_tables_by_metric[metric][month_key] for month_key in month_keys if month_key in month_tables_by_metric[metric]]
            if monthly_frames:
                season_table = _combine_monthly_curves(monthly_frames, metric_col)
            else:
                season_table = pd.DataFrame(columns=["init_time", "lead_time_minutes", metric_col])
            ordered_months = [
                label
                for month_key, label in sorted(
                    month_labels_by_season[season],
                    key=lambda item: _season_month_sort_key(season, item[0], season_month_order),
                )
            ]
            season_tables[season] = (season_table, ordered_months)

        out_path = _plot_season_panels(
            season_tables,
            season_order,
            metric,
            output_dir,
            args.max_lead_time_minutes,
            args.season_scheme,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()