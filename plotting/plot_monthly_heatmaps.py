#!/usr/bin/env python3
"""Plot monthly validation summaries and averaged heatmaps from daily CSV outputs.

The validation pipeline writes one CSV per day named like:
    scores_YYYYMMDD.csv

This script groups those daily files by month and writes plots per month from
whatever daily files are available:
- a summary heatmap by day and lead time
- an averaged heatmap by initialization time and lead time
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pvlib

# Add parent directory to path to allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sunflow_scores import (
    _load_daily_csv,
    _metric_columns,
    _month_from_path,
    _heatmap_norm_fixed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot monthly average validation heatmaps from daily CSVs.",
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
        help="Directory where monthly plots will be written.",
    )
    parser.add_argument(
        "--metric",
        choices=["mae", "rmse", "both"],
        default="both",
        help="Metric to plot.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Write the month-by-day summary heatmap.",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Write the averaged init-time/lead-time heatmap.",
    )
    parser.add_argument(
        "--leadtime-average",
        action="store_true",
        help="Write a monthly mean score per lead time line plot.",
    )
    parser.add_argument(
        "--init-average",
        action="store_true",
        help="Write a monthly mean score per initialization hour line plot.",
    )
    parser.add_argument(
        "--diurnal-cycle",
        action="store_true",
        help="Write a monthly diurnal-cycle line plot with one 15-minute timeseries per initialization hour.",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=55.6761,
        help="Latitude used to compute sunrise and sunset times.",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=12.5683,
        help="Longitude used to compute sunrise and sunset times.",
    )
    return parser.parse_args()


def _prepare_day(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.copy()
    daily["init_time"] = daily["initialization_time"].dt.strftime("%H:%M")
    return (
        daily.groupby(["init_time", "lead_time_minutes"], as_index=False)[["mae_by_init", "rmse_by_init"]]
        .mean()
    )


def _solar_event_minutes_utc(date: pd.Timestamp, latitude: float, longitude: float, sunrise: bool) -> float | None:
    day_of_year = date.timetuple().tm_yday
    lng_hour = longitude / 15.0
    t = day_of_year + ((6 - lng_hour) / 24.0 if sunrise else (18 - lng_hour) / 24.0)

    mean_anomaly = 0.9856 * t - 3.289
    true_longitude = mean_anomaly + 1.916 * math.sin(math.radians(mean_anomaly)) + 0.020 * math.sin(math.radians(2 * mean_anomaly)) + 282.634
    true_longitude %= 360.0

    right_ascension = math.degrees(math.atan(0.91764 * math.tan(math.radians(true_longitude))))
    right_ascension %= 360.0
    l_quadrant = (math.floor(true_longitude / 90.0)) * 90.0
    ra_quadrant = (math.floor(right_ascension / 90.0)) * 90.0
    right_ascension = (right_ascension + (l_quadrant - ra_quadrant)) / 15.0

    sin_declination = 0.39782 * math.sin(math.radians(true_longitude))
    cos_declination = math.cos(math.asin(sin_declination))
    cos_hour_angle = (
        math.cos(math.radians(90.833)) - sin_declination * math.sin(math.radians(latitude))
    ) / (cos_declination * math.cos(math.radians(latitude)))

    if cos_hour_angle > 1.0 or cos_hour_angle < -1.0:
        return None

    hour_angle = math.degrees(math.acos(cos_hour_angle))
    if sunrise:
        hour_angle = 360.0 - hour_angle
    hour_angle /= 15.0

    local_mean_time = hour_angle + right_ascension - 0.06571 * t - 6.622
    universal_time = (local_mean_time - lng_hour) % 24.0
    return universal_time * 60.0


def _average_sunrise_sunset(days: list[str], latitude: float, longitude: float) -> tuple[float | None, float | None]:
    sunrise_minutes: list[float] = []
    sunset_minutes: list[float] = []

    for day in days:
        date = pd.Timestamp(day)
        sunrise = _solar_event_minutes_utc(date, latitude, longitude, sunrise=True)
        sunset = _solar_event_minutes_utc(date, latitude, longitude, sunrise=False)
        if sunrise is not None:
            sunrise_minutes.append(sunrise)
        if sunset is not None:
            sunset_minutes.append(sunset)

    sunrise_avg = float(np.mean(sunrise_minutes)) if sunrise_minutes else None
    sunset_avg = float(np.mean(sunset_minutes)) if sunset_minutes else None
    return sunrise_avg, sunset_avg


def _minute_to_axis_position(pivot_index: pd.Index, minute_value: float) -> float:
    axis_minutes = []
    for value in pivot_index:
        time_label = pd.Timestamp(str(value)).strftime("%H:%M")
        hour, minute = map(int, time_label.split(":"))
        axis_minutes.append(hour * 60 + minute)

    axis_positions = np.arange(len(axis_minutes), dtype=float)
    return float(np.interp(minute_value, axis_minutes, axis_positions))


def _format_minutes_utc(minute_value: float) -> str:
    total_minutes = int(round(minute_value)) % (24 * 60)
    hour, minute = divmod(total_minutes, 60)
    return f"{hour:02d}:{minute:02d} UTC"


def _plot_month_summary(month: str, daily_frames: list[tuple[pd.DataFrame, str]], output_dir: Path, metric: str) -> Path:
    metric_col = _metric_columns(metric)[0]
    rows = []
    for frame, day in daily_frames:
        daily = frame.groupby("lead_time_minutes", as_index=False)[[metric_col]].mean()
        daily.insert(0, "day", day)
        rows.append(daily)

    summary = pd.concat(rows, ignore_index=True)
    summary["day"] = pd.to_datetime(summary["day"], format="%Y%m%d")

    pivot = summary.pivot(index="day", columns="lead_time_minutes", values=metric_col).sort_index()

    fig, ax = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdYlGn_r",
        #norm=_heatmap_norm_fixed(),
    )
    ax.set_title(f"{month} daily mean {metric.upper()} by lead time")
    ax.set_xlabel("Lead time (minutes)")
    ax.set_ylabel("Day")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(v) for v in pivot.columns], rotation=45, ha="right")
    y_ticks = list(range(0, len(pivot.index), max(1, len(pivot.index) // 12)))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([pivot.index[i].strftime("%Y-%m-%d") for i in y_ticks])
    fig.colorbar(im, ax=ax, shrink=0.85)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"monthly_scores_{month}_{metric}_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_month_average(
    month: str,
    frames: list[tuple[pd.DataFrame, str]],
    output_dir: Path,
    metric: str,
    latitude: float,
    longitude: float,
) -> Path:
    summary = pd.concat([frame for frame, _ in frames], ignore_index=True)
    metric_col = _metric_columns(metric)[0]
    average = summary.groupby(["init_time", "lead_time_minutes"], as_index=False)[[metric_col]].mean()

    pivot = average.pivot(index="init_time", columns="lead_time_minutes", values=metric_col).sort_index()
    day_labels = [day for _, day in frames]
    sunrise_avg, sunset_avg = _average_sunrise_sunset(day_labels, latitude, longitude)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdYlGn_r",
        norm=_heatmap_norm_fixed(),
    )
    title = f"{month} average {metric.upper()} by init time and lead time"
    ax.set_xlabel("Lead time (minutes)")
    ax.set_ylabel("Initialization time")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(v) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)

    if sunrise_avg is not None:
        sunrise_y = _minute_to_axis_position(pivot.index, sunrise_avg)
        ax.axhline(sunrise_y, color="white", linestyle="--", linewidth=1.5, alpha=0.95)
        ax.text(
            0.01,
            sunrise_y,
            f"Sunrise {_format_minutes_utc(sunrise_avg)}",
            color="white",
            fontsize=9,
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=2),
        )

    if sunset_avg is not None:
        sunset_y = _minute_to_axis_position(pivot.index, sunset_avg)
        ax.axhline(sunset_y, color="white", linestyle="--", linewidth=1.5, alpha=0.95)
        ax.text(
            0.99,
            sunset_y,
            f"Sunset {_format_minutes_utc(sunset_avg)}",
            color="white",
            fontsize=9,
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="right",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=2),
        )

    fig.colorbar(im, ax=ax, shrink=0.85)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"monthly_scores_{month}_{metric}_average_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_month_leadtime_average(month: str, frames: list[tuple[pd.DataFrame, str]], output_dir: Path, metric: str) -> Path:
    metric_col = _metric_columns(metric)[0]
    rows = []
    for frame, day in frames:
        daily = frame.groupby("lead_time_minutes", as_index=False)[[metric_col]].mean()
        daily.insert(0, "day", day)
        rows.append(daily)

    summary = pd.concat(rows, ignore_index=True)
    average = summary.groupby("lead_time_minutes", as_index=False)[[metric_col]].mean()

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    ax.plot(average["lead_time_minutes"], average[metric_col], marker="o", label=metric.upper())
    ax.set_title(f"{month} mean {metric.upper()} per lead time")
    ax.set_xlabel("Lead time (minutes)")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"monthly_scores_{month}_{metric}_leadtime_average.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_month_init_average(month: str, frames: list[tuple[pd.DataFrame, str]], output_dir: Path, metric: str) -> Path:
    metric_col = _metric_columns(metric)[0]
    rows = []
    for frame, day in frames:
        daily = frame.copy()
        daily["init_hour"] = daily["init_time"].str.slice(0, 2).astype(int)
        daily = daily.groupby("init_hour", as_index=False)[[metric_col]].mean()
        daily.insert(0, "day", day)
        rows.append(daily)

    summary = pd.concat(rows, ignore_index=True)
    average = summary.groupby("init_hour", as_index=False)[[metric_col]].mean().sort_values("init_hour")

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    ax.plot(average["init_hour"], average[metric_col], marker="o", label=metric.upper())
    ax.set_title(f"{month} mean {metric.upper()} per initialization hour")
    ax.set_xlabel("Initialization hour (UTC)")
    ax.set_ylabel("Score")
    ax.set_xticks(average["init_hour"])
    ax.set_xticklabels([f"{hour:02d}:00" for hour in average["init_hour"]], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"monthly_scores_{month}_{metric}_init_average.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_month_diurnal_cycle(month: str, paths: list[Path], output_dir: Path, metric: str) -> Path:
    metric_col = _metric_columns(metric)[0]
    rows = []
    for path in paths:
        df = _load_daily_csv(path)
        df = _filter_nowcasts_with_nan_at_leadtime_0(df)
        daily = df[["initialization_time", "lead_time_minutes", metric_col]].copy()
        daily["init_hour"] = daily["initialization_time"].dt.floor("h").dt.strftime("%H:%M")
        rows.append(daily)

    summary = pd.concat(rows, ignore_index=True)
    grouped = (
        summary.groupby(["init_hour", "lead_time_minutes"], as_index=False)[[metric_col]]
        .mean()
        .sort_values(["init_hour", "lead_time_minutes"])
    )

    lead_times = [lead_time for lead_time in sorted(grouped["lead_time_minutes"].unique()) if lead_time <= 90]
    grouped = grouped[grouped["lead_time_minutes"] <= 90]

    fig, ax = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)
    base_date = pd.Timestamp("2000-01-01")
    init_hours = sorted(grouped["init_hour"].unique())
    for init_hour in init_hours:
        init_hour_dt = pd.to_datetime(init_hour, format="%H:%M")
        start_time = base_date + pd.Timedelta(hours=init_hour_dt.hour, minutes=init_hour_dt.minute)
        init_data = grouped[grouped["init_hour"] == init_hour].set_index("lead_time_minutes")[metric_col].reindex(lead_times)
        x_values = [start_time + pd.Timedelta(minutes=int(lead_time)) for lead_time in lead_times]
        ax.plot(x_values, init_data.values, marker="o", linewidth=1.8, label=init_hour)

    ax.set_title(f"{month} mean {metric.upper()} by initialization hour")
    ax.set_xlabel("Time of day (UTC)")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 160)
    ax.set_xlim(base_date, base_date + pd.Timedelta(hours=24))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Init hour", ncols=2, fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"monthly_scores_{month}_{metric}_diurnal_cycle.png"
    fig.savefig(out_path, dpi=150)
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

    for month, paths in sorted(month_to_files.items()):
        print(f"Processing {month}: found {len(paths)} daily files")

        daily_frames = []
        average_frames = []
        for path in paths:
            df = _load_daily_csv(path)
            df = _filter_nowcasts_with_nan_at_leadtime_0(df)
            day = path.stem.replace("scores_", "")
            daily_frames.append((_prepare_day(df), day))
            average_frames.append((_prepare_day(df), day))

        metrics = [args.metric] if args.metric != "both" else ["mae", "rmse"]
        wrote_any = False
        if args.summary or not args.heatmap:
            for metric in metrics:
                summary_path = _plot_month_summary(month, daily_frames, output_dir, metric)
                print(f"Wrote {summary_path}")
                wrote_any = True
        if args.heatmap or not args.summary:
            for metric in metrics:
                average_path = _plot_month_average(
                    month,
                    average_frames,
                    output_dir,
                    metric,
                    args.latitude,
                    args.longitude,
                )
                print(f"Wrote {average_path}")
                wrote_any = True
        if args.leadtime_average:
            for metric in metrics:
                leadtime_average_path = _plot_month_leadtime_average(month, average_frames, output_dir, metric)
                print(f"Wrote {leadtime_average_path}")
                wrote_any = True
        if args.init_average:
            for metric in metrics:
                init_average_path = _plot_month_init_average(month, average_frames, output_dir, metric)
                print(f"Wrote {init_average_path}")
                wrote_any = True
        if args.diurnal_cycle:
            for metric in metrics:
                diurnal_cycle_path = _plot_month_diurnal_cycle(month, paths, output_dir, metric)
                print(f"Wrote {diurnal_cycle_path}")
                wrote_any = True
        if not wrote_any:
            print(f"SKIP {month}: no plot type selected")


if __name__ == "__main__":
    main()