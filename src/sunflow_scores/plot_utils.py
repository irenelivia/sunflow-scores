"""Shared plotting utilities for validation score visualization.

Provides common helpers for loading, filtering, and aggregating validation CSVs
across multiple plotting scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import pandas as pd


def _load_daily_csv(path: Path) -> pd.DataFrame:
    """Load a daily validation CSV and validate required columns.

    Args:
        path: Path to a scores_YYYYMMDD.csv file

    Returns:
        DataFrame with columns: initialization_time, valid_time, lead_time_minutes,
        mae_by_init, rmse_by_init (at minimum)

    Raises:
        ValueError: If lead_time_minutes column is missing
    """
    df = pd.read_csv(path, parse_dates=["initialization_time", "valid_time"])
    if "lead_time_minutes" not in df.columns:
        raise ValueError(f"Expected lead_time_minutes column in {path}")
    return df


def _metric_columns(metric: str) -> list[str]:
    """Map metric choice to column names.

    Args:
        metric: One of "mae", "rmse", or "both"

    Returns:
        List of column names: ["mae_by_init"], ["rmse_by_init"], or both
    """
    if metric == "mae":
        return ["mae_by_init"]
    if metric == "rmse":
        return ["rmse_by_init"]
    return ["mae_by_init", "rmse_by_init"]


def _month_from_path(path: Path) -> str:
    """Extract YYYYMM month tag from a scores_YYYYMMDD.csv filename.

    Args:
        path: Path to scores_YYYYMMDD.csv

    Returns:
        YYYYMM string (e.g., "202501")
    """
    return path.stem.replace("scores_", "")[:6]


def _extract_date(path: Path) -> str:
    """Extract YYYYMMDD date tag from a scores_YYYYMMDD.csv filename.

    Args:
        path: Path to scores_YYYYMMDD.csv

    Returns:
        YYYYMMDD string (e.g., "20250115")
    """
    return path.stem.replace("scores_", "")


def _month_label_iso(month: str) -> str:
    """Convert YYYYMM to ISO date label YYYY-MM.

    Args:
        month: YYYYMM string (e.g., "202501")

    Returns:
        ISO format label (e.g., "2025-01")
    """
    return f"{month[:4]}-{month[4:6]}"


def _month_label_short(month: str) -> str:
    """Convert YYYYMM to short month label Mon YYYY.

    Args:
        month: YYYYMM string (e.g., "202501")

    Returns:
        Short format label (e.g., "Jan 2025")
    """
    return pd.Timestamp(month + "01").strftime("%b %Y")


def _heatmap_norm_dynamic(vmax: float = 120) -> mcolors.TwoSlopeNorm:
    """Two-slope colormap norm with dynamic vcenter for heatmaps.

    Adapts the center point of a diverging colormap based on the data's max value,
    capping vcenter at 60 W/m² to avoid washing out variation in low-error plots.

    Args:
        vmax: Maximum value for the colormap (default 120 W/m²)

    Returns:
        TwoSlopeNorm with vmin=0, vcenter=min(60, vmax*0.5), vmax=vmax
    """
    vcenter = min(60, vmax * 0.5)
    return mcolors.TwoSlopeNorm(vmin=0, vcenter=vcenter, vmax=vmax)


def _heatmap_norm_fixed() -> mcolors.TwoSlopeNorm:
    """Two-slope colormap norm with fixed scaling for heatmaps.

    Uses a fixed vmin=0, vcenter=60, vmax=120 regardless of data range.
    Suitable for monthly/aggregated plots with consistent scaling expectations.

    Returns:
        TwoSlopeNorm with vmin=0, vcenter=60, vmax=120
    """
    return mcolors.TwoSlopeNorm(vmin=0, vcenter=60, vmax=120)
