"""
Satellite Nowcast Score Framework
=======================================
Classes:
    SatelliteNowcastLoader     – loads nowcast NetCDF files into xarray
    SatelliteObservationLoader – loads satellite observation NetCDF files
    GroundObservationLoader    – loads ground station (pyranometer) CSV/NetCDF files
    ScoreCalculator            – aligns satellite nowcast/obs and computes MAE / RMSE
    GroundScoreCalculator      – point-based validation against station observations

File naming conventions:
    Nowcasts:     SolarNowcast_YYYYMMDDHHMM.nc
    Observations: NetCDF4_sds_YYYY-MM-DDTHH_MM_SSZ.nc

Alignment modes (ScoreCalculator.align_data):
    "fast"    – exact-time same-grid, no reindex, no loop, fast
    "general" – chunked loop with nearest-time reindex, tolerates grid mismatch
    "auto"    – tries fast first, falls back to general on failure
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# HDF5 file locking must be disabled BEFORE the HDF5 C library is loaded
# (i.e. before importing xarray/h5netcdf/h5py below); setting it afterwards
# is a no-op. Locking is unreliable on networked filesystems (NFS/Lustre)
# and causes spurious "OSError: [Errno 121] ... unable to lock file"
# failures when reading .nc files from the mounted data drive.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import dask
import numpy as np
import pandas as pd
import scores.continuous
import xarray as xr

# errno values that indicate a transient/remote filesystem hiccup rather than
# a real problem with the file itself; worth a short retry before giving up.
_TRANSIENT_IO_ERRNOS = {121, 5, 11, 116}  # Remote I/O error, I/O error, EAGAIN, stale handle

# Substrings (lower-case) seen in exception messages that indicate a corrupted
# / flaky h5netcdf-h5py file handle rather than a genuine data problem. These
# come from real-world observations of network-filesystem HDF5 corruption
# (broken file handles, dangling dataset identifiers, low-level HDF5 library
# errors surfaced through h5py). Keep this list broad but specific enough to
# avoid masking real bugs.
_H5_CORRUPTION_KEYWORDS = (
    "nonetype",
    "unable to synchronously",
    "unable to lock",
    "invalid identifier",
    "invalid dataset identifier",
    "not of specified type",
    "_root",
    "_h5file",
    "h5ds",
    "unspecified error",
    "num_scales",
)


# =============================================================================
# HELPERS
# =============================================================================


def _is_h5_corruption_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like transient h5netcdf/h5py file corruption."""
    exc_str = str(exc).lower()
    return any(keyword in exc_str for keyword in _H5_CORRUPTION_KEYWORDS)


def _open_with_retry(open_fn, *args, retries: int = 3, delay: float = 2.0, **kwargs):
    """
    Call *open_fn* (e.g. xr.open_mfdataset/xr.open_dataset), retrying a few
    times if it fails with transient remote-I/O / file-locking errors or h5netcdf
    corruption from flaky network-filesystem access (NFS/Lustre).

    This guards against spurious "OSError: [Errno 121] Remote I/O error" and
    h5netcdf/h5py errors like "'NoneType' object has no attribute '_root'"
    (broken file handles on network mounts).
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return open_fn(*args, **kwargs)
        except (OSError, AttributeError, RuntimeError, KeyError, ValueError, TypeError) as exc:
            last_exc = exc
            is_transient_os = isinstance(exc, OSError) and getattr(exc, "errno", None) in _TRANSIENT_IO_ERRNOS
            is_h5_corruption = isinstance(exc, (AttributeError, RuntimeError, KeyError, ValueError, TypeError)) and \
                _is_h5_corruption_error(exc)
            if not (is_transient_os or is_h5_corruption) or attempt == retries:
                raise
            print(
                f"  WARNING: transient I/O / h5netcdf error "
                f"(attempt {attempt}/{retries}): {exc.__class__.__name__}: {exc}. Retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
    raise last_exc  # pragma: no cover - unreachable, satisfies type checkers

def _compute_scores_per_init(mae_lazy: xr.DataArray, rmse_lazy: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute scores by initialization_time, filling corrupted (init, lead_time) steps with NaN
    on transient I/O errors.

    Each initialization_time is first attempted as a single batched compute (fast path). If
    that fails due to h5netcdf corruption, the individual lead_time steps of that init are
    retried one by one so that only the specific corrupt lead_time(s) — i.e. the specific
    valid_time/cut-out-domain snapshot that is actually bad — end up as NaN, instead of NaN-ing
    the entire init (all lead times). All computation is eager (not lazy) to avoid re-opening
    h5netcdf files during later operations.
    """
    inits = mae_lazy.coords["initialization_time"].values
    lead_times = mae_lazy.coords["lead_time"].values
    n_lead = len(lead_times)

    mae_parts = []
    rmse_parts = []
    failed_inits = []          # inits where every lead_time was corrupt (fully NaN)
    partial_inits = []         # inits where only some lead_time(s) were corrupt

    for init in inits:
        mae_sel = mae_lazy.sel(initialization_time=init)
        rmse_sel = rmse_lazy.sel(initialization_time=init)
        try:
            mae_val, rmse_val = dask.compute(mae_sel, rmse_sel)
            # Ensure we have numpy arrays (eager, not lazy)
            mae_parts.append(np.asarray(mae_val.values, dtype="float32"))
            rmse_parts.append(np.asarray(rmse_val.values, dtype="float32"))
            continue
        except Exception as exc:
            if not _is_h5_corruption_error(exc):
                raise
            init_exc = exc

        # Whole-init compute failed with a corruption-like error. Retry lead_time by
        # lead_time so that a single bad valid_time doesn't wipe out the whole init.
        mae_lead_vals = np.full(n_lead, np.nan, dtype="float32")
        rmse_lead_vals = np.full(n_lead, np.nan, dtype="float32")
        n_bad_leads = 0
        last_lead_exc: Exception = init_exc
        for lead_idx, lead in enumerate(lead_times):
            try:
                mae_lt, rmse_lt = dask.compute(
                    mae_sel.sel(lead_time=lead), rmse_sel.sel(lead_time=lead)
                )
                mae_lead_vals[lead_idx] = float(np.asarray(mae_lt.values))
                rmse_lead_vals[lead_idx] = float(np.asarray(rmse_lt.values))
            except Exception as lead_exc:
                if not _is_h5_corruption_error(lead_exc):
                    raise
                n_bad_leads += 1
                last_lead_exc = lead_exc

        mae_parts.append(mae_lead_vals)
        rmse_parts.append(rmse_lead_vals)

        if n_bad_leads == n_lead:
            failed_inits.append(init)
            print(f"    FILL init {init} with NaN: h5netcdf corruption ({last_lead_exc.__class__.__name__})")
        else:
            partial_inits.append(init)
            print(
                f"    PARTIAL FILL init {init}: {n_bad_leads}/{n_lead} lead_time step(s) set to "
                f"NaN due to h5netcdf corruption ({last_lead_exc.__class__.__name__}), "
                f"{n_lead - n_bad_leads} lead_time step(s) computed successfully"
            )

    if not mae_parts:
        raise ValueError("All initialization_time steps failed; day is unusable")

    # Stack eager numpy arrays back into xarray DataArrays with proper coords
    mae_data = np.stack(mae_parts, axis=0)
    rmse_data = np.stack(rmse_parts, axis=0)

    # Compute valid_time from initialization_time + lead_time
    # valid_time is a 2D coordinate (init_time, lead_time)
    valid_times = np.empty((len(inits), len(lead_times)), dtype="datetime64[ns]")
    for i, init_t in enumerate(inits):
        for j, lead_t in enumerate(lead_times):
            valid_times[i, j] = np.datetime64(init_t) + lead_t

    mae_result = xr.DataArray(
        mae_data,
        coords={
            "initialization_time": inits,
            "lead_time": lead_times,
            "valid_time": (("initialization_time", "lead_time"), valid_times),
        },
        dims=["initialization_time", "lead_time"],
        name="mae_by_init"
    )
    rmse_result = xr.DataArray(
        rmse_data,
        coords={
            "initialization_time": inits,
            "lead_time": lead_times,
            "valid_time": (("initialization_time", "lead_time"), valid_times),
        },
        dims=["initialization_time", "lead_time"],
        name="rmse_by_init"
    )

    n_clean_inits = len(mae_parts) - len(failed_inits) - len(partial_inits)
    if failed_inits or partial_inits:
        print(
            f"    Filled {len(failed_inits)} fully corrupt init(s) with NaN, "
            f"{len(partial_inits)} init(s) partially NaN-filled (specific lead_time steps only), "
            f"{n_clean_inits} inits computed cleanly"
        )

    return mae_result, rmse_result

def _parse_nowcast_timestamp(path: Path) -> pd.Timestamp | None:
    """Return the initialization time encoded in a nowcast filename, or None."""
    name = path.stem
    if not name.startswith("SolarNowcast_"):
        return None
    raw = name.replace("SolarNowcast_", "")
    try:
        return pd.to_datetime(raw, format="%Y%m%d%H%M")
    except ValueError:
        return None


def _parse_observation_timestamp(path: Path) -> pd.Timestamp | None:
    """Return the timestamp encoded in a satellite observation filename, or None."""
    name = path.stem
    if not name.startswith("NetCDF4_sds_"):
        return None
    raw = name.replace("NetCDF4_sds_", "")
    try:
        return pd.to_datetime(raw, format="%Y-%m-%dT%H_%M_%SZ")
    except ValueError:
        return None


def _filter_files_by_time(
    files: list[Path],
    parser,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[Path]:
    """Return sorted subset of *files* whose parsed timestamps fall within [start, end]."""
    selected: list[Path] = []
    for path in files:
        ts = parser(path)
        if ts is None:
            continue
        if start_date <= ts <= end_date:
            selected.append(path)
    return sorted(selected)


def _subset_to_bbox(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float] | None,
) -> xr.Dataset:
    """
    Restrict a lat/lon-gridded Dataset to a geographic bounding box.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 1-D 'lat' and 'lon' coordinates.
    bbox : tuple(lon_min, lat_min, lon_max, lat_max) or None
        Bounding box in degrees. If None the dataset is returned unchanged.
        Example (Denmark): (7.5, 54.5, 13.0, 58.0).

    Notes
    -----
    Handles both ascending and descending latitude/longitude axes so that
    xarray's ``slice`` selection never returns an empty result due to axis
    ordering.
    """
    if bbox is None:
        return ds
    if "lat" not in ds.coords or "lon" not in ds.coords:
        print("  WARNING: bbox requested but dataset has no lat/lon coords; skipping subset.")
        return ds

    lon_min, lat_min, lon_max, lat_max = bbox

    lat_vals = np.asarray(ds["lat"].values)
    lon_vals = np.asarray(ds["lon"].values)
    lat_ascending = lat_vals.size < 2 or lat_vals[0] <= lat_vals[-1]
    lon_ascending = lon_vals.size < 2 or lon_vals[0] <= lon_vals[-1]

    lat_slice = slice(lat_min, lat_max) if lat_ascending else slice(lat_max, lat_min)
    lon_slice = slice(lon_min, lon_max) if lon_ascending else slice(lon_max, lon_min)

    subset = ds.sel(lat=lat_slice, lon=lon_slice)

    if subset.sizes.get("lat", 0) == 0 or subset.sizes.get("lon", 0) == 0:
        raise ValueError(
            f"Bounding box {bbox} selected no grid cells. "
            f"Data covers lat [{lat_vals.min()}, {lat_vals.max()}], "
            f"lon [{lon_vals.min()}, {lon_vals.max()}]."
        )

    print(
        f"  Restricted to bbox lon=[{lon_min}, {lon_max}] lat=[{lat_min}, {lat_max}] "
        f"-> {subset.sizes.get('lat')} x {subset.sizes.get('lon')} grid cells."
    )
    return subset


# =============================================================================
# 1. NOWCAST LOADER
# =============================================================================

class SatelliteNowcastLoader:
    """Loads solar irradiance nowcast files into a single xarray Dataset."""

    def __init__(self, data_dir: str, bbox: tuple[float, float, float, float] | None = None):
        self.data_dir = Path(data_dir)
        self.bbox = bbox

    def _preprocess_nowcast(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Prepare a single nowcast file for concatenation.

        Each file's 'time' coordinate holds valid datetimes (e.g. 10:30, 10:45 …
        for a 10:15 init run). This function:
          - computes lead_time = valid_time − initialization_time
          - replaces the 'time' dim with 'lead_time' so all files share the same axis
          - adds 'initialization_time' as a new scalar dim for stacking
          - stores valid_time as a 2D auxiliary coordinate for later reference
        """
        source = Path(ds.encoding["source"])
        initialization_time = _parse_nowcast_timestamp(source)
        if initialization_time is None:
            # Fallback to old inline parse for any non-standard filename
            fname = source.name
            initialization_time = pd.to_datetime(
                fname.split("_")[1].replace(".nc", ""), format="%Y%m%d%H%M"
            )

        valid_times = ds["time"].values
        lead_times  = valid_times - initialization_time.to_datetime64()

        ds = ds.assign_coords(time=("time", lead_times))
        ds = ds.rename({"time": "lead_time"})
        ds = ds.expand_dims(initialization_time=[initialization_time.to_datetime64()])
        ds = ds.assign_coords(
            valid_time=(["initialization_time", "lead_time"], [valid_times])
        )
        return ds

    def load_data(self, start_date, end_date) -> xr.Dataset:
        """
        Load all nowcast files with initialization_time in [start_date, end_date].

        Uses a glob + timestamp-filter approach (faster than date_range iteration
        over large directories). Falls back with an informative error if no files
        are found.

        Returns a lazy Dataset with dims (initialization_time, lead_time, lat, lon).
        """
        start_date = pd.Timestamp(start_date)
        end_date   = pd.Timestamp(end_date)

        all_files = list(self.data_dir.rglob("SolarNowcast_*.nc"))
        files_to_open = _filter_files_by_time(
            all_files, _parse_nowcast_timestamp, start_date, end_date
        )

        if not files_to_open:
            if not all_files:
                raise ValueError(f"No nowcast files found in {self.data_dir}")
            timestamps = [
                ts for ts in (_parse_nowcast_timestamp(p) for p in all_files)
                if ts is not None
            ]
            if not timestamps:
                raise ValueError(f"No parseable nowcast files found in {self.data_dir}")
            earliest, latest = min(timestamps), max(timestamps)
            print(f"  WARNING: no nowcast files in requested interval.")
            print(f"  Available data ranges from {earliest} to {latest}.")
            # Clip to available range and try again
            adjusted_start = max(start_date, earliest)
            adjusted_end   = min(end_date, latest)
            if adjusted_start > adjusted_end:
                raise ValueError(
                    f"No nowcast files found between {start_date} and {end_date}. "
                    f"Available range: {earliest} to {latest}."
                )
            files_to_open = _filter_files_by_time(
                all_files, _parse_nowcast_timestamp, adjusted_start, adjusted_end
            )
            if files_to_open:
                print(
                    f"  Using adjusted nowcast range {adjusted_start} to {adjusted_end} "
                    f"({len(files_to_open)} files)."
                )

        if not files_to_open:
            raise ValueError(f"No nowcast files found between {start_date} and {end_date}.")

        print(f"  Found {len(files_to_open)} nowcast files from {start_date} to {end_date}.")

        ds = _open_with_retry(
            xr.open_mfdataset,
            [str(p) for p in files_to_open],
            combine="nested",
            concat_dim="initialization_time",
            preprocess=self._preprocess_nowcast,
            engine="h5netcdf",
            parallel=True,
            lock=False,
            chunks={
                "initialization_time": 8,
                "lead_time": 32,
                "lat": 128,
                "lon": 128,
            },
        )
        return _subset_to_bbox(ds, self.bbox)


# =============================================================================
# 2. SATELLITE OBSERVATION LOADER
# =============================================================================

class SatelliteObservationLoader:
    """Loads satellite observation (ground-truth) files into a single xarray Dataset."""

    def __init__(self, data_dir: str, bbox: tuple[float, float, float, float] | None = None):
        self.data_dir = Path(data_dir)
        self.bbox = bbox

    def _preprocess_observation(self, ds: xr.Dataset) -> xr.Dataset:
        """Drop the 'crs' scalar variable and normalise spatial dim names."""
        if "crs" in ds:
            ds = ds.drop_vars("crs")
        if "y" in ds.dims and "x" in ds.dims:
            ds = ds.rename({"y": "lat", "x": "lon"})
        return ds

    def load_data(self, start_date, end_date) -> xr.Dataset:
        """
        Load observation files covering [start_date, end_date].

        Returns a Dataset with dim 'time' at 15-minute resolution.
        """
        start_date = pd.Timestamp(start_date)
        end_date   = pd.Timestamp(end_date)

        all_files = list(self.data_dir.rglob("NetCDF4_sds_*.nc"))
        files_to_open = _filter_files_by_time(
            all_files,
            _parse_observation_timestamp,
            start_date.floor("15min"),
            end_date.ceil("15min"),
        )

        if not files_to_open:
            if not all_files:
                raise ValueError(f"No observation files found in {self.data_dir}")
            timestamps = [
                ts for ts in (_parse_observation_timestamp(p) for p in all_files)
                if ts is not None
            ]
            if not timestamps:
                raise ValueError(f"No parseable observation files found in {self.data_dir}")
            raise ValueError(
                f"No observation files found in requested interval "
                f"{start_date} to {end_date}. Available range: "
                f"{min(timestamps)} to {max(timestamps)}."
            )

        print(f"  Loading {len(files_to_open)} observation files from {start_date.date()} to {end_date.date()}")

        ds = _open_with_retry(
            xr.open_mfdataset,
            [str(p) for p in files_to_open],
            combine="by_coords",
            preprocess=self._preprocess_observation,
            engine="h5netcdf",
            parallel=True,
            lock=False,
            chunks={"time": 256, "lat": 128, "lon": 128},
        )
        ds = ds.sel(time=slice(start_date, end_date)).sortby("time")
        return _subset_to_bbox(ds, self.bbox)


# =============================================================================
# 3. GROUND OBSERVATION LOADER
# =============================================================================

class GroundObservationLoader:
    """
    Loads point-based ground station observations (pyranometers / AWS) and
    returns them as an xarray Dataset with dims (time, station_id).

    Expected input formats:
      - CSV   : columns [time, station_id, ghi, cs_ghi, lat, lon]
                'time' must be parseable by pandas; one row per (time, station).
      - NetCDF: dims (time, station_id) with variables ghi, cs_ghi and
                scalar coordinates lat/lon per station.

    The loader auto-detects the format from the file extension.

    Parameters
    ----------
    data_path : str
        Path to a single CSV/NetCDF file, or a directory of CSV files.
    ghi_var : str
        Name of the GHI column/variable (default 'ghi').
    cs_ghi_var : str
        Name of the clear-sky GHI column/variable (default 'cs_ghi').
    station_col : str
        Name of the station identifier column (default 'station_id').
    lat_col : str
        Name of the latitude column (default 'lat').
    lon_col : str
        Name of the longitude column (default 'lon').
    """

    def __init__(
        self,
        data_path: str,
        ghi_var: str = "ghi",
        cs_ghi_var: str = "cs_ghi",
        station_col: str = "station_id",
        lat_col: str = "lat",
        lon_col: str = "lon",
    ):
        self.data_path  = Path(data_path)
        self.ghi_var    = ghi_var
        self.cs_ghi_var = cs_ghi_var
        self.station_col = station_col
        self.lat_col    = lat_col
        self.lon_col    = lon_col

    def _load_csv(self, path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> xr.Dataset:
        """Load one or more CSV files and pivot to (time, station_id) Dataset."""
        if path.is_dir():
            frames = [pd.read_csv(f, parse_dates=["time"]) for f in sorted(path.glob("*.csv"))]
            if not frames:
                raise ValueError(f"No CSV files found in {path}")
            df = pd.concat(frames, ignore_index=True)
        else:
            df = pd.read_csv(path, parse_dates=["time"])

        df = df[(df["time"] >= start_date) & (df["time"] <= end_date)]
        if df.empty:
            raise ValueError(f"No ground observations in [{start_date}, {end_date}].")

        # Extract per-station lat/lon metadata before pivoting
        station_meta = (
            df.groupby(self.station_col)[[self.lat_col, self.lon_col]]
            .first()
        )

        # Pivot GHI
        ghi_pivot = (
            df.pivot_table(index="time", columns=self.station_col, values=self.ghi_var)
            .sort_index()
        )
        # Pivot clear-sky GHI
        cs_pivot = (
            df.pivot_table(index="time", columns=self.station_col, values=self.cs_ghi_var)
            .sort_index()
        )

        stations = ghi_pivot.columns.tolist()
        times    = pd.DatetimeIndex(ghi_pivot.index)

        ds = xr.Dataset(
            {
                self.ghi_var:    xr.DataArray(ghi_pivot.values,  dims=("time", "station_id")),
                self.cs_ghi_var: xr.DataArray(cs_pivot.values,   dims=("time", "station_id")),
                "lat":           xr.DataArray(station_meta.loc[stations, self.lat_col].values, dims="station_id"),
                "lon":           xr.DataArray(station_meta.loc[stations, self.lon_col].values, dims="station_id"),
            },
            coords={
                "time":       times,
                "station_id": stations,
            },
        )
        print(f"  Loaded {len(stations)} stations, {len(times)} time steps from {path}")
        return ds

    def _load_netcdf(self, path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> xr.Dataset:
        """Load a NetCDF ground-observation file."""
        ds = _open_with_retry(xr.open_dataset, str(path), engine="h5netcdf", lock=False)
        ds = ds.sel(time=slice(start_date, end_date)).sortby("time")
        if ds.sizes["time"] == 0:
            raise ValueError(f"No ground observations in [{start_date}, {end_date}] in {path}.")
        n_stations = ds.sizes.get("station_id", "?")
        print(f"  Loaded {n_stations} stations from {path}")
        return ds

    def load_data(self, start_date, end_date) -> xr.Dataset:
        """
        Load ground observations covering [start_date, end_date].

        Returns an xr.Dataset with dims (time, station_id) and variables
        ghi_var, cs_ghi_var, plus coordinates lat/lon per station.
        """
        start_date = pd.Timestamp(start_date)
        end_date   = pd.Timestamp(end_date)

        path = self.data_path
        if path.is_dir() or path.suffix.lower() == ".csv":
            return self._load_csv(path, start_date, end_date)
        elif path.suffix.lower() in (".nc", ".nc4", ".netcdf"):
            return self._load_netcdf(path, start_date, end_date)
        else:
            # Try CSV first, then NetCDF
            try:
                return self._load_csv(path, start_date, end_date)
            except Exception:
                return self._load_netcdf(path, start_date, end_date)


# =============================================================================
# 4. SATELLITE SCORE CALCULATOR
# =============================================================================

class ScoreCalculator:
    """
    Aligns nowcast and satellite-observation datasets and computes validation scores.

    Parameters
    ----------
    nowcast_data, observation_data : xr.Dataset
    nowcast_ghi_var, obs_ghi_var, obs_cs_ghi_var : str
    """

    def __init__(
        self,
        nowcast_data: xr.Dataset,
        observation_data: xr.Dataset,
        nowcast_ghi_var: str,
        obs_ghi_var: str,
        obs_cs_ghi_var: str,
    ):
        self.nowcast_data     = nowcast_data
        self.observation_data = observation_data
        self.nowcast_ghi_var  = nowcast_ghi_var
        self.obs_ghi_var      = obs_ghi_var
        self.obs_cs_ghi_var   = obs_cs_ghi_var
        self.aligned_data: xr.Dataset | None = None
        self.kt_data: xr.Dataset | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_nowcast_var(self) -> xr.DataArray:
        nwc = self.nowcast_data[self.nowcast_ghi_var]
        if "ensemble" in nwc.dims:
            nwc = nwc.isel(ensemble=0, drop=True)
        return nwc

    def _assert_same_grid(self, nowcast: xr.DataArray, obs: xr.Dataset) -> None:
        """Raise ValueError if spatial grids differ (used by fast path)."""
        for dim in ("lat", "lon"):
            if dim not in nowcast.dims or dim not in obs.dims:
                raise ValueError(
                    f"Expected '{dim}' dimension in both nowcast and observation data"
                )
            if nowcast.sizes[dim] != obs.sizes[dim]:
                raise ValueError(
                    f"Grid mismatch on '{dim}': nowcast={nowcast.sizes[dim]}, "
                    f"obs={obs.sizes[dim]}"
                )
            if not np.array_equal(np.asarray(nowcast[dim].values), np.asarray(obs[dim].values)):
                raise ValueError(
                    f"Coordinate mismatch on '{dim}'; fast same-grid mode requires exact match"
                )

    # ------------------------------------------------------------------
    # Fast alignment path (exact-time, same-grid)
    # ------------------------------------------------------------------

    def _align_fast(self) -> xr.Dataset:
        """
        Single-pass exact-time alignment assuming identical spatial grids.

        Raises ValueError if grids don't match or observation timestamps are missing.
        """
        nwc = self._get_nowcast_var()
        obs = self.observation_data

        self._assert_same_grid(nwc, obs)

        valid_time = self.nowcast_data["valid_time"]
        flat_valid_times = np.asarray(valid_time.values).reshape(-1)

        obs_time_values = np.asarray(obs["time"].values)
        missing_times = np.setdiff1d(
            np.unique(flat_valid_times), np.unique(obs_time_values)
        )
        if missing_times.size > 0:
            raise ValueError(
                f"Exact-time alignment failed: {missing_times.size} nowcast valid times "
                f"have no matching observation. First missing: {missing_times[0]}"
            )

        selected = obs.sel(time=xr.DataArray(flat_valid_times, dims=("sample",)))

        n_init = nwc.sizes["initialization_time"]
        n_lead = nwc.sizes["lead_time"]
        n_lat  = nwc.sizes["lat"]
        n_lon  = nwc.sizes["lon"]

        obs_data = selected[self.obs_ghi_var].data.reshape((n_init, n_lead, n_lat, n_lon))
        cs_data  = selected[self.obs_cs_ghi_var].data.reshape((n_init, n_lead, n_lat, n_lon))

        coords = {
            "initialization_time": nwc.initialization_time,
            "lead_time": nwc.lead_time,
            "lat": nwc.lat,
            "lon": nwc.lon,
        }
        dims = ("initialization_time", "lead_time", "lat", "lon")

        obs_aligned = xr.DataArray(obs_data, dims=dims, coords=coords, name="observation")
        cs_aligned  = xr.DataArray(cs_data,  dims=dims, coords=coords, name="clearsky")

        # Enforce a shared chunk layout to prevent Dask chunk-splitting blowups.
        lead_chunk = int(nwc.sizes["lead_time"])
        chunk_spec = {"initialization_time": 8, "lead_time": lead_chunk, "lat": 128, "lon": 128}
        nwc         = nwc.chunk(chunk_spec)
        obs_aligned = obs_aligned.chunk(chunk_spec)
        cs_aligned  = cs_aligned.chunk(chunk_spec)

        ds = xr.Dataset({"nowcast": nwc, "observation": obs_aligned, "clearsky": cs_aligned})
        ds = ds.assign_coords(valid_time=(("initialization_time", "lead_time"), valid_time.data))
        print("  Data alignment complete (fast same-grid mode).")
        return ds

    # ------------------------------------------------------------------
    # General alignment path (chunked loop, nearest-time reindex)
    # ------------------------------------------------------------------

    def _align_general(self, chunk_size: int = 50) -> xr.Dataset:
        """
        Chunked loop alignment with nearest-time reindex.

        Tolerates minor time offsets and grid differences (y/x renamed to lat/lon).
        """
        nwc = self._get_nowcast_var()
        num_inits = len(self.nowcast_data.initialization_time)
        aligned_chunks = []

        all_valid_times = self.nowcast_data.valid_time.values.ravel()
        valid_min = pd.Timestamp(all_valid_times.min()) - pd.Timedelta(minutes=15)
        valid_max = pd.Timestamp(all_valid_times.max()) + pd.Timedelta(minutes=15)

        obs_source = self.observation_data.sel(time=slice(valid_min, valid_max))
        if "y" in obs_source.dims and "x" in obs_source.dims:
            obs_source = obs_source.rename({"y": "lat", "x": "lon"})

        target_times = np.unique(all_valid_times)
        obs_source = obs_source.reindex(time=target_times, method="nearest")

        for i in range(0, num_inits, chunk_size):
            chunk_slice = slice(i, i + chunk_size)
            nowcast_chunk    = nwc.isel(initialization_time=chunk_slice)
            valid_time_chunk = self.nowcast_data.valid_time.isel(initialization_time=chunk_slice)

            flat_valid_times = valid_time_chunk.values.ravel()
            obs_chunk = obs_source.sel(time=flat_valid_times)
            obs_flat  = obs_chunk[self.obs_ghi_var]
            cs_flat   = obs_chunk[self.obs_cs_ghi_var]

            n_init_chunk = nowcast_chunk.sizes["initialization_time"]
            n_lead = nowcast_chunk.sizes["lead_time"]
            n_lat  = nowcast_chunk.sizes["lat"]
            n_lon  = nowcast_chunk.sizes["lon"]

            coords = {
                "initialization_time": nowcast_chunk.initialization_time,
                "lead_time":           nowcast_chunk.lead_time,
                "lat":                 nowcast_chunk.lat,
                "lon":                 nowcast_chunk.lon,
            }
            dims = ["initialization_time", "lead_time", "lat", "lon"]

            obs_da_chunk = xr.DataArray(
                obs_flat.values.reshape(n_init_chunk, n_lead, n_lat, n_lon),
                dims=dims, coords=coords,
            )
            cs_da_chunk = xr.DataArray(
                cs_flat.values.reshape(n_init_chunk, n_lead, n_lat, n_lon),
                dims=dims, coords=coords,
            )

            aligned_chunks.append(xr.Dataset({
                "nowcast": nowcast_chunk,
                "observation": obs_da_chunk,
                "clearsky": cs_da_chunk,
            }))
            print(f"  Processed chunk {i // chunk_size + 1}/{(num_inits + chunk_size - 1) // chunk_size}")

        aligned = xr.concat(aligned_chunks, dim="initialization_time")
        aligned = aligned.assign_coords(
            valid_time=(("initialization_time", "lead_time"), self.nowcast_data.valid_time.data)
        )
        print("  Data alignment complete (general mode).")
        return aligned

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align_data(self, mode: str = "auto", chunk_size: int = 50) -> xr.Dataset:
        """
        Align nowcast and observation data, storing the result in self.aligned_data.

        Parameters
        ----------
        mode : {"fast", "general", "auto"}
            "fast"    – exact-time same-grid, no reindex, fastest.
            "general" – chunked loop with nearest-time reindex.
            "auto"    – tries fast first, falls back to general on any ValueError.
        chunk_size : int
            Chunk size for the general path (ignored in fast mode).
        """
        if mode == "fast":
            self.aligned_data = self._align_fast()
        elif mode == "general":
            self.aligned_data = self._align_general(chunk_size=chunk_size)
        elif mode == "auto":
            try:
                self.aligned_data = self._align_fast()
            except ValueError as exc:
                print(f"  Fast alignment failed ({exc}); falling back to general mode.")
                self.aligned_data = self._align_general(chunk_size=chunk_size)
        else:
            raise ValueError(f"Unknown alignment mode '{mode}'. Choose 'fast', 'general', or 'auto'.")
        return self.aligned_data

    def calculate_kt(self) -> xr.Dataset:
        """
        Calculate the clear-sky index (kt) for nowcasts and observations.
        Stored in self.kt_data.
        """
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")

        aligned = self.aligned_data.chunk(
            {"initialization_time": 10, "lead_time": 5, "lat": 64, "lon": 64}
        )
        clearsky = aligned["clearsky"].where(aligned["clearsky"] > 1e-6)

        kt_nowcast     = (aligned["nowcast"]      / clearsky).fillna(0)
        kt_observation = (aligned["observation"]  / clearsky).fillna(0)

        self.kt_data = xr.Dataset({"nowcast": kt_nowcast, "observation": kt_observation})
        self.kt_data = self.kt_data.assign_coords(
            valid_time=(("initialization_time", "lead_time"), aligned.valid_time.data)
        )
        print("  Clear-sky index (kt) calculation complete.")
        return self.kt_data

    def calculate_mae(self, data: xr.Dataset, groupby_time_of_day: bool = False) -> xr.DataArray:
        """
        Mean Absolute Error.

        Without grouping: dims (lead_time, lat, lon).
        With groupby_time_of_day=True: dims (hour, lead_time, lat, lon).
        """
        if data is None:
            raise RuntimeError("Provide a dataset to calculate scores on.")
        if isinstance(data, xr.DataArray):
            if groupby_time_of_day:
                raise ValueError("groupby_time_of_day is not supported for DataArray input")
            return data

        if groupby_time_of_day:
            abs_err = np.abs(data["nowcast"] - data["observation"])
            stacked = abs_err.stack(init_lead=("initialization_time", "lead_time"))
            stacked = stacked.assign_coords(
                valid_time=data.valid_time.stack(init_lead=("initialization_time", "lead_time"))
            )
            mae = stacked.groupby(stacked.valid_time.dt.hour).mean(dim="init_lead")
            mae.name = "mae_by_hour"
        else:
            mae = scores.continuous.mae(
                data["nowcast"], data["observation"], reduce_dims="initialization_time"
            )
            mae.name = "mae"
        return mae

    def calculate_rmse(self, data: xr.Dataset, groupby_time_of_day: bool = False) -> xr.DataArray:
        """
        Root Mean Squared Error.

        Without grouping: dims (lead_time, lat, lon).
        With groupby_time_of_day=True: dims (hour, lead_time, lat, lon).
        """
        if data is None:
            raise RuntimeError("Provide a dataset to calculate scores on.")
        if isinstance(data, xr.DataArray):
            if groupby_time_of_day:
                raise ValueError("groupby_time_of_day is not supported for DataArray input")
            return data

        if groupby_time_of_day:
            sq_err = (data["nowcast"] - data["observation"]) ** 2
            rmse = np.sqrt(
                sq_err.groupby(data.valid_time.dt.hour).mean(dim="initialization_time")
            )
            rmse.name = "rmse_by_hour"
        else:
            rmse = scores.continuous.rmse(
                data["nowcast"], data["observation"], reduce_dims="initialization_time"
            )
            rmse.name = "rmse"
        return rmse

    def calculate_mae_by_init(self) -> xr.DataArray:
        """MAE for each (initialization_time, lead_time) averaged over (lat, lon)."""
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        abs_err = np.abs(self.aligned_data["nowcast"] - self.aligned_data["observation"])
        out = abs_err.mean(dim=["lat", "lon"])
        out.name = "mae_by_init"
        return out

    def calculate_rmse_by_init(self) -> xr.DataArray:
        """RMSE for each (initialization_time, lead_time) averaged over (lat, lon)."""
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        sq_err = (self.aligned_data["nowcast"] - self.aligned_data["observation"]) ** 2
        out = np.sqrt(sq_err.mean(dim=["lat", "lon"]))
        out.name = "rmse_by_init"
        return out

    def calculate_mae_kt_by_init(self) -> xr.DataArray:
        """KT MAE for each (initialization_time, lead_time) averaged over (lat, lon)."""
        if self.kt_data is None:
            raise RuntimeError("Call .calculate_kt() first.")
        abs_err = np.abs(self.kt_data["nowcast"] - self.kt_data["observation"])
        out = abs_err.mean(dim=["lat", "lon"])
        out.name = "mae_kt_by_init"
        return out

    def calculate_rmse_kt_by_init(self) -> xr.DataArray:
        """KT RMSE for each (initialization_time, lead_time) averaged over (lat, lon)."""
        if self.kt_data is None:
            raise RuntimeError("Call .calculate_kt() first.")
        sq_err = (self.kt_data["nowcast"] - self.kt_data["observation"]) ** 2
        out = np.sqrt(sq_err.mean(dim=["lat", "lon"]))
        out.name = "rmse_kt_by_init"
        return out


# =============================================================================
# 5. GROUND SCORE CALCULATOR
# =============================================================================

class GroundScoreCalculator:
    """
    Validates nowcasts against point-based ground station observations.

    For each station, the nearest nowcast grid cell is found using Euclidean
    distance on (lat, lon) coordinates. Scores are then aggregated over stations.

    Parameters
    ----------
    nowcast_data : xr.Dataset
        Output of SatelliteNowcastLoader.load_data() — dims (initialization_time, lead_time, lat, lon).
    ground_obs : xr.Dataset
        Output of GroundObservationLoader.load_data() — dims (time, station_id),
        with 'lat' and 'lon' coordinate variables per station.
    nowcast_ghi_var : str
    obs_ghi_var : str
    obs_cs_ghi_var : str
    """

    def __init__(
        self,
        nowcast_data: xr.Dataset,
        ground_obs: xr.Dataset,
        nowcast_ghi_var: str,
        obs_ghi_var: str,
        obs_cs_ghi_var: str = "cs_ghi",
    ):
        self.nowcast_data    = nowcast_data
        self.ground_obs      = ground_obs
        self.nowcast_ghi_var = nowcast_ghi_var
        self.obs_ghi_var     = obs_ghi_var
        self.obs_cs_ghi_var  = obs_cs_ghi_var
        self.aligned_data: xr.Dataset | None = None

    def _nearest_grid_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        For each station, find the (lat_idx, lon_idx) of the nearest nowcast grid cell.

        Returns
        -------
        lat_idxs, lon_idxs : ndarray of int
            One index per station.
        """
        grid_lat = np.asarray(self.nowcast_data["lat"].values)
        grid_lon = np.asarray(self.nowcast_data["lon"].values)
        sta_lat  = np.asarray(self.ground_obs["lat"].values)
        sta_lon  = np.asarray(self.ground_obs["lon"].values)

        # Broadcast to find nearest cell for each station
        lat_idxs = np.argmin(np.abs(grid_lat[:, None] - sta_lat[None, :]), axis=0)
        lon_idxs = np.argmin(np.abs(grid_lon[:, None] - sta_lon[None, :]), axis=0)
        return lat_idxs, lon_idxs

    def align_data(self) -> xr.Dataset:
        """
        Extract nowcast values at each station's nearest grid cell and align with
        observed values by valid_time.

        Result is stored in self.aligned_data with dims
        (initialization_time, lead_time, station_id).
        """
        lat_idxs, lon_idxs = self._nearest_grid_indices()
        stations = self.ground_obs["station_id"].values
        n_stations = len(stations)

        nwc = self.nowcast_data[self.nowcast_ghi_var]
        if "ensemble" in nwc.dims:
            nwc = nwc.isel(ensemble=0, drop=True)

        n_init = nwc.sizes["initialization_time"]
        n_lead = nwc.sizes["lead_time"]

        # Extract nowcast at station grid cells: (init, lead, n_stations)
        nwc_at_stations = np.full((n_init, n_lead, n_stations), np.nan, dtype=np.float32)
        obs_at_stations = np.full((n_init, n_lead, n_stations), np.nan, dtype=np.float32)
        cs_at_stations  = np.full((n_init, n_lead, n_stations), np.nan, dtype=np.float32)

        valid_times = np.asarray(self.nowcast_data["valid_time"].values)  # (n_init, n_lead)

        # Pull nowcast grid values for each station
        for s_idx in range(n_stations):
            li = int(lat_idxs[s_idx])
            loi = int(lon_idxs[s_idx])
            nwc_at_stations[:, :, s_idx] = nwc.isel(lat=li, lon=loi).values

        # Match observations by time
        obs_time = pd.DatetimeIndex(self.ground_obs["time"].values)
        obs_ghi  = self.ground_obs[self.obs_ghi_var].values   # (n_time, n_stations)
        obs_cs   = self.ground_obs[self.obs_cs_ghi_var].values if self.obs_cs_ghi_var in self.ground_obs else None

        obs_time_to_idx = {t: i for i, t in enumerate(obs_time)}

        for init_idx in range(n_init):
            for lead_idx in range(n_lead):
                vt = pd.Timestamp(valid_times[init_idx, lead_idx])
                t_idx = obs_time_to_idx.get(vt)
                if t_idx is not None:
                    obs_at_stations[init_idx, lead_idx, :] = obs_ghi[t_idx, :]
                    if obs_cs is not None:
                        cs_at_stations[init_idx, lead_idx, :] = obs_cs[t_idx, :]

        dims   = ("initialization_time", "lead_time", "station_id")
        coords = {
            "initialization_time": nwc.initialization_time,
            "lead_time":           nwc.lead_time,
            "station_id":          stations,
        }

        self.aligned_data = xr.Dataset({
            "nowcast":     xr.DataArray(nwc_at_stations, dims=dims, coords=coords),
            "observation": xr.DataArray(obs_at_stations, dims=dims, coords=coords),
            "clearsky":    xr.DataArray(cs_at_stations,  dims=dims, coords=coords),
        }).assign_coords(
            valid_time=(("initialization_time", "lead_time"), valid_times)
        )
        print(f"  Ground alignment complete: {n_stations} stations x {n_init} init times.")
        return self.aligned_data

    def calculate_mae_by_init(self) -> xr.DataArray:
        """MAE for each (initialization_time, lead_time) averaged over stations."""
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        abs_err = np.abs(self.aligned_data["nowcast"] - self.aligned_data["observation"])
        out = abs_err.mean(dim="station_id")
        out.name = "mae_by_init"
        return out

    def calculate_rmse_by_init(self) -> xr.DataArray:
        """RMSE for each (initialization_time, lead_time) averaged over stations."""
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        sq_err = (self.aligned_data["nowcast"] - self.aligned_data["observation"]) ** 2
        out = np.sqrt(sq_err.mean(dim="station_id"))
        out.name = "rmse_by_init"
        return out

    def calculate_mae_by_station(self) -> xr.DataArray:
        """MAE for each station averaged over all (initialization_time, lead_time) pairs."""
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        abs_err = np.abs(self.aligned_data["nowcast"] - self.aligned_data["observation"])
        out = abs_err.mean(dim=["initialization_time", "lead_time"])
        out.name = "mae_by_station"
        return out

    def calculate_rmse_by_station(self) -> xr.DataArray:
        """RMSE for each station averaged over all (initialization_time, lead_time) pairs."""
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        sq_err = (self.aligned_data["nowcast"] - self.aligned_data["observation"]) ** 2
        out = np.sqrt(sq_err.mean(dim=["initialization_time", "lead_time"]))
        out.name = "rmse_by_station"
        return out


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Nowcast Validation")
    parser.add_argument("nowcast_dir",  type=str, help="Directory with nowcast files")
    parser.add_argument("obs_dir",      type=str, help="Directory with observation files")
    parser.add_argument("start_date",   type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date",     type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("output_file",  type=str, help="Path to save the output scores NetCDF file")
    parser.add_argument("--align-mode", type=str, default="auto",
                        choices=["fast", "general", "auto"],
                        help="Alignment mode (default: auto)")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Chunk size for general alignment mode")
    parser.add_argument("--nowcast_ghi_var",  type=str, default="probabilistic_advection")
    parser.add_argument("--obs_ghi_var",      type=str, default="sds")
    parser.add_argument("--obs_cs_ghi_var",   type=str, default="sds_cs")
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=None,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Restrict validation to a geographic bounding box, given as "
             "lon_min lat_min lon_max lat_max (e.g. Denmark: 7.5 54.5 13.0 58.0).",
    )

    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox is not None else None

    print("1. Loading data...")
    nowcast_loader = SatelliteNowcastLoader(data_dir=args.nowcast_dir, bbox=bbox)
    nowcast_data   = nowcast_loader.load_data(args.start_date, args.end_date)

    obs_loader = SatelliteObservationLoader(data_dir=args.obs_dir, bbox=bbox)
    obs_data   = obs_loader.load_data(args.start_date, args.end_date)

    print("\n2. Aligning data...")
    calculator = ScoreCalculator(
        nowcast_data, obs_data,
        nowcast_ghi_var=args.nowcast_ghi_var,
        obs_ghi_var=args.obs_ghi_var,
        obs_cs_ghi_var=args.obs_cs_ghi_var,
    )
    aligned_data = calculator.align_data(mode=args.align_mode, chunk_size=args.chunk_size)

    print("\n3. Calculating scores...")
    mae_by_init  = calculator.calculate_mae_by_init()
    rmse_by_init = calculator.calculate_rmse_by_init()

    scores_ds = xr.Dataset({
        "mae_by_init":  mae_by_init,
        "rmse_by_init": rmse_by_init,
    })

    print(f"\n4. Saving scores to {args.output_file}...")
    scores_ds.to_netcdf(args.output_file)

    print("\nValidation complete.")
