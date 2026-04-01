"""
get_nc_freq.py  –  Infer temporal frequency of NetCDF / xarray data.

Usage as a library:
    from util.get_nc_freq import get_freq
    freq = get_freq(ds)          # xr.Dataset, xr.DataArray, or file path
    if freq.label == 'monthly':
        ...
    print(freq.hours)            # None for calendar-irregular (monthly/annual)

Usage as a CLI tool:
    python get_nc_freq.py /path/to/file.nc
"""

from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class TimeFrequency:
    """
    Describes the temporal frequency of a dataset.

    Attributes
    ----------
    label : str
        Human-readable string: 'annual', 'monthly', 'daily', '6hourly', etc.
    hours : int or None
        Nominal interval length in hours.  None for calendar-irregular
        frequencies (monthly, quarterly, annual) where the number of hours
        per step varies.
    pandas_freq : str or None
        The pandas offset alias returned by pd.infer_freq, if available.
    """
    label: str
    hours: Optional[int]
    pandas_freq: Optional[str]

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return (
            f"TimeFrequency(label={self.label!r}, "
            f"hours={self.hours!r}, pandas_freq={self.pandas_freq!r})"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_freq(source: "str | Path | xr.Dataset | xr.DataArray") -> TimeFrequency:
    """
    Infer the temporal frequency of a dataset or file.

    Parameters
    ----------
    source : str, Path, xr.Dataset, or xr.DataArray
        A file path, an xarray Dataset, or an xarray DataArray.  If a
        DataArray is passed it must be (or contain) the time coordinate.

    Returns
    -------
    TimeFrequency
        Dataclass with .label (str), .hours (int | None), and
        .pandas_freq (str | None).

    Raises
    ------
    ValueError
        If no time coordinate is found or frequency cannot be determined.
    """
    if isinstance(source, (str, Path)):
        with xr.open_dataset(source) as ds:
            return _infer(ds)
    elif isinstance(source, xr.Dataset):
        return _infer(source)
    elif isinstance(source, xr.DataArray):
        return _infer_from_time(source, bounds=None)
    else:
        raise TypeError(f"Expected file path, Dataset, or DataArray; got {type(source)}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer(ds: xr.Dataset) -> TimeFrequency:
    """Infer frequency from a Dataset, using bounds if available."""
    if "time" not in ds:
        raise ValueError("Dataset has no 'time' coordinate.")

    bounds = None
    for candidate in ("time_bnds", "time_bounds"):
        if candidate in ds:
            bounds = ds[candidate]
            break

    return _infer_from_time(ds["time"], bounds=bounds)


def _infer_from_time(
    time: xr.DataArray,
    bounds: Optional[xr.DataArray],
) -> TimeFrequency:
    ntime = len(time)

    # --- Strategy 1: pd.infer_freq (needs >= 3 points, handles all cases) ---
    if ntime >= 3:
        result = _try_pandas_infer(time)
        if result is not None:
            return result

    # --- Strategy 2: single diff (works for >= 2 points) ---
    if ntime >= 2:
        return _from_timedelta(time[1] - time[0], pandas_freq=None)

    # --- Strategy 3: single time step — span from bounds ---
    if bounds is not None:
        span = bounds[0, 1] - bounds[0, 0]
        return _from_timedelta(span, pandas_freq=None)

    raise ValueError(
        "Cannot determine frequency: only one time step and no time bounds."
    )


def _try_pandas_infer(time: xr.DataArray) -> Optional[TimeFrequency]:
    """Attempt frequency inference; return None on failure.

    Tries xr.infer_freq first (handles both numpy datetime64 and cftime),
    then falls back to pd.infer_freq for numpy-only arrays.
    """
    # xr.infer_freq works with cftime and numpy datetime64
    try:
        pfreq = xr.infer_freq(time)
        if pfreq is not None:
            return _parse_pandas_freq(pfreq)
    except Exception:
        pass

    # pd.infer_freq fallback (numpy datetime64 only)
    try:
        idx = pd.DatetimeIndex(time.values)
        pfreq = pd.infer_freq(idx)
        if pfreq is not None:
            return _parse_pandas_freq(pfreq)
    except Exception:
        pass

    return None


def _parse_pandas_freq(pfreq: str) -> TimeFrequency:
    """
    Convert a pandas offset alias (e.g. '6h', 'MS', 'YS') to TimeFrequency.

    pandas >= 2.2 uses lowercase 'h'/'min'; older versions use 'H'/'T'.
    We normalise to uppercase internally.
    """
    # Strip leading/trailing anchor (e.g. 'QS-DEC' -> 'QS')
    base_alias = pfreq.split("-")[0].upper()

    # --- Annual ---
    if any(base_alias.startswith(p) for p in ("YS", "YE", "AS", "AE", "A-", "Y-")):
        return TimeFrequency("annual", None, pfreq)
    if base_alias in ("A", "Y"):
        return TimeFrequency("annual", None, pfreq)

    # --- Quarterly ---
    if base_alias.startswith(("QS", "QE", "Q")):
        return TimeFrequency("quarterly", None, pfreq)

    # --- Monthly ---
    if base_alias.startswith(("MS", "ME", "M")):
        return TimeFrequency("monthly", None, pfreq)

    # --- Parse numeric multiplier + base unit (e.g. '6H', '3T', '30MIN') ---
    i = 0
    while i < len(base_alias) and base_alias[i].isdigit():
        i += 1
    multiplier = int(base_alias[:i]) if i > 0 else 1
    unit = base_alias[i:]  # e.g. 'H', 'D', 'T', 'MIN', 'S'

    if unit in ("H", "HR"):            # hourly
        hours = multiplier
        label = "hourly" if hours == 1 else f"{hours}hourly"
        return TimeFrequency(label, hours, pfreq)

    if unit in ("T", "MIN"):           # minutely (sub-hourly)
        label = f"{multiplier}min"
        return TimeFrequency(label, None, pfreq)

    if unit == "D":                    # daily (or multi-day)
        hours = multiplier * 24
        label = "daily" if multiplier == 1 else f"{multiplier}daily"
        return TimeFrequency(label, hours, pfreq)

    if unit == "S":                    # secondly
        label = f"{multiplier}sec"
        return TimeFrequency(label, None, pfreq)

    # Unknown — return raw alias as label
    return TimeFrequency(pfreq, None, pfreq)


def _from_timedelta(
    dt: xr.DataArray,
    pandas_freq: Optional[str],
) -> TimeFrequency:
    """
    Convert a single xarray timedelta (e.g. from time[1]-time[0]) to
    TimeFrequency.  Used as a fallback when pd.infer_freq is unavailable.
    """
    # Extract hours, handling both numpy.timedelta64 and datetime.timedelta
    # (cftime subtraction produces datetime.timedelta, not timedelta64)
    val = dt.values
    if isinstance(val, np.timedelta64):
        hours = float(val / np.timedelta64(1, "h"))   # numpy division works; float() alone does not
    elif isinstance(val, datetime.timedelta):   # cftime subtraction
        hours = val.total_seconds() / 3600.0
    else:
        raise TypeError(f"Unrecognised timedelta type: {type(val)}")
    days = hours / 24.0

    if 350.0 <= days <= 370.0:
        return TimeFrequency("annual", None, pandas_freq)

    if 27.0 <= days <= 32.0:
        return TimeFrequency("monthly", None, pandas_freq)

    if 85.0 <= days <= 95.0:
        return TimeFrequency("quarterly", None, pandas_freq)

    if abs(hours - 24.0) < 0.5:
        return TimeFrequency("daily", 24, pandas_freq)

    # Sub-daily: find nearest common interval
    for h in (1, 2, 3, 4, 6, 8, 12):
        if abs(hours - h) < 0.1:
            label = "hourly" if h == 1 else f"{h}hourly"
            return TimeFrequency(label, h, pandas_freq)

    # Multi-day but not monthly
    if hours > 24:
        d = round(days)
        return TimeFrequency(f"{d}daily", d * 24, pandas_freq)

    # Arbitrary sub-hourly
    minutes = round(hours * 60)
    return TimeFrequency(f"{minutes}min", None, pandas_freq)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_nc_freq.py <file.nc> [file2.nc ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        try:
            freq = get_freq(path)
            if len(sys.argv) > 2:
                print(f"{path}: {freq}")
            else:
                print(freq)
        except Exception as exc:
            print(f"ERROR ({path}): {exc}", file=sys.stderr)
            sys.exit(1)
