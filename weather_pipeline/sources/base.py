"""Base DataSource abstraction (Stage 3).

Contract: any data source implements

    fetch(area, start, end) -> xarray.Dataset

with standardised variable names (see load_data._VAR_MAP: z->geopotential
etc. and _DIM_MAP: valid_time->time, pressure_level->level), so the rest of
the pipeline is source-agnostic. `Era5Source` wraps the existing CDS/ERA5
download path so the current behaviour is the reference implementation of
the contract.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import xarray as xr

from ..download import download_range
from ..load_data import normalize_era5


@runtime_checkable
class DataSource(Protocol):
    """Protocol every data source must satisfy."""
    name: str

    def fetch(self, area, start: str, end: str) -> xr.Dataset:
        """Returns standardised (T, level, lat, lon) data for [start, end]."""
        ...


class Era5Source:
    """ERA5/ERA5T via the CDS (reference implementation of DataSource)."""
    name = "era5"

    def fetch(self, area, start: str, end: str) -> xr.Dataset:
        """Download the window and return the normalised xarray Dataset."""
        path = download_range(area, start, end, tag="src")
        if path is None:
            raise RuntimeError("ERA5 download failed (see CDS config)")
        ds = xr.load_dataset(path)          # loads into memory, closes file
        return normalize_era5(ds)