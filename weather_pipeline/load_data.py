"""Phase 1b: load NetCDF/GRIB weather files with xarray, inspect and
normalise coordinates.

Coordinate handling (documented):
  - ERA5 netCDF from cdsapi: lon in [0, 360), lat descending.
  - latitude/longitude may appear as dimensions or as 2-D coordinate grids
    (curvilinear). We handle both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .config import CONFIG, DATA_RAW


@dataclass
class LoadedData:
    ds: xr.Dataset
    path: str
    lon: np.ndarray
    lat: np.ndarray
    kinds: dict  # variable -> coarse kind ('vector' | 'scalar')

    def inspect(self) -> None:
        print("=" * 70)
        print("DATASET:", self.path)
        print("=" * 70)
        for name, da in self.ds.data_vars.items():
            dims = ", ".join(da.dims)
            print(f"  {name:<24} dims=({dims}) dtype={da.dtype}")

    def physical_kind(self, var: str) -> str:
        """Heuristic: vector fields (u/v wind) are u- or v-sensitive."""
        return self.kinds.get(var, "scalar")


def _mean_latlon(ds: xr.Dataset):
    lat = ds.latitude.values
    lon = ds.longitude.values
    lat = np.asarray(lat, dtype=float).squeeze()
    lon = np.asarray(lon, dtype=float).squeeze()
    if lat.ndim == 2 and lon.ndim == 2:
        # curvilinear grid: collapse to 1-D decimated axes (used only for
        # mesh interpolation bookkeeping)
        lat = lat.mean(axis=1)
        lon = lon.mean(axis=0)
    # normalise lon to [0, 360)
    lon = np.mod(lon, 360.0)
    sort = np.argsort(lon)
    lon = lon[sort]
    return lat, lon, sort


_DIM_MAP = {"valid_time": "time", "pressure_level": "level"}
_VAR_MAP = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
}


def normalize_era5(ds: xr.Dataset) -> xr.Dataset:
    """Rename cdsapi GRIB-shortname variables/dims to pipeline names."""
    renames = {k: v for k, v in _VAR_MAP.items() if k in ds}
    renames.update({k: v for k, v in _DIM_MAP.items() if k in ds.dims})
    if renames:
        ds = ds.rename(renames)
    drop = [c for c in ("expver", "number", "step") if c in ds.coords]
    if drop:
        ds = ds.drop_vars(drop, errors="ignore")
    return ds


def open_data(path: str | None = None) -> LoadedData:
    """Open an ERA5 netCDF file and print/return its structure."""
    path = path or str(DATA_RAW / CONFIG.era5.target)
    ds = xr.open_dataset(path)
    ds = normalize_era5(ds)
    lat, lon, sort = _mean_latlon(ds)

    kinds = {}
    for name in ds.data_vars:
        lname = name.lower()
        if lname in {"u_component_of_wind", "v_component_of_wind"}:
            kinds[name] = "vector"
        else:
            kinds[name] = "scalar"

    return LoadedData(ds=ds, path=str(path), lon=lon, lat=lat, kinds=kinds)


def select_subset(field: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                  domain=None):
    """Return field values masked to the domain, plus 1-D lon/lat subset.

    Returns (subset_values, sub_lon, sub_lat, row_col_index_mapping) where
    mapping lets us place mesh results back onto the full grid.
    """
    if domain is None:
        domain = CONFIG.domain
    lat_ok = np.where((lat >= domain.south) & (lat <= domain.north))[0]
    lon_ok = np.where((lon >= domain.west) & (lon <= domain.east))[0]
    sub_lat = lat[lat_ok]
    sub_lon = lon[lon_ok]
    # ERA5 lat desc, lon asc -> meshgrid (lat, lon)
    llat, llon = np.meshgrid(sub_lat, sub_lon, indexing="ij")
    idx = np.ravel_multi_index(
        np.meshgrid(np.arange(len(sub_lat)), np.arange(len(sub_lon)),
                    indexing="ij"),
        (len(sub_lat), len(sub_lon)),
    )
    return (llat, llon, idx, lat_ok, lon_ok)