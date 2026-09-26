"""Data-source registry (Stage 3).

Every source implements the same tiny contract:

    fetch(area, start, end) -> xarray.Dataset

where `area` is the [N, W, S, E] bbox, `start`/`end` are inclusive ISO
dates, and the returned Dataset must use the *standardised* names handled
by `weather_pipeline.load_data._VAR_MAP` / `_DIM_MAP` (geopotential,
temperature, u_component_of_wind, v_component_of_wind,
specific_humidity, dims time/level/latitude/longitude).

`get_source(name)` returns a ready instance; unknown names raise. Adding a
real NCMRWF source is therefore a new adapter (cf. sources/ncmrwf.py), not
a rewrite of the pipeline.
"""
from __future__ import annotations

from .base import DataSource, Era5Source
from .ncmrwf import NCMRWFSource

_REGISTRY = {
    "era5": Era5Source,
    "ncmrwf": NCMRWFSource,
}


def get_source(name: str) -> DataSource:
    if name not in _REGISTRY:
        raise ValueError(f"unknown data source {name!r}; "
                         f"choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name]()

__all__ = ["DataSource", "Era5Source", "NCMRWFSource", "get_source"]