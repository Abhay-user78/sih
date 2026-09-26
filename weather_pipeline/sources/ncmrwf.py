"""NCMRWF adapter placeholder (Stage 3).

Real access to NCMRWF's NCUM/NEPS-G (12 km) operational products has not
been granted yet, so this is a deliberate seam: the class exists, is
registered under "ncmrwf", and raises NotImplementedError if anyone calls
fetch() before bulk access is provisioned. Fill this file in once access
is arranged (see rds.ncmrwf.gov.in) — implement fetch() per the contract in
base.py and the pipeline needs no other changes.
"""
from __future__ import annotations

import xarray as xr

MISSING_ACCESS = (
    "NCMRWF bulk access not yet granted. Once access is available "
    "(see rds.ncmrwf.gov.in), implement NCMRWFSource.fetch() to return an "
    "xarray.Dataset with the standardised variables handled by "
    "weather_pipeline.load_data._VAR_MAP (geopotential, temperature, "
    "u_component_of_wind, v_component_of_wind, specific_humidity) and dims "
    "time/level/latitude/longitude over the requested [N,W,S,E] bbox and "
    "[start, end] ISO window. No other pipeline changes are required."
)


class NCMRWFSource:
    """NCUM/NEPS-G (12 km) operational data. Not implemented yet."""
    name = "ncmrwf"

    def fetch(self, area, start: str, end: str) -> xr.Dataset:
        raise NotImplementedError(MISSING_ACCESS)