"""Phase 1d (real-data path): build the ERA5 climatology baseline used by
the EFI. Intended to run once CDS credentials are configured.

Downloads geopotential for May 1990-2019 (30 years) over the Amphan domain
and stores per-gridpoint location/year samples at data/processed/
era5_climatology.nc. This is the baseline for EFI in phase1_run.py.

Note: full 30-year multi-variable climatology is a large download; this
script defaults to a single variable (see --variable) to remain lightweight.
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr

from weather_pipeline.config import CONFIG, DATA_RAW, DATA_PROCESSED
from weather_pipeline.download import cds_configured


def build_baseline(variable="geopotential", levels=("500", "850"),
                   years=None, days=("16",)) -> None:
    if not cds_configured():
        sys.exit("[climatology] CDS credentials missing - cannot download.")
    if years is None:
        years = [str(y) for y in range(1990, 2020)]
    import cdsapi

    tmp = DATA_RAW / "era5_climatology_downloaded.nc"
    c = cdsapi.Client()
    print(f"[climatology] downloading {len(years)} years x {days} days x "
          f"{len(levels)} levels of {variable} ...")
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": [variable],
            "pressure_level": list(levels),
            "year": years,
            "month": list(CONFIG.era5.month.split()) if isinstance(
                CONFIG.era5.month, str) else CONFIG.era5.month,
            "day": list(days),
            "time": ["00:00"],
            "area": list(CONFIG.domain.bbox),
            "grid": "0.25/0.25",
            "format": "netcdf",
        },
        str(tmp),
    )
    ds = xr.open_dataset(tmp)
    # squeeze month/day/time; keep year samples along 'time'
    stacked = ds.assign_coords(year=("time", np.arange(len(ds.time))))
    out = stacked.stack(sample=("time",)).transpose("sample", "level",
                                                     "latitude", "longitude")
    out.to_netcdf(DATA_PROCESSED / "era5_climatology.nc")
    print(f"[climatology] saved {DATA_PROCESSED / 'era5_climatology.nc'}")
    ds.close()


if __name__ == "__main__":
    build_baseline(variable=sys.argv[1] if len(sys.argv) > 1 else "geopotential")