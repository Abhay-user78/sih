"""Global configuration for the weather anomaly pipeline.

Coordinate conventions documented here (per brief rule):
  - ERA5 from cdsapi NetCDF: lon in [0, 360], lat descending north->south,
    0.25 deg grid.
  - NCUM/NEPS-G (12 km): to be documented once obtained.
  - GFS (fallback): lon in [0, 360], lat ascending.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS = BASE_DIR / "outputs"
MODELS = BASE_DIR / "models"

for _d in (DATA_RAW, DATA_PROCESSED, OUTPUTS, MODELS):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
@dataclass
class AmphanDomain:
    """Cyclone Amphan (May 2020, Bay of Bengal) test-case region.

    bbox is (north, west, south, east) in degrees, ERA5 cdsapi convention.
    """
    north: float = 30.0
    west: float = 75.0
    south: float = 5.0
    east: float = 100.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.north, self.west, self.south, self.east)


@dataclass
class Era5Request:
    """ERA5 pressure-level request params (see agent.txt section 4)."""
    dataset: str = "reanalysis-era5-pressure-levels"
    product_type: str = "reanalysis"
    variables: list[str] = field(default_factory=lambda: [
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
    ])
    pressure_levels: list[str] = field(default_factory=lambda: [
        "200", "500", "700", "850", "1000",
    ])
    year: str = "2020"
    month: str = "05"
    days: list[str] = field(default_factory=lambda: [
        "16", "17", "18", "19", "20", "21",
    ])
    times: list[str] = field(default_factory=lambda: [
        "00:00", "06:00", "12:00", "18:00",
    ])
    domain: AmphanDomain = field(default_factory=AmphanDomain)
    grid: str = "0.25/0.25"
    fmt: str = "netcdf"
    target: str = "era5_amphan_test.nc"

    def to_dict(self) -> dict:
        return {
            "product_type": self.product_type,
            "variable": self.variables,
            "pressure_level": self.pressure_levels,
            "year": self.year,
            "month": self.month,
            "day": self.days,
            "time": self.times,
            "area": list(self.domain.bbox),
            "grid": self.grid,
            "format": self.fmt,
        }


# ---------------------------------------------------------------------------
# Pipeline knobs
# ---------------------------------------------------------------------------
@dataclass
class Config:
    domain: AmphanDomain = field(default_factory=AmphanDomain)
    era5: Era5Request = field(default_factory=Era5Request)

    # icosahedral mesh subdivision level for the GNN (Phase 2/3)
    # vertex spacing ~= 63 deg / 2**level. level 7 ~= 0.5 deg regional GNN
    mesh_level: int = 7
    # nearest neighbours kept per node when building the mesh graph
    mesh_k: int = 6

    # EFI thresholds (ECMWF convention, unusual / very unusual)
    efi_unusual: float = 0.5
    efi_extreme: float = 0.8

    device: str = "cuda"  # fall back to cpu at runtime if unavailable

    # Phase 6 physics-informed loss toggle (off until Phase 6)
    use_physics_loss: bool = False
    # weights of the physics penalty terms (gentle regularisers)
    physics_gnn_weight: float = 0.005
    physics_diff_weight: float = 0.02

    # placeholders: replaced when real operational data (NCUM/NEPS-G) is used
    data_source: str = "synthetic"  # synthetic | era5

    seed: int = 42


CONFIG = Config()

# dask / xarray chunking for large files
XARRAY_CHUNKS: dict = {"time": 1}


def model_dir(sub: str) -> Path:
    d = MODELS / sub
    d.mkdir(parents=True, exist_ok=True)
    return d