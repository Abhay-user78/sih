"""PLACEHOLDER dataset generator (Amphan-like cyclone) used for offline
development until real ERA5 credentials are configured.

This is NOT real data. It produces a synthetic but physically-shaped tropical
cyclone field (vortex winds, low-level geopotential fall, upper-level warm
core, moist core) moving across the Bay of Bengal in May 2020, plus a
multi-year synthetic climatology used to exercise the EFI baseline.

When real ERA5 data is available, code paths switch via CONFIG.data_source.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .config import CONFIG, DATA_RAW

DEG_KM = 111.32


def _dist_grid(center: tuple[float, float], llat, llon) -> np.ndarray:
    """Great-circle-ish distance (km) from center for each (lat,lon) grid pt."""
    clat, clon = np.deg2rad(center[0]), np.deg2rad(center[1])
    plat, plon = np.deg2rad(llat), np.deg2rad(llon)
    dlat = plat - clat
    dlon = plon - clon
    a = np.sin(dlat / 2) ** 2 + np.cos(clat) * np.cos(plat) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.0 * c


def make_grid():
    d = CONFIG.domain
    lat = np.arange(d.north, d.south - 1e-9, -0.25)          # descending
    lon = np.arange(d.west, d.east + 1e-9, 0.25)             # 0..360
    return lat, lon


def _background_field(llat, llon, seed=0):
    """Smooth seasonal background + mild meridional gradient per variable."""
    rng = np.random.default_rng(seed)
    latn = (llat - llat.min()) / (llat.max() - llat.min() + 1e-9)  # 0..1 N->S
    lonn = (llon - llon.min()) / (llon.max() - llon.min() + 1e-9)
    # broad sinusoidal gradient + smooth noise
    f = 0.5 * np.sin(2 * np.pi * (latn + 0.6 * lonn - 0.5)) + 0.3 * np.cos(
        3 * np.pi * lonn - 1.2 * latn)
    f = f + 0.15 * rng.standard_normal(np.shape(llat))
    f = (f - f.mean()) / (f.std() + 1e-9)
    return f


def _cyclone(lat, lon, center_lat, center_lon, r_max_km=60.0, v_max=45.0):
    """Cyclone anomaly fields on the (lat,lon) grid.

    Returns dict field -> (lat,lon) arrays (scaled below in build_data).
    """
    llat, llon = np.meshgrid(lat, lon, indexing="ij")
    dist = _dist_grid((center_lat, center_lon), llat, llon)

    # Rankine vortex (m/s): solid-body rotation inside r_max, decay outside
    # with an easing envelope so the storm stays localised on the domain.
    env = np.exp(-((dist - r_max_km) / 300.0) ** 2)
    vt = np.where(dist < r_max_km, v_max * dist / np.maximum(r_max_km, 1e-9),
                  v_max * (r_max_km / np.maximum(dist, 1e-9)) * env)
    # unit vectors tangential = (-sin(theta), +cos(theta)) in (del, dlambda)
    thet = np.arctan2(llat - center_lat, llon - center_lon)
    u_cycl = -vt * np.sin(thet)   # eastward component
    v_cycl = vt * np.cos(thet)    # northward component

    # low-level geopotential / pressure fall (m2/s2, -ive), warm-core aloft
    gph_850 = -2400.0 * np.exp(-(dist ** 2) / (2 * (120.0) ** 2))
    gph_200 = +1200.0 * np.exp(-(dist ** 2) / (2 * (140.0) ** 2))
    temp_core = 6.0 * np.exp(-(dist ** 2) / (2 * (130.0) ** 2))  # K, warm core
    q_core = 8e-3 * np.exp(-(dist ** 2) / (2 * (90.0) ** 2))     # kg/kg moist
    return {
        "gph_low": gph_850,
        "gph_high": gph_200,
        "temp_core": temp_core,
        "q_core": q_core,
        "u_cycl": u_cycl,
        "v_cycl": v_cycl,
    }


def build_climatology(n_climate_years: int = 30, seed=7) -> dict:
    """Synthetic ~30-year climate field (no cyclone events) covering the same
    spatial grid. Returns dict of per-level base fields and an array.
    """
    lat, lon = make_grid()
    llat, llon = np.meshgrid(lat, lon, indexing="ij")
    rng = np.random.default_rng(seed)

    levels = [200, 500, 700, 850, 1000]
    t0 = 240.0 - (llat - 15) / 25 * 18.0 + _background_field(llat, llon, 2) * 2
    h500 = 5.6e4 + 1.2e4 * (llat - 30) / 15.0 + _background_field(llat, llon, 1) * 300
    base = {}
    base["geopotential"] = {}
    base["temperature"] = {}
    base["specific_humidity"] = {}
    base["u_component_of_wind"] = {}
    base["v_component_of_wind"] = {}
    hsed = {200: h500 + 4e3, 500: h500, 700: h500 + 6e3,
            850: h500 + 1.0e4, 1000: h500 + 1.8e4 - 3.0e3 * (1 - (llat - 5) / 25)}
    tsed = {200: t0 - 50, 500: t0 + 5, 700: t0, 850: t0 - 4, 1000: t0 - 6}
    qsed = {200: 5e-6, 500: 1e-3, 700: 4e-3, 850: 9e-3, 1000: 1.4e-2}
    ubg = -3.0 * (1 - (llat - 5) / 25) - 4.0 * (llon - llon.min()) / (
        llon.max() - llon.min() + 1e-9)
    vbg = 0.8 * np.sin(2 * np.pi * (llat - llat.min()) / (
        llat.max() - llat.min() + 1e-9))
    for lv in levels:
        base["geopotential"][lv] = hsed[lv]
        base["temperature"][lv] = tsed[lv]
        base["specific_humidity"][lv] = qsed[lv]
        base["u_component_of_wind"][lv] = ubg
        base["v_component_of_wind"][lv] = vbg

    clim = {}
    for vname, levs in base.items():
        arr = np.zeros((n_climate_years, len(levs), len(lat), len(lon)), dtype=np.float32)
        for li, (lev, bg) in enumerate(levs.items()):
            bg = np.asarray(bg)
            for y in range(n_climate_years):
                noise = 0.008 * np.abs(bg) + 0.002 * np.abs(bg).mean()
                arr[y, li] = bg + rng.standard_normal(np.shape(bg)) * noise
        clim[vname] = arr
    return {"levels": list(base["geopotential"].keys()), "data": clim,
            "n_years": n_climate_years}


def build_synthetic_amphan(save: bool = True) -> xr.Dataset:
    """Build an ERA5-shaped NetCDF Dataset containing an Amphan-like cyclone
    during 16-21 May 2020 (6 days x 4 steps) on the config domain."""
    lat, lon = make_grid()
    levels = [200, 500, 700, 850, 1000]
    times = np.array(
        [f"2020-05-{d:02d} {h:02d}:00" for d in range(16, 22)
         for h in (0, 6, 12, 18)],
        dtype="datetime64[h]",
    )
    n = len(times)

    # cyclone track (Amphan-like: heading NNE, 21 May landfall ~ lat 22)
    track = {
        "lat": np.linspace(10.5, 22.5, n),
        "lon": np.linspace(89.5, 88.7, n),
        "vmax": np.linspace(28, 55, n),   # intensifying m/s
    }

    def var_space(vname, lev):
        """(time, lat, lon) field for one variable/level."""
        c = build_climatology()["data"][vname][:, levels.index(lev)][:].mean(axis=0)
        bg = np.asarray(c, dtype=np.float64)
        llat, llon = np.meshgrid(lat, lon, indexing="ij")
        out = np.empty((n, len(lat), len(lon)), dtype=np.float32)
        for t in range(n):
            cy = _cyclone(lat, lon, track["lat"][t], track["lon"][t],
                          v_max=track["vmax"][t])
            f = bg.copy()
            if vname == "geopotential":
                f = f + (cy["gph_low"] if lev <= 850 else cy["gph_high"]) * (1.0 if lev == 850 else 0.5)
            elif vname == "temperature":
                f = f + cy["temp_core"] * (1.0 if lev >= 500 else 0.4)
            elif vname == "specific_humidity":
                f = f + cy["q_core"] * (1.0 if lev >= 700 else 0.2)
            elif vname == "u_component_of_wind":
                f = f + cy["u_cycl"]
            elif vname == "v_component_of_wind":
                f = f + cy["v_cycl"]
            out[t] = f
        return out

    data_vars = {}
    for vname in ("geopotential", "temperature", "specific_humidity",
                  "u_component_of_wind", "v_component_of_wind"):
        data_vars[vname] = xr.DataArray(
            np.stack([var_space(vname, lv) for lv in levels], axis=1),
            dims=("time", "level", "latitude", "longitude"),
            coords={"time": times, "level": levels,
                    "latitude": lat, "longitude": lon},
            attrs={"units": {"geopotential": "m2 s-2",
                             "temperature": "K",
                             "specific_humidity": "kg kg-1",
                             "u_component_of_wind": "m s-1",
                             "v_component_of_wind": "m s-1"}[vname],
                   "source": "synthetic-placeholder (Amphan-like) v1"},
        )

    ds = xr.Dataset(data_vars)
    if save:
        out = DATA_RAW / "synthetic_amphan.nc"
        ds.to_netcdf(out)
        print(f"[synthetic] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return ds


def climatology_xr() -> xr.Dataset:
    """Return climatology as an xarray Dataset shaped (year, level, lat, lon),
    matching the grid of `build_synthetic_amphan`, for EFI computation."""
    lat, lon = make_grid()
    c = build_climatology()
    levels = c["levels"]
    coords = {"year": np.arange(c["n_years"]), "level": levels,
              "latitude": lat, "longitude": lon}
    ds = xr.Dataset({k: (("year", "level", "latitude", "longitude"), v)
                     for k, v in c["data"].items()}, coords=coords)
    return ds


if __name__ == "__main__":
    build_synthetic_amphan(save=True)