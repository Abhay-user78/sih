"""Phase 3 data preparation: cropping anomaly regions into
(coarse, target) pairs for conditional diffusion downscaling.

Task formulation (placeholder, real at runtime):
  * "coarse" = the forecast-resolution field. On the synthetic test case the
    native grid is 0.25 deg (~25 km); we emulate a coarser 12-km-class
    forecast by block-averaging 2x2 (0.5 deg) then bilinearly upsampling
    back to the target grid.
  * "target" = the sharp field we want at ~5-km class (here the native
    0.25 deg grid). The linear upscale factor is `scale` (=2 in this setup;
    parameterised so 12km->5km, factor ~2.4, and 0.25->0.125 deg work with
    the same code path at runtime).
  * Crops are centred on the moving cyclone track (Phase 2/GNN output at
    runtime), with jitter for augmentation.

Normalisation: each variable/level is mapped to [-1,1] using dataset
min/max; `stats` saved alongside for inverse transforms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DATA_PROCESSED


@dataclass
class DownscaleData:
    coarse: np.ndarray   # (n, 1, H, W) upscaled coarse (cond channel)
    target: np.ndarray   # (n, 1, H, W) sharp target
    stats: dict          # min/max for inverse transform
    lat: np.ndarray      # 1-D crop latitudes (W,)
    lon: np.ndarray      # 1-D crop longitudes
    timesteps: np.ndarray
    levels: np.ndarray
    cyclone_centre: np.ndarray  # (n, 2) lat, lon
    support: np.ndarray | None = None   # (n, 1, H, W) [0,1] convergence


def _block_mean(field: np.ndarray, f: int) -> np.ndarray:
    """Block-average a 2-D field by factor f."""
    H, W = field.shape
    h, w = H // f, W // f
    return field[:h * f, :w * f].reshape(h, f, w, f).mean(axis=(1, 3))


def _bilinear_up(field: np.ndarray, shape) -> np.ndarray:
    """Bilinear resize via scipy.ndimage.zoom (order=1)."""
    from scipy.ndimage import zoom
    h, w = field.shape
    H, W = shape
    return zoom(field, (H / h, W / w), order=1)


def _bilinear2(field: np.ndarray, H, W) -> np.ndarray:
    """Bilinear upsampling (scipy.ndimage.zoom, order=1)."""
    from scipy.ndimage import zoom
    return zoom(field, (H / field.shape[0], W / field.shape[1]), order=1)


def build_crops(ds, levels=(850,), var="geopotential",
                size: int = 48, scale: int = 2,
                jitter_px: int = 3, seed: int = 0,
                track: np.ndarray | None = None) -> DownscaleData:
    """Build coarse/target crop pairs centred on the cyclone track.

    ds       : xarray Dataset (synthetic or ERA5-shaped)
    levels   : pressure levels to use (each becomes an independent sample
               row; levels are treated as channels-repeats for now)
    size     : target crop edge length in native grid points
    scale    : downscale factor of the coarse grid (block-mean + upsample)
    track    : (nt, 2) lat/lon cyclone centres; defaults to the synthetic
               Amphan linspace to preserve Phase-3 behaviour
    """
    rng = np.random.default_rng(seed)
    lat = np.asarray(ds.latitude.values, dtype=float)
    lon = np.mod(np.asarray(ds.longitude.values, dtype=float), 360.0)
    nt = ds.sizes["time"]

    if track is not None:
        track = np.asarray(track, dtype=float)
        cy_lat = track[:nt, 0]
        cy_lon = track[:nt, 1]
    else:
        cy_lat = np.linspace(10.5, 22.5, nt)
        cy_lon = np.linspace(89.5, 88.7, nt)

    # Phase-6 physics support (moisture convergence), optional
    from .config import CONFIG
    from . import physics as phys
    supports = None
    if CONFIG.use_physics_loss:
        _conv_cache = {}
        supports = []

    half = size // 2
    out_c, out_t, out_ts, out_lev, out_cy = [], [], [], [], []
    for t in range(nt):
        ci_lat = int(np.argmin(np.abs(lat - cy_lat[t])))
        ci_lon = int(np.argmin(np.abs(lon - cy_lon[t])))
        for lv in levels:
            f = ds[var].sel(level=lv).isel(time=t).values
            # clamp crop so it fits; jitter within bounds
            jy = rng.integers(-jitter_px, jitter_px + 1)
            jx = rng.integers(-jitter_px, jitter_px + 1)
            y0 = min(max(ci_lat - half + jy, 0), len(lat) - size)
            x0 = min(max(ci_lon - half + jx, 0), len(lon) - size)
            crop = f[y0:y0 + size, x0:x0 + size]
            coarse = _bilinear2(_block_mean(crop, scale), size, size)
            out_t.append(crop)
            out_c.append(coarse)
            out_ts.append(t)
            out_lev.append(lv)
            out_cy.append((cy_lat[t], cy_lon[t]))
            if supports is not None:
                conv = _conv_cache.get(lv)
                if conv is None:
                    conv = phys.moisture_convergence(ds, lv, t)
                    _conv_cache[lv] = conv
                supports.append(phys.support_in_crop(conv, y0, x0, size))

    target = np.stack(out_t)[:, None].astype(np.float32)
    coarse = np.stack(out_c)[:, None].astype(np.float32)

    # normalise to [-1, 1] (vmin/vmax over dataset crops)
    tmin, tmax = float(target.min()), float(target.max())
    tr = target.max() - target.min()
    target_n = (target - tmin) / (tr + 1e-9) * 2 - 1
    coarse_n = np.clip((coarse - tmin) / (tr + 1e-9) * 2 - 1, -1, 1)

    # crop lon/lat reference (following the first crop's x0)
    sup = (np.stack(supports)[:, None].astype(np.float32)
           if supports is not None else None)
    return DownscaleData(
        coarse=coarse_n, target=target_n,
        stats={"vmin": tmin, "vmax": tmax, "scale": scale, "var": var},
        lat=lat[y0:y0 + size], lon=lon[x0:x0 + size],
        timesteps=np.asarray(out_ts), levels=np.asarray(out_lev),
        cyclone_centre=np.asarray(out_cy),
        support=sup,
    )


def save_ds(dd: DownscaleData, name: str = "phase3_dataset.npz") -> str:
    p = DATA_PROCESSED / name
    sup = (dd.support if dd.support is not None
           else np.ones_like(dd.coarse))
    np.savez(p, coarse=dd.coarse, target=dd.target, lat=dd.lat, lon=dd.lon,
             timesteps=dd.timesteps, levels=dd.levels,
             cyclone_centre=dd.cyclone_centre,
             stats_vmin=dd.stats["vmin"], stats_vmax=dd.stats["vmax"],
             stats_scale=dd.stats["scale"], stats_var=dd.stats["var"],
             support=sup)
    print(f"[downscale] saved {p}")
    return str(p)


def load_ds(name: str = "phase3_dataset.npz") -> DownscaleData:
    z = np.load(DATA_PROCESSED / name)
    sup = z["support"] if "support" in z else None
    return DownscaleData(
        coarse=z["coarse"], target=z["target"],
        stats={"vmin": float(z["stats_vmin"]), "vmax": float(z["stats_vmax"]),
               "scale": int(z["stats_scale"]), "var": str(z["stats_var"])},
        lat=z["lat"], lon=z["lon"], timesteps=z["timesteps"],
        levels=z["levels"], cyclone_centre=z["cyclone_centre"],
        support=sup,
    )


def inv_normalise(x: np.ndarray, stats: dict) -> np.ndarray:
    a = (np.asarray(x) + 1) / 2
    return a * (stats["vmax"] - stats["vmin"]) + stats["vmin"]