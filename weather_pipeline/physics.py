"""Phase 6: physics-informed loss terms using MetPy.

The short-hand the brief uses is "heavy rainfall predicted with no
corresponding moisture convergence". Our forecast/downscaled product is a
geopotential-anomaly field (cyclone = sharp low), so the analogous
physically-*impossible* output is a *sharp extreme peak with no moist
convergence support* at the same location.

Implementation:
  * `moisture_support()`   - grid of [-div(q V)] at a given level/time,
                             computed with MetPy `divergence` +
                             `lat_lon_grid_deltas`; positive = convergence.
  * `standardized_support` - tanh-squashed [0,1] support map.
  * `support_on_nodes()`   - interpolates the support grid onto mesh nodes.
  * `support_in_crop()`    - extracts a crop-aligned support tensor.
  * `gnn_physics_penalty(logit, support)` - penalises high anomaly scores
      on nodes with no convergence support.
  * `diffusion_physics_penalty(x0_hat, support)` - penalises the diffusion
      model's predicted clean field having a large magnitude where the
      conditioning field has no convergence support.

Both terms are differentiable w.r.t. the model output (support is a
constant array) and toggleable via CONFIG.use_physics_loss.
"""
from __future__ import annotations

import numpy as np
import torch

from metpy.calc import divergence, lat_lon_grid_deltas
from metpy.units import units


def moisture_convergence(ds, level: int = 850, t: int = 0) -> np.ndarray:
    """Meteorological moisture convergence field at (level, t).

    conv = -div(q * V_2d)  [kg/(m^2 s)], positive means convergence.
    """
    u = ds["u_component_of_wind"].sel(level=level).isel(time=t).values
    v = ds["v_component_of_wind"].sel(level=level).isel(time=t).values
    q = ds["specific_humidity"].sel(level=level).isel(time=t).values
    lat = np.asarray(ds.latitude.values, dtype=float)
    lon = np.mod(np.asarray(ds.longitude.values, dtype=float), 360.0)
    lon2, lat2 = np.meshgrid(lon, lat)
    dx, dy = lat_lon_grid_deltas(lon2 * units.degrees,
                                 lat2 * units.degrees)
    fu = (q * u) * units("m/s")           # q [kg/kg] * u [m/s]
    fv = (q * v) * units("m/s")
    div = divergence(fu, fv, dx=dx, dy=dy)    # positive = outflow
    return -np.asarray(div.magnitude, dtype=float)


def standardized_support(grid: np.ndarray,
                         k: float = 2.0) -> np.ndarray:
    """tanh-squashed [0,1] convergence support; strong convergence -> 1."""
    g = (grid - grid.mean()) / (grid.std() + 1e-9)
    return np.clip(np.tanh(g * k) * 0.5 + 0.5, 0.0, 1.0)


def support_on_nodes(conv_grid: np.ndarray, lat: np.ndarray,
                     lon: np.ndarray, node_lat: np.ndarray,
                     node_lon: np.ndarray, k: int = 4) -> np.ndarray:
    """Convergence support interpolated onto icosahedral mesh nodes."""
    from .icosahedral import interpolate_field
    g = standardized_support(conv_grid)
    return interpolate_field(g, lat, lon, node_lat, node_lon, k=k)


def support_in_crop(conv_grid: np.ndarray, y0: int, x0: int,
                    size: int = 48) -> np.ndarray:
    """[0,1] support crop aligned with a (y0, x0, size) field crop."""
    return standardized_support(conv_grid[y0:y0 + size, x0:x0 + size])


# ------------------------------------------------------------ penalties --
def gnn_physics_penalty(logit: torch.Tensor, support: torch.Tensor,
                        weight: float = 0.02) -> torch.Tensor:
    """High anomaly score on a node with NO convergence support is
    physically impossible; penalise p^2 * (1 - support)."""
    if support is None or logit.numel() == 0:
        return logit.sum() * 0.0
    p = torch.sigmoid(logit)
    return weight * (p.clamp(min=0) ** 2 * (1.0 - support)).mean()


def diffusion_physics_penalty(x0_hat: torch.Tensor, support: torch.Tensor,
                              weight: float = 0.02) -> torch.Tensor:
    """Predicted clean-field magnitude where support is low = unphysical
    (a sharp extreme with no moist convergence). Magnitude-weighted."""
    if support is None:
        return x0_hat.sum() * 0.0
    mag = x0_hat.abs()
    return weight * (mag * (1.0 - support)).mean()