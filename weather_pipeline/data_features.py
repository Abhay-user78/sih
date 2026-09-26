"""Feature preparation for the Phase 2 GNN.

Builds, from a loaded weather Dataset + EFI fields + an icosphere mesh, a
per-timestep tensor of mesh-node features and anomaly labels:

  node_features[t] : (N, F)  standardised atmospheric + EFI features
  node_labels[t]   : (N,1)   binary anomaly label (1 = anomaly node)

The mesh node set is the bbox-subset (fixed across time).
"""
from __future__ import annotations

import numpy as np

from .config import CONFIG
from .icosahedral import build_mesh, subset_to_bbox, latlon, \
    interpolate_field
from . import efi as efi_mod

# variable / level combinations interpolated onto mesh nodes
RAW_VARLEVELS = [
    ("geopotential", 200), ("geopotential", 500), ("geopotential", 700),
    ("geopotential", 850), ("geopotential", 1000),
    ("temperature", 500), ("temperature", 850),
    ("specific_humidity", 850),
    ("u_component_of_wind", 850), ("v_component_of_wind", 850),
]
EFI_VARLEVELS = [("geopotential", 850), ("temperature", 850),
                 ("specific_humidity", 850)]


def _xyz(lat, lon):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)


class FeatureBuilder:
    def __init__(self, ds, mesh_level: int | None = None):
        self.ds = ds
        self.lat = ds.latitude.values.astype(float)
        self.lon = np.mod(ds.longitude.values.astype(float), 360.0)
        self.n_time = int(ds.sizes["time"])
        mesh = build_mesh(level=mesh_level or CONFIG.mesh_level)
        sub = subset_to_bbox(mesh, CONFIG.domain)
        self.node_lat, self.node_lon = latlon(sub["vertices"])
        self.node_xyz = _xyz(self.node_lat, self.node_lon)
        self.edge_index = sub["edge_index"]
        self.N = len(self.node_lat)
        self._raw = None   # (n_time, N, F_raw)
        self._efi = None   # (n_time, N, F_efi)

    def _grid_field(self, var: str, level: int, t: int) -> np.ndarray:
        da = self.ds[var].sel(level=level)
        return da.isel(time=t).values if "time" in da.dims else da.values

    def build(self, recompute_efi: bool = True) -> "FeatureBuilder":
        raw = np.empty((self.n_time, self.N, len(RAW_VARLEVELS)),
                       dtype=np.float32)
        for ci, (var, lev) in enumerate(RAW_VARLEVELS):
            for t in range(self.n_time):
                f = self._grid_field(var, lev, t)
                raw[t, :, ci] = interpolate_field(f, self.lat, self.lon,
                                                  self.node_lat, self.node_lon,
                                                  k=4)
        self._raw = raw

        efi = np.zeros((self.n_time, self.N, len(EFI_VARLEVELS)),
                       dtype=np.float32)
        if recompute_efi:
            clim = None
            if CONFIG.data_source == "synthetic":
                from .synthetic import climatology_xr
                clim = climatology_xr()
            if clim is not None:
                for ci, (var, lev) in enumerate(EFI_VARLEVELS):
                    cvals = clim[var].sel(level=lev).values
                    for t in range(self.n_time):
                        cur = self._grid_field(var, lev, t)
                        efi_grid = efi_mod.compute_efi(cur, cvals, axis=0)
                        efi[t, :, ci] = interpolate_field(
                            efi_grid, self.lat, self.lon,
                            self.node_lat, self.node_lon, k=4)
            else:
                print("[features] EFI not computed (no climatology available)")
        self._efi = efi
        return self

    @property
    def node_features(self) -> np.ndarray:
        """(n_time, N, F) concatenated raw + efi, standardised per feature."""
        feats = np.concatenate([self._raw, self._efi], axis=2)
        mu = feats.mean(axis=(0, 1), keepdims=True)
        sd = feats.std(axis=(0, 1), keepdims=True) + 1e-6
        return (feats - mu) / sd

    def synthetic_labels(self, radius_km: float = 160.0,
                         soft: bool = True,
                         track: np.ndarray | None = None) -> np.ndarray:
        """Ground-truth anomaly labels from a cyclone track.

        `track` (n_time, 2) lat/lon centres; defaults to the synthetic
        Amphan linspace when None.

        soft=True: smooth decay exp(-d/radius) in [0,1] (regression target).
        soft=False: hard binary mask inside `radius_km`.
        """
        from .synthetic import _dist_grid
        if track is None:
            cy_center_lat = np.linspace(10.5, 22.5, self.n_time)
            cy_center_lon = np.linspace(89.5, 88.7, self.n_time)
        else:
            cy_center_lat = np.asarray(track)[:, 0]
            cy_center_lon = np.asarray(track)[:, 1]
        lab = np.zeros((self.n_time, self.N), dtype=np.float32)
        for t in range(self.n_time):
            d = _dist_grid((cy_center_lat[t], cy_center_lon[t]),
                           self.node_lat, self.node_lon)
            lab[t] = np.exp(-d / radius_km) if soft else \
                (d <= radius_km).astype(np.float32)
        return lab


def prepare_dataset(ds, mesh_level=None, track=None,
                    labels: bool = True) -> dict:
    """One-call builder used by phase2_train/detect.

    `labels=False` skips the ground-truth label block entirely. Used by the
    live/operational path, which runs WITHOUT any best-track truth (labels
    are only ever used for training, never for inference).
    """
    fb = FeatureBuilder(ds, mesh_level=mesh_level).build(recompute_efi=True)
    feats = fb.node_features                       # (T, N, F)
    if not labels:
        labels_arr = np.zeros((feats.shape[0], fb.N), dtype=np.float32)
    elif CONFIG.data_source == "synthetic" and track is None:
        labels_arr = fb.synthetic_labels()
    else:
        from .tracks import AMPHAN_TRACK
        labels_arr = fb.synthetic_labels(
            track=track if track is not None else AMPHAN_TRACK[:fb.n_time])
    node_conv = None
    if CONFIG.use_physics_loss:
        from . import physics
        if {"u_component_of_wind", "v_component_of_wind",
            "specific_humidity"} <= set(ds.data_vars):
            conv = np.empty((feats.shape[0], fb.N), dtype=np.float32)
            for t in range(feats.shape[0]):
                c = physics.moisture_convergence(ds, 850, t)
                conv[t] = physics.support_on_nodes(c, fb.lat, fb.lon,
                                                   fb.node_lat, fb.node_lon)
            node_conv = conv
            print(f"[physics] node convergence-support computed "
                  f"({conv.sum() / conv.size:.2f} mean support)")
        else:
            print("[physics] moisture-convergence fields unavailable; "
                  "node_conv=None")
    return {
        "features": feats,
        "labels": labels_arr,
        "node_lat": fb.node_lat,
        "node_lon": fb.node_lon,
        "node_xyz": fb.node_xyz,
        "edge_index": fb.edge_index,
        "n_time": feats.shape[0],
        "n_nodes": fb.N,
        "feature_dim": feats.shape[2],
        "node_conv": node_conv,
    }