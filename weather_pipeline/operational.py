"""Operational cycle: run the trained GNN -> diffusion -> temporal pipeline
on a freshly downloaded (live) ERA5T window with NO best-track truth.

This is the deployment path: models are loaded from disk (never retrained),
detections are self-located, crops follow the detections, and the outputs
are written as alerts/forecasts exactly like real operation. Every helper
needed by scripts/live_run.py and scripts/run_pipeline_cycle.py lives here
so the scripts stay thin and their logic is not duplicated.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import CONFIG, OUTPUTS, model_dir
from .load_data import open_data
from .data_features import prepare_dataset
from .pipeline import detection_confidence
from . import gnn, diffusion as diff
from .downscale_data import build_crops
from . import temporal as tp


def km(lat1, lon1, lat2, lon2):
    a = np.sin(np.deg2rad(lat2 - lat1) / 2) ** 2 + \
        np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * \
        np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def detect_centres(dataset, model, device=None):
    """GNN anomaly scores -> per-timestep region centres/bboxes and the
    per-timestep cluster mean/max anomaly scores (NaN on a miss).

    Missed timesteps carry NaN centres/bboxes (unlike the phase_era5 helper,
    which leaves zeros), so callers never mistake a miss for lat/lon (0,0).
    """
    T = int(dataset["n_time"])
    probs = gnn.predict(dataset, model, 0, T, device=device)
    centres = np.full((T, 2), np.nan, dtype=float)
    bboxes = np.full((T, 4), np.nan, dtype=float)
    clust_mean = np.full(T, np.nan, dtype=float)
    clust_max = np.full(T, np.nan, dtype=float)
    miss = np.ones(T, dtype=float)
    for t in range(T):
        reg = gnn.bounding_region(probs[t], thr=None,
                                  node_lat=dataset["node_lat"],
                                  node_lon=dataset["node_lon"])
        if "centre" in reg:
            centres[t] = reg["centre"]["lat"], reg["centre"]["lon"]
            bboxes[t] = reg["bbox"]
            clust_mean[t] = float(reg["clusters"][0]["mean_score"])
            clust_max[t] = float(reg["clusters"][0]["max_score"])
            miss[t] = 0.0
    return probs, centres, miss, bboxes, clust_mean, clust_max


def refine_vortex_core(ds, bboxes, truth=None, T=None, half_deg=1.25):
    """Centre each flagged region on the 850 hPa geopotential minimum.

    `truth` is optional: when a region is missing, the position falls back
    to `truth[t]` if given, otherwise it stays NaN (caller decides).
    """
    T = T if T is not None else int(ds.sizes["time"])
    z = ds["geopotential"].sel(level=850).values          # (T, lat, lon)
    lat = np.asarray(ds.latitude.values, dtype=float)
    lon = np.mod(np.asarray(ds.longitude.values, dtype=float), 360.0)
    core = np.full((T, 2), np.nan, dtype=float)
    for t in range(T):
        if np.isnan(bboxes[t]).any():
            if truth is not None:
                core[t] = np.asarray(truth[t], dtype=float)
            continue
        slat, nlat, wlon, elon = bboxes[t]
        ilat = np.where((lat >= slat - half_deg) & (lat <= nlat + half_deg))[0]
        ilon = np.where((lon >= wlon - half_deg) & (lon <= elon + half_deg))[0]
        if len(ilat) == 0 or len(ilon) == 0:
            if truth is not None:
                core[t] = np.asarray(truth[t], dtype=float)
            continue
        crop = z[t][ilat][:, ilon]
        iy, ix = np.unravel_index(np.argmin(crop), crop.shape)
        core[t] = (float(lat[ilat[iy]]), float(lon[ilon[ix]]))
    return core


def _fill_track(cores: np.ndarray) -> np.ndarray:
    """Forward-fill (then backward) so consumers get a finite track."""
    tr = np.asarray(cores, dtype=float).copy()
    bad = ~np.isfinite(tr).all(axis=1)
    last = None
    for t in range(len(tr)):
        if bad[t]:
            if last is not None:
                tr[t] = last
        else:
            last = tr[t]
    if bad.any():                            # leading misses: back-fill
        for t in range(len(tr)):
            if bad[t] and np.isfinite(tr[t]).all():
                break
            tr[t] = tr[t] if np.isfinite(tr[t]).all() \
                else np.nan_to_num(tr[~bad][0] if (~bad).any() else 0.0)
    return tr


def _load_weights(path, kind):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing {kind} weights: {p}")
    return torch.load(p, map_location="cpu", weights_only=False)


def _radius_km(lat, lon, bbox):
    """Approx distance from centre to farthest bbox corner (km)."""
    if bbox is None or any(np.isnan(x) for x in bbox):
        return 0.0
    slat, nlat, wlon, elon = [float(x) for x in bbox]
    return max(km(lat, lon, nlat, elon), km(lat, lon, slat, wlon))


def run_live_cycle(era5_path, out_dir, *, members: int = 8,
                   steps: int = 50, warmup: int = 10,
                   gnn_path=None, diff_path=None, tracker_path=None,
                   do_track: bool = True, label: str = "live") -> dict:
    """Full live chain on one ERA5T file; writes artefacts under `out_dir`.

    Returns the summary dict (also saved to out_dir/live_report.json).
    Raises on any missing weight or broken field so failures surface.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.data_source = "era5"
    CONFIG.use_physics_loss = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ld = open_data(str(era5_path))
    ds = ld.ds
    T = int(ds.sizes["time"])
    ts = [str(ds.time.values[t])[:19] for t in range(T)]
    print(f"--- live cycle: {ld.path} ---")
    print(f"  {T} timesteps, grid {ds.sizes}")

    # ---- [2] GNN detection (inference only) -----------------------------
    dataset = prepare_dataset(ds, labels=False)   # no truth on live data
    ck = _load_weights(gnn_path or model_dir("gnn") / "gnn_anomaly_era5.pt",
                       "GNN")
    model = gnn.AnomalyGNN(int(dataset["feature_dim"]), hidden=int(ck["hidden"]))
    model.load_state_dict(ck["state"])
    probs, centres, miss, bboxes, clust_mean, clust_max = \
        detect_centres(dataset, model, device)
    cores = refine_vortex_core(ds, bboxes, truth=None, T=T)
    det_t = [t for t in range(T) if np.isfinite(cores[t]).all()]
    print(f"  detected {len(det_t)}/{T} timesteps "
          f"(no truth available on live data)")

    # ---- [3] diffusion downscaling centred on the detections ------------
    unet, sched = diff.load_downscale(
        path=diff_path or model_dir("diffusion") / "downscale_unet_era5.pt",
        device=device)
    track = _fill_track(cores)
    dd = build_crops(ds, levels=(850,), var="geopotential",
                     size=48, scale=2, track=track)
    vmin, vmax = dd.stats["vmin"], dd.stats["vmax"]

    gains, sev_means, alerts = [], [], []
    for t in det_t:
        ens = diff.sample_crop(unet, sched, dd.coarse[t:t + 1],
                               n_members=members, n_steps=steps,
                               device=device, seed=t)
        m_phys = (ens + 1) / 2 * (vmax - vmin) + vmin
        vm = m_phys[:, 0]
        vs = (dd.coarse[t, 0] + 1) / 2 * (vmax - vmin) + vmin
        coarse_peak = float(np.abs(vs).max())
        ds_peak = float(np.abs(vm).max(axis=(1, 2)).mean())
        gain = ds_peak / max(coarse_peak, 1e-9)
        sev = diff.severity_from_ensemble(ens, dd.coarse[t, 0])
        sev_p = float(np.mean(sev["severity_prob"]))
        gains.append(gain); sev_means.append(sev_p)
        severity = ("severe" if gain >= 1.0 and sev_p > 0.4 else
                    "moderate" if gain >= 0.9 else "low")
        clat, clon = float(cores[t, 0]), float(cores[t, 1])
        gnn_ms = (float(clust_mean[t]) if np.isfinite(clust_mean[t])
                  else float(np.mean(probs[t])))
        gnn_mx = (float(clust_max[t]) if np.isfinite(clust_max[t])
                  else float(np.max(probs[t])))
        conf = detection_confidence(gnn_ms, m_phys, coarse_peak)
        alerts.append({
            "id": f"{label}-{t:03d}",
            "timestep_index": t,
            "timestep": ts[t],
            "centre_lat": clat,
            "centre_lon": clon,
            "bbox": [float(x) for x in bboxes[t]],
            "radius_km": round(_radius_km(clat, clon, bboxes[t]), 1),
            "severity": severity,
            "confidence": round(conf, 3),
            "coarse_peak": round(coarse_peak, 1),
            "downscaled_peak": round(ds_peak, 1),
            "peak_gain": round(gain, 3),
            "n_ensemble": members,
            "severity_prob_mean": round(sev_p, 4),
            "gnn_mean_score": round(gnn_ms, 3),
            "gnn_max_score": round(gnn_mx, 3),
        })
        if (t + 1) % 4 == 0:
            print(f"  t{t:02d} peak-gain {gain:.3f} sev-prob {sev_p:.3f} "
                  f"-> {severity}")
    med_gain = float(np.median(gains)) if gains else float("nan")
    np.savez(out_dir / "live_det.npz", centres=cores, miss=miss,
             probs=probs, cluster_centres=centres, gains=np.asarray(gains))
    json.dump(alerts, open(out_dir / "live_alerts.json", "w"), indent=2)
    print(f"  median peak-gain {med_gain:.3f} across {len(det_t)} "
          f"timesteps; alerts -> {out_dir.name}/live_alerts.json")

    # ---- [5] temporal smoothing + forecast (no truth to compare) --------
    forecast_boxes = []
    if do_track and len(det_t) >= warmup + 1:
        tracker_ck = _load_weights(
            tracker_path or model_dir("temporal") /
            "temporal_tracker_era5_ft.pt", "temporal tracker")
        tracker = tp.TemporalTracker(in_dim=int(tracker_ck["in_dim"]),
                                     d_model=int(tracker_ck["d_model"]),
                                     nhead=int(tracker_ck["nhead"]),
                                     n_layers=int(tracker_ck["n_layers"]))
        tracker.load_state_dict(tracker_ck["state"])
        scores = np.clip(probs.max(axis=1) * (1.0 - miss), 0.0, 1.0).astype(
            np.float32)
        det_filled = _fill_track(cores)
        pred = tp.rollout_forecast(tracker, det_filled, miss, scores,
                                   warmup=warmup, device=device)
        for t in range(warmup, T):
            forecast_boxes.append({
                "kind": "forecast",
                "timestep_index": t,
                "time": ts[t],
                "centre": {"lat": round(float(pred[t, 0]), 3),
                           "lon": round(float(pred[t, 1]), 3)},
                "lead_hours": (t - warmup + 1) * 6,
                "level_hPa": 850,
                "error_km": None,
            })
        json.dump(forecast_boxes, open(out_dir / "live_forecast.json", "w"),
                  indent=2)
        print(f"  temporal: smoothed {warmup} steps + {T - warmup} "
              f"forecast boxes -> {out_dir.name}/live_forecast.json")
    else:
        print("  temporal: skipped (insufficient detected timesteps)")

    summary = {
        "run_ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "out_dir": str(out_dir),
        "source": str(era5_path),
        "timesteps": T,
        "detected": len(det_t),
        "median_peak_gain": round(med_gain, 3),
        "n_alerts": len(alerts),
        "n_forecast_boxes": len(forecast_boxes),
        "weights": {"gnn": str(gnn_path or model_dir("gnn") /
                               "gnn_anomaly_era5.pt"),
                    "diffusion": str(diff_path or model_dir("diffusion") /
                                     "downscale_unet_era5.pt"),
                    "tracker": str(tracker_path or model_dir("temporal") /
                                   "temporal_tracker_era5_ft.pt")},
    }
    json.dump(summary, open(out_dir / "live_report.json", "w"), indent=2)
    return summary


def new_run_dir(label: str = "live") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = OUTPUTS / f"live_run_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d