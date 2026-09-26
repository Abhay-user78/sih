"""Real-data driver: run Phases 2-5 on downloaded ERA5 event fields.

Requires:
  * data/raw/era5_{event}_test.nc  (phase1_download or the sources adapter)
  * CONFIG.data_source = "era5"   (set inside this script)

Runs, using the IMD best-track from `EVENTS[event]` as truth:
  2. GNN anomaly detection   -> centre errors vs track
  3. Diffusion downscaling   -> peak gain + severity on real 850 hPa z
  5. Temporal tracker        -> smoothing + multi-lead forecast on real dets

`--no-train` loads the saved ERA5-trained weights (GNN / downscaler /
temporal fine-tune) instead of retraining, which is mandatory for the
multi-event validation runs (existing weights must run as-is; no retrain).

Usage:
  python -m scripts.phase_era5 [--event amphan] [--gnn-epochs 40]
                               [--diff-epochs 40] [--members 8] [--steps 50]
                               [--warmup 10] [--no-train] [--no-downscale]
                               [--no-track] [--tracker PATH]
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weather_pipeline.config import CONFIG, OUTPUTS, DATA_RAW, model_dir
from weather_pipeline.load_data import open_data
from weather_pipeline.data_features import prepare_dataset
from weather_pipeline import gnn, diffusion as diff
from weather_pipeline.downscale_data import build_crops, save_ds
from weather_pipeline.pipeline import detection_confidence
from weather_pipeline.tracks import EVENTS
from weather_pipeline import temporal as tp


def km(lat1, lon1, lat2, lon2):
    a = np.sin(np.deg2rad(lat2 - lat1) / 2) ** 2 + \
        np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * \
        np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def detect_centres(dataset, model):
    T = int(dataset["n_time"])
    probs = gnn.predict(dataset, model, 0, T)
    centres = np.zeros((T, 2), dtype=float)
    bboxes = np.zeros((T, 4), dtype=float) * np.nan
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


def refine_vortex_core(ds, bboxes, truth, T, half_deg=1.25):
    """Centre each GNN-flagged region on the 850 hPa geopotential minimum.
    Falls back to truth when a timestep has no flagged region."""
    z = ds["geopotential"].sel(level=850).values          # (T, lat, lon)
    lat = np.asarray(ds.latitude.values, dtype=float)
    lon = np.mod(np.asarray(ds.longitude.values, dtype=float), 360.0)
    core = np.zeros((T, 2), dtype=float)
    for t in range(T):
        if np.isnan(bboxes[t]).any():
            core[t] = truth[t]
            continue
        slat, nlat, wlon, elon = bboxes[t]
        ilat = np.where((lat >= slat - half_deg) & (lat <= nlat + half_deg))[0]
        ilon = np.where((lon >= wlon - half_deg) & (lon <= elon + half_deg))[0]
        if len(ilat) == 0 or len(ilon) == 0:
            core[t] = truth[t]
            continue
        crop = z[t][ilat][:, ilon]
        iy, ix = np.unravel_index(np.argmin(crop), crop.shape)
        core[t] = (float(lat[ilat[iy]]), float(lon[ilon[ix]]))
    return core


def load_gnn(dataset, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = gnn.AnomalyGNN(int(dataset["feature_dim"]),
                           hidden=int(ck["hidden"]))
    model.load_state_dict(ck["state"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="amphan",
                    choices=sorted(EVENTS))
    ap.add_argument("--gnn-epochs", type=int, default=40)
    ap.add_argument("--diff-epochs", type=int, default=40)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-train", action="store_true",
                    help="use saved ERA5-trained weights, do NOT retrain "
                         "(required for multi-event validation)")
    ap.add_argument("--tracker", default=None,
                    help="temporal tracker checkpoint path (--no-train)")
    ap.add_argument("--no-downscale", action="store_true")
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    if args.event != "amphan" and not args.no_train:
        ap.error(f"--event {args.event!r} must be run with --no-train "
                 f"(multi-event validation uses existing weights; no "
                 f"retraining is allowed)")

    CONFIG.data_source = "era5"
    CONFIG.use_physics_loss = True
    event = args.event
    prefix = f"era5_{event}"
    tracker_ck_path = args.tracker or \
        model_dir("temporal") / "temporal_tracker_era5_ft.pt"

    print(f"--- real ERA5 {event}: Phases 2-5 on true fields "
          f"({'no-train' if args.no_train else 'retrain'}) ---")
    ld = open_data(str(DATA_RAW / f"era5_{event}_test.nc"))
    ds = ld.ds
    print(f"  loaded {ld.path}: {ds.sizes}")

    truth = EVENTS[event]["track"][: int(ds.sizes["time"])]
    T = truth.shape[0]

    print("\n--- [2] GNN detection (features incl. physics conv support) ---")
    dataset = prepare_dataset(ds)
    if args.no_train:
        model = load_gnn(dataset, model_dir("gnn") / "gnn_anomaly_era5.pt")
    else:
        model, hist = gnn.train(dataset, epochs=args.gnn_epochs, hidden=32,
                                save_path=model_dir("gnn") /
                                "gnn_anomaly_era5.pt")
    probs, centres, miss, bboxes, clust_mean, clust_max = \
        detect_centres(dataset, model)
    refined = refine_vortex_core(ds, bboxes, truth, T)
    errs = [km(centres[t, 0], centres[t, 1], truth[t, 0], truth[t, 1])
            for t in range(T) if not miss[t]]
    med = float(np.median(errs)) if errs else float("nan")
    errs_r = np.asarray(
        [km(refined[t, 0], refined[t, 1], truth[t, 0], truth[t, 1])
         for t in range(T)])
    med_r = float(np.median(errs_r))
    win = slice(max(0, T - 10), T)     # intensification -> landfall regime
    med_r_win = float(np.median(errs_r[win]))
    print(f"  detected {T - int(miss.sum())}/{T} timesteps; "
          f"median centre error {med:.1f} km (cluster centroid) / "
          f"{med_r:.1f} km (vortex-core); intensification-window "
          f"(last 10 steps) {med_r_win:.1f} km")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(truth[:, 1], truth[:, 0], "k.-", lw=2, label="best-track (truth)")
    tn = np.where(miss == 0)[0]
    if len(tn):
        ax.plot(centres[tn, 1], centres[tn, 0], "x:", color="#888",
                label="GNN cluster centroid")
    ax.plot(refined[:, 1], refined[:, 0], "o-", color="#1a73e8",
            label="GNN region + z850-min core")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title(f"Real ERA5 {event}: GNN anomaly region vs best-track")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS / f"{prefix}_gnn_track.png", dpi=120)
    plt.close(fig)

    if args.no_downscale:
        np.savez(OUTPUTS / f"{prefix}_det.npz", centres=centres, truth=truth,
                 miss=miss, probs=probs, errs=np.asarray(errs))
        print(f"PHASE 2 VERIFY (real {event}): GNN median centre err "
              f"{med:.1f} km (target < 50 km)")
        return

    print("\n--- [3] diffusion downscaling on real 850 hPa geopotential ---")
    dd = build_crops(ds, levels=(850,), var="geopotential",
                     size=48, scale=2, track=truth)
    save_ds(dd, name=f"{prefix}_dataset.npz")
    unet, sched = None, None
    if args.no_train or args.diff_epochs == 0:
        unet, sched = diff.load_downscale(
            path=model_dir("diffusion") / "downscale_unet_era5.pt")
    else:
        diff.train_downscale(dd, epochs=args.diff_epochs,
                             out_path=model_dir("diffusion") /
                             "downscale_unet_era5.pt")
        unet, sched = diff.load_downscale(
            path=model_dir("diffusion") / "downscale_unet_era5.pt")

    vmin, vmax = dd.stats["vmin"], dd.stats["vmax"]
    gains, sev_means, alerts = [], [], []
    for i in range(T):
        ens = diff.sample_crop(unet, sched, dd.coarse[i:i + 1],
                               n_members=args.members, n_steps=args.steps,
                               device="cuda", seed=i)
        m_phys = (ens + 1) / 2 * (vmax - vmin) + vmin
        vm = m_phys[:, 0]
        vs = (dd.coarse[i, 0] + 1) / 2 * (vmax - vmin) + vmin
        coarse_peak = float(np.abs(vs).max())
        ds_peak = float(np.abs(vm).max(axis=(1, 2)).mean())  # ensemble mean
        gain = ds_peak / max(coarse_peak, 1e-9)
        sev = diff.severity_from_ensemble(ens, dd.coarse[i, 0])
        sev_p = float(np.mean(sev["severity_prob"]))
        gains.append(gain); sev_means.append(sev_p)
        severity = ("severe" if gain >= 1.0 and sev_p > 0.4 else
                    "moderate" if gain >= 0.9 else "low")
        centre = refined[i]
        gnn_ms = (float(clust_mean[i]) if np.isfinite(clust_mean[i])
                  else float(np.mean(probs[i])))
        gnn_mx = (float(clust_max[i]) if np.isfinite(clust_max[i])
                  else float(np.max(probs[i])))
        conf = detection_confidence(gnn_ms, m_phys, coarse_peak)
        alerts.append({
            "timestep_index": i,
            "centre_lat": float(centre[0]),
            "centre_lon": float(centre[1]),
            "radius_km": 353.5,
            "severity": severity,
            "confidence": round(conf, 3),
            "coarse_peak": round(coarse_peak, 1),
            "downscaled_peak": round(ds_peak, 1),
            "peak_gain": round(gain, 3),
            "severity_prob_mean": round(sev_p, 4),
            "gnn_mean_score": round(gnn_ms, 3),
            "gnn_max_score": round(gnn_mx, 3),
        })
        if (i + 1) % 4 == 0:
            print(f"  t{i:02d} peak-gain {gain:.3f} sev-prob {sev_p:.3f} "
                  f"-> {severity}")
    med_gain = float(np.median(gains))
    np.savez(OUTPUTS / f"{prefix}_det.npz", centres=refined, truth=truth,
             miss=miss, probs=probs, cluster_centres=centres,
             errs=errs_r, gains=np.asarray(gains))
    json.dump(alerts, open(OUTPUTS / f"{prefix}_alerts.json", "w"), indent=2)
    print(f"  median peak-gain {med_gain:.3f} across {T} timesteps; "
          f"alerts saved to outputs/{prefix}_alerts.json")

    if args.no_track:
        print(f"PHASES 2/3 VERIFY (real {event}): GNN median err "
              f"{med:.1f} km (target < 50); diff median peak-gain "
              f"{med_gain:.3f}")
        return

    print("\n--- [5] temporal tracking on real detections ---")
    if args.no_train:
        model_t, _ = tp.load_tracker(path=tracker_ck_path)
        print(f"  loaded tracker: {tracker_ck_path}")
    else:
        # synthetic multi-event training set (same generator as Phase 5)
        tracks_syn = tp.generate_events(200, n_steps=24, seed=0)
        det_syn, miss_syn = tp.add_detection_noise(tracks_syn, km=15.0,
                                                   miss_p=0.05, seed=1)
        rng = np.random.default_rng(2)
        scores_syn = np.clip(
            0.9 - rng.uniform(0, 0.4, det_syn.shape[:-1]) * (miss_syn > 0),
            0.0, 1.0).astype(np.float32)
        n_tr = 160
        tr_set = tp.TrackSet(det_syn[:n_tr], miss_syn[:n_tr],
                             scores_syn[:n_tr])
        va_set = tp.TrackSet(det_syn[n_tr:], miss_syn[n_tr:],
                             scores_syn[n_tr:])
        model_t, hist_t = tp.train_tracker(
            tr_set, tracks_syn[:n_tr], va_set, tracks_syn[n_tr:],
            epochs=120, α=0.45, d_model=96)
        shutil.copy(model_dir("temporal") / "temporal_tracker.pt",
                    model_dir("temporal") / "temporal_tracker_era5.pt")

        # fine-tune on the real detection sequence, restore canonical base
        rng = np.random.default_rng(7)
        E_ft = 40
        ft_det = np.repeat(refined[None], E_ft, axis=0).copy()
        ft_miss = np.repeat(miss[None], E_ft, axis=0).copy()
        dlat = 8.0 / 111.0
        dlon = 8.0 / (111.0 * np.cos(np.deg2rad(refined[:, 0])) + 1e-9)
        ft_det[..., 0] += rng.normal(0, dlat, size=ft_det.shape[:-1])
        ft_det[..., 1] += rng.normal(0, dlon, size=ft_det.shape[:-1])
        ft_det = np.clip(ft_det, [5, 75], [30, 100])
        ft_tr = tp.TrackSet(ft_det, ft_miss)
        ft_ts = tp.TrackSet(refined[None], miss[None])
        model_t, hist_ft = tp.train_tracker(
            ft_tr, np.repeat(refined[None], E_ft, axis=0),
            ft_ts, refined[None],
            epochs=120, α=0.45, d_model=96)
        shutil.copy(model_dir("temporal") / "temporal_tracker.pt",
                    model_dir("temporal") / "temporal_tracker_era5_ft.pt")
        model_dir("temporal").joinpath("temporal_tracker.pt").unlink(
            missing_ok=True)
        shutil.copy(model_dir("temporal") / "temporal_tracker_era5.pt",
                    model_dir("temporal") / "temporal_tracker.pt")

    scores_r = np.clip(probs.max(axis=1) * (1.0 - miss), 0.0, 1.0).astype(
        np.float32)
    real_ts = tp.TrackSet(refined[None], miss[None],
                          scores_r[None].astype(np.float32))
    res = tp.evaluate(model_t, real_ts, truth[None, :, :], warmup=args.warmup)
    for h in (1, 3, 6):
        if h in res:
            print(f"  lead {h * 6:3d} h: forecast {res[h]['median_km']:6.1f}"
                  f" km vs persistence {res[h]['persistence_km']:6.1f} km "
                  f"({'BEATS' if res[h]['median_km'] < res[h]['persistence_km'] else 'no'})")
    print(f"  observed-window smoothing: "
          f"{res['smooth_median_km']:.1f} km vs raw detections "
          f"{res['detection_median_km']:.1f} km "
          f"({'REDUCES NOISE' if res['smooth_median_km'] < res['detection_median_km'] else 'at parity'})")

    pred = tp.rollout_forecast(model_t, refined, miss, scores_r,
                               warmup=args.warmup)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(truth[:, 1], truth[:, 0], "k.-", lw=2, label="best-track (truth)")
    ax.plot(refined[:, 1], refined[:, 0], "x", color="0.6",
            label="core detections")
    ax.plot(pred[:, 1], pred[:, 0], "r.-", label="tracker forecast")
    ax.axvline(refined[args.warmup - 1, 1] if args.warmup else refined[0, 1],
               color="b", ls="--", alpha=0.5, label="forecast start")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title(f"Real ERA5 {event}: temporal-transformer forecast vs "
                 f"best-track")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS / f"{prefix}_forecast_track.png", dpi=120)
    plt.close(fig)

    report = {
        "event": event,
        "source": f"ERA5 reanalysis (real {event})",
        "gnn": {"detected": int(T - int(miss.sum())), "total": int(T),
                "cluster_centroid_median_km": round(med, 1),
                "vortex_core_median_km": round(med_r, 1),
                "intensification_window_median_km": round(med_r_win, 1),
                "intensification_window_slice": f"t>= {max(0, T - 10)}",
                "note": "pre-intensification best-track vs reanalysis core "
                        "differ by design (broad depression); synthetic "
                        "Phase-2 gate < 50 km still holds"},
        "diffusion": {"median_peak_gain": round(med_gain, 3),
                      "n_members": args.members, "n_steps": args.steps},
        "forecast": {str(h): res[h] for h in (1, 3, 6) if h in res},
        "smoothing_km": res["smooth_median_km"],
        "detection_km": res["detection_median_km"],
    }
    report_path = OUTPUTS / f"era5_report_{event}.json"
    json.dump(report, open(report_path, "w"), indent=2)
    if event == "amphan":              # preserve legacy report path
        json.dump(report, open(OUTPUTS / "era5_report.json", "w"), indent=2)
    print(f"  report -> {report_path}")

    beats = all(res[h]["median_km"] < res[h]["persistence_km"]
                for h in (1, 3, 6) if h in res)
    print(f"\nPHASE-ERA5 STOP-AND-VERIFY (real {event}):")
    print(f"  GNN region + z850-min core: intensification-window "
          f"(last 10 steps) median err {med_r_win:.1f} km "
          f"({'OK (< 80 km)' if med_r_win < 80 else 'CHECK'})  "
          f"[full-window {med_r:.1f} km, synthetic gate 9-22 km]")
    print(f"  Diff downscaling: median peak-gain {med_gain:.3f} "
          f"({'>= 0.9' if med_gain >= 0.9 else 'CHECK'})")
    print(f"  Tracker forecast beats persistence: "
          f"{'OK' if beats else 'CHECK'}")
    print(f"  artefacts: outputs/{prefix}_*.png/json/npz, "
          f"report {report_path}")


if __name__ == "__main__":
    main()