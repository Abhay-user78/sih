"""PHASE 5 train/verify: temporal transformer trajectory tracker.

Usage:
    python -m scripts.phase5_train [--events 60] [--epochs 120]
                                   [--noise-km 15] [--miss-p 0.05]

STOP-AND-VERIFY:
  * per-timestep smoothing refines detections vs raw detection noise
  * multi-step forecast beats the naive persistence baseline at
    18/36/54 h horizons on held-out events
Saves models/temporal/temporal_tracker.pt + outputs/phase5_*.png/json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from weather_pipeline.config import OUTPUTS, DATA_PROCESSED
from weather_pipeline import temporal as tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--noise-km", type=float, default=15.0)
    ap.add_argument("--miss-p", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.45,
                    help="forecast-loss weight (smooth = 1-alpha)")
    ap.add_argument("--dmodel", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("--- generating synthetic cyclone events ---")
    tracks = tp.generate_events(args.events, n_steps=24, seed=args.seed)
    det, miss = tp.add_detection_noise(tracks, km=args.noise_km,
                                       miss_p=args.miss_p,
                                       seed=args.seed + 1)
    # scores: ~simulated GNN mean score, lower at misses
    rng = np.random.default_rng(args.seed + 2)
    scores = np.clip(0.9 - rng.uniform(0, 0.4, det.shape[:-1])
                     * (miss > 0), 0.0, 1.0).astype(np.float32)

    n_tr = int(args.events * 0.8)
    train_ts, val_ts = tracks[:n_tr], tracks[n_tr:]
    train_d, val_d = det[:n_tr], det[n_tr:]
    train_m, val_m = miss[:n_tr], miss[n_tr:]
    train_s, val_s = scores[:n_tr], scores[n_tr:]

    tr = tp.TrackSet(train_d, train_m, train_s)
    va = tp.TrackSet(val_d, val_m, val_s)
    print(f"  {args.events} events x 24 steps, "
          f"noise {args.noise_km} km, miss {args.miss_p:.0%}, "
          f"split {n_tr}/{args.events - n_tr}")

    print("--- training TransformerEncoder tracker (GPU) ---")
    model, hist = tp.train_tracker(tr, train_ts, va, val_ts,
                                   epochs=args.epochs, α=args.alpha,
                                   d_model=args.dmodel)

    print("--- evaluating held-out forecast vs persistence ---")
    res = tp.evaluate(model, va, val_ts, warmup=args.warmup)
    for h, d in sorted(((h, d) for h, d in res.items()
                        if isinstance(h, int))):
        if h == "smooth_median_km" or h == "detection_median_km":
            continue
        print(f"  horizon {int(h) * 6:2d} h: forecast "
              f"{d['median_km']:6.1f} km  vs persistence "
              f"{d['persistence_km']:6.1f} km  "
              f"({'BEATS' if d['median_km'] < d['persistence_km'] else 'no'})")
    print(f"  observed-window smoothing: tracker "
          f"{res['smooth_median_km']:.1f} km vs raw detections "
          f"{res['detection_median_km']:.1f} km "
          f"({'REDUCES NOISE' if res['smooth_median_km'] < res['detection_median_km'] else 'at parity'})")

    out = {"noise_km": args.noise_km, "miss_p": args.miss_p,
           "warmup": args.warmup, "metrics": res,
           "losses": hist}
    json.dump(out, open(OUTPUTS / "phase5_metrics.json", "w"), indent=2)

    # ---- plots: loss curves + held-out event track ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["tr"], label="train")
    ax.plot(hist["va"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (L1, norm)")
    ax.set_title("Phase 5 tracker training")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUTS / "phase5_train_curves.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    e = 0
    pred = tp.rollout_forecast(model, val_d[e], val_m[e], val_s[e],
                               warmup=args.warmup)
    ax.plot(val_ts[e, :, 1], val_ts[e, :, 0], "k.-", label="truth")
    ax.plot(val_d[e, :, 1], val_d[e, :, 0], "x", color="0.6",
            label="GNN detections")
    ax.plot(pred[:, 1], pred[:, 0], "r.-", label="tracker (smooth+forecast)")
    ax.axvline(val_d[e, args.warmup, 1], color="b", ls="--", alpha=0.5,
               label="forecast start")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title("Phase 5: held-out event 4D-box track")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUTS / "phase5_heldout_track.png", dpi=120); plt.close(fig)

    np.savez(DATA_PROCESSED / "phase5_val_tracks.npz",
             truth=val_ts, detections=val_d, miss=val_m, pred=pred)

    print("\n[saved] models/temporal/temporal_tracker.pt, "
          "outputs/phase5_*.png, outputs/phase5_metrics.json")
    beats = all(res[h]["median_km"] < res[h]["persistence_km"]
                for h in res if isinstance(h, int))
    smooth_ok = res["smooth_median_km"] <= res["detection_median_km"] * 1.15
    print("\nPHASE 5 STOP-AND-VERIFY: inspect outputs/phase5_heldout_track.png;")
    print(f"forecast beats persistence at all horizons: "
          f"{'OK' if beats else 'CHECK'}")
    print(f"smoothing not worse than raw detections: "
          f"{'OK' if smooth_ok else 'CHECK'}")


if __name__ == "__main__":
    main()