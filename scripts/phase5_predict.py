"""PHASE 5 predict: run the trained temporal tracker on the Amphan GNN
detection stream and emit dynamic 4D bounding boxes (lat, lon, level,
time).

Usage:
    python -m scripts.phase5_predict [--warmup 6] [--level 850]

Reads the GNN detection centres (recomputed from the synthetic Amphan
sequence), produces a tracked + forecast 4D box per forecast step, saves
outputs/phase5_amphan.json and a track plot.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from weather_pipeline.config import OUTPUTS
from weather_pipeline import gnn, temporal as tp
from weather_pipeline.data_features import prepare_dataset
from weather_pipeline.synthetic import build_synthetic_amphan

LABELS = {"forecast": "f", "observed": "o"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--level", type=int, default=850)
    args = ap.parse_args()

    ds = build_synthetic_amphan(save=True)
    dataset = prepare_dataset(ds)
    ck = torch.load(model_dir_gnn(), map_location="cpu", weights_only=False)
    model = gnn.AnomalyGNN(dataset["feature_dim"], hidden=ck["hidden"])
    model.load_state_dict(ck["state"])
    probs = gnn.predict(dataset, model, 0, dataset["n_time"])

    det, miss = [], []
    for t in range(dataset["n_time"]):
        reg = gnn.bounding_region(probs[t], thr=None,
                                  node_lat=dataset["node_lat"],
                                  node_lon=dataset["node_lon"])
        if reg.get("centre"):
            det.append((reg["centre"]["lat"], reg["centre"]["lon"]))
            miss.append(0.0)
        else:
            det.append(det[-1] if det else (11.0, 90.0))
            miss.append(1.0)
    det = np.array(det, dtype=float)
    miss = np.array(miss, dtype=float)

    tracker, _ = tp.load_tracker()
    pred = tp.rollout_forecast(tracker, det, miss, None,
                               warmup=args.warmup)

    # truth track (synthetic) for verification
    truth = np.column_stack([np.linspace(10.5, 22.5, dataset["n_time"]),
                             np.linspace(89.5, 88.7, dataset["n_time"])])

    boxes = []
    for t in range(dataset["n_time"]):
        j = t - args.warmup
        boxes.append({
            "timestep_index": t,
            "time": str(ds.time.values[t]),
            "kind": "observed" if t < args.warmup else "forecast",
            "lead_hours": max(0, j) * 6,
            "level_hPa": args.level,
            "centre": {"lat": round(float(pred[t, 0]), 3),
                       "lon": round(float(pred[t, 1]), 3)},
            "error_km": round(float(tp._km(pred[t, 0], pred[t, 1],
                                           truth[t, 0], truth[t, 1])), 1),
        })
    json.dump(boxes, open(OUTPUTS / "phase5_amphan.json", "w"), indent=2)

    obs = [b for b in boxes if b["kind"] == "observed"]
    fc = [b for b in boxes if b["lead_hours"] > 0]
    print(f"  4D boxes: {len(obs)} observed + {len(fc)} forecast (48 h out)")
    for lead in (6, 18, 36):
        r = [b["error_km"] for b in boxes if b["lead_hours"] == lead]
        if r:
            print(f"  forecast @ {lead:2d} h: median error {np.median(r):.1f} km")
    print(f"  observed-box err (smoothed) median: "
          f"{np.median([b['error_km'] for b in obs]):.1f} km")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(truth[:, 1], truth[:, 0], "k.-", label="truth")
    ax.plot(det[:, 1], det[:, 0], "x", color="0.6", ms=6,
            label="GNN detections")
    ax.plot(pred[:args.warmup, 1], pred[:args.warmup, 0], "g.-",
            label="tracked (observed)")
    ax.plot(pred[args.warmup:, 1], pred[args.warmup:, 0], "r.-",
            label="forecast (48 h)")
    ax.axvline(det[args.warmup, 1], color="b", ls="--", alpha=0.5)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title("Phase 5: Amphan dynamic 4D-box track + forecast")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUTS / "phase5_amphan_track.png", dpi=120); plt.close(fig)

    print("\n[saved] outputs/phase5_amphan.json, "
          "outputs/phase5_amphan_track.png")
    print("\nPHASE 5 STOP-AND-VERIFY: the forecast boxes should stay within")
    print("~50-150 km of the true track as it recurves north / makes landfall.")


def model_dir_gnn():
    from weather_pipeline.config import model_dir
    return model_dir("gnn") / "gnn_anomaly.pt"


if __name__ == "__main__":
    main()