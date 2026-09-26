"""PHASE 2 detect/verify: score test timesteps, extract bounding regions,
save alerts + plots.

Usage:
    python -m scripts.phase2_detect

Compares, timestep by timestep, the detected bounding-region centre against
the true cyclone centre (synthetic placeholder) and prints the error.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from weather_pipeline.config import DATA_PROCESSED, OUTPUTS, model_dir
from weather_pipeline import gnn
from weather_pipeline.data_features import prepare_dataset
from weather_pipeline.synthetic import build_synthetic_amphan
from weather_pipeline import plot as plot_mod


def main():
    ds = build_synthetic_amphan(save=True)
    print("--- rebuilding dataset (features) ---")
    dataset = prepare_dataset(ds)

    ckpt = torch.load(model_dir("gnn") / "gnn_anomaly.pt",
                      map_location="cpu", weights_only=False)
    model = gnn.AnomalyGNN(dataset["feature_dim"],
                           hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state"])

    print("--- scoring timesteps (all 24) ---")
    probs = gnn.predict(dataset, model, 0, dataset["n_time"])
    np.save(DATA_PROCESSED / "phase2_probs.npy", probs)

    # ground truth cyclone centre per timestep (synthetic)
    cy_lat = np.linspace(10.5, 22.5, dataset["n_time"])
    cy_lon = np.linspace(89.5, 88.7, dataset["n_time"])

    results = []
    for t in range(dataset["n_time"]):
        reg = gnn.bounding_region(probs[t], thr=None,
                                  node_lat=dataset["node_lat"],
                                  node_lon=dataset["node_lon"])
        rec = {"timestep": int(t),
               "true_centre": {"lat": float(cy_lat[t]),
                               "lon": float(cy_lon[t])},
               "bbox": reg.get("bbox"),
               "centre": reg.get("centre"),
               "n_hits": reg.get("n_hits"),
               "thr": reg.get("thr")}
        if reg.get("centre") and reg.get("bbox"):
            d = gnn_xy_delta(reg["centre"], rec["true_centre"])
            rec["centre_error_km"] = round(float(d), 1)
        results.append(rec)
        print(f"  t={t:2d} hits={rec['n_hits']:4d} bbox={rec['bbox']} "
              f"err_km={rec.get('centre_error_km', '-')}")

    # detection quality over time
    ok = [r for r in results if r.get("centre")]
    hits = len(ok)
    covered = int(np.mean([bool(r.get("centre")) and r["centre"] is not None
                           for r in results]) * 100)
    errs = [r["centre_error_km"] for r in ok if "centre_error_km" in r]
    print(f"\n  timesteps with a detected region: {hits}/{len(results)} "
          f"({covered}%)")
    if errs:
        print(f"  median centre error vs truth: {np.median(errs):.1f} km")

    json.dump(results, open(OUTPUTS / "phase2_alerts.json", "w"), indent=2)
    print("\n[saved] outputs/phase2_alerts.json, "
          "data/processed/phase2_probs.npy")

    # maps for a few timesteps
    for t in (10, 16, 20, 23):
        plot_mod.plot_mesh(dataset["node_lat"], dataset["node_lon"], probs[t],
                           f"GNN anomaly prob t={t} (Amphan track)",
                           str(OUTPUTS / f"phase2_prob_t{t:02d}.png"),
                           cmap="magma")
    print("\nPHASE 2 STOP-AND-VERIFY: inspect outputs/phase2_prob_*.png; the")
    print("hotspot must sit on the cyclone track (89.5E->88.7E, lat 10->22).")


def gnn_xy_delta(p1: dict, p2: dict) -> float:
    p1 = (p1["lat"], p1["lon"]); p2 = (p2["lat"], p2["lon"])
    lat1, lon1 = np.deg2rad(p1); lat2, lon2 = np.deg2rad(p2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * \
        np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


if __name__ == "__main__":
    main()