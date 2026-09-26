"""PHASE 2 train: GNN anomaly detector on the Amphan (synthetic placeholder)
test case.

Usage:
    python -m scripts.phase2_train [--epochs 30] [--hidden 64]

CPU/GPU is auto-selected (RTX 4050 available).
STOP AND VERIFY fallback check: after training, run phase2_detect to confirm
the flagged region tracks the cyclone.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from weather_pipeline.config import CONFIG, DATA_PROCESSED, OUTPUTS, model_dir
from weather_pipeline import gnn
from weather_pipeline.data_features import prepare_dataset
from weather_pipeline.synthetic import build_synthetic_amphan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--split", type=str, default="0.8/0.1/0.1")
    args = ap.parse_args()
    split = tuple(float(x) for x in args.split.split("/"))
    assert sum(split) == 1.0

    print(f"--- loading data (source={CONFIG.data_source}) ---")
    ds = build_synthetic_amphan(save=True)
    print("--- building mesh node features + labels ---")
    dataset = prepare_dataset(ds)
    print(f"  {dataset['n_time']} timesteps x {dataset['n_nodes']} nodes x "
          f"{dataset['feature_dim']} features")
    # cache dataset for the detect script (no rebuild)
    np.savez(DATA_PROCESSED / "phase2_dataset.npz",
             features=dataset["features"], **{f"k_{k}": np.asarray(v)
             for k, v in dataset.items()})
    json.dump({k: (v if isinstance(v, (int, float, list)) else str(type(v)))
               for k, v in dataset.items() if not hasattr(v, "dtype") or
               v.ndim == 0},
              open(model_dir("gnn") / "dataset_meta.json", "w"), indent=2)

    print("--- training GNN ---")
    model, history = gnn.train(dataset, split=split, epochs=args.epochs,
                               hidden=args.hidden)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.legend(); plt.grid(alpha=0.3)
    plt.title("GNN anomaly detector")
    p = OUTPUTS / "phase2_train_curves.png"
    plt.savefig(p, dpi=110, bbox_inches="tight")
    print(f"[plot] saved {p}")

    print("\nPHASE 2 STOP-AND-VERIFY: run `python -m scripts.phase2_detect`")
    print("and confirm the bounding region tracks the cyclone.")


if __name__ == "__main__":
    main()