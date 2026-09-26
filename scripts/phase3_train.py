"""PHASE 3 train: conditional diffusion downscaling model.

Usage:
    python -m scripts.phase3_train [--epochs 40] [--size 48] [--scale 2]

Builds (coarse, target) crop pairs from the synthetic (or ERA5-shaped)
dataset and trains the DDPM. STOP-AND-VERIFY via phase3_sample.
"""
from __future__ import annotations

import argparse

from weather_pipeline.downscale_data import build_crops, save_ds
from weather_pipeline import diffusion
from weather_pipeline.synthetic import build_synthetic_amphan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--levels", type=str, default="850")
    ap.add_argument("--variable", default="geopotential")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    ds = build_synthetic_amphan(save=True)
    print(f"--- building crop pairs ({args.size}x{args.size}, scale "
          f"{args.scale}) ---")
    dd = build_crops(ds, levels=levels, var=args.variable,
                     size=args.size, scale=args.scale)
    print(f"  {dd.target.shape[0]} samples from {ds.sizes['time']} timesteps")
    print(f"  target range [{dd.stats['vmin']:.1f}, {dd.stats['vmax']:.1f}]")
    save_ds(dd)

    print("--- training diffusion downscaler (GPU) ---")
    diffusion.train_downscale(dd, epochs=args.epochs)

    print("\nPHASE 3 STOP-AND-VERIFY: run `python -m scripts.phase3_sample`")
    print("and compare peak (max |value|) before vs after downscaling.")


if __name__ == "__main__":
    main()