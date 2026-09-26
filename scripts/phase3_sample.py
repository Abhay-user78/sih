"""PHASE 3 sample/verify: run the stochastic ensemble downscaling and check
peak-value preservation.

Usage:
    python -m scripts.phase3_sample [--index 20] [--members 24]

STOP-AND-VERIFY: the downscaled (upsampled-sharpened) output must keep the
coarse input's extreme peak (or move it toward the true sharp peak), NOT
smooth it away. Prints coarse max vs ensemble-mean max and per-member
distribution, saves plots.
"""
from __future__ import annotations

import argparse

import numpy as np

from weather_pipeline.config import OUTPUTS
from weather_pipeline.downscale_data import load_ds, inv_normalise
from weather_pipeline import diffusion
from weather_pipeline import plot as plot_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=20)
    ap.add_argument("--members", type=int, default=24)
    args = ap.parse_args()

    dd = load_ds()
    unet, sched = diffusion.load_downscale()

    print(f"--- sampling {args.members} members for crop {args.index} ---")
    res = diffusion.sample_ensemble(dd, unet, sched, args.index,
                                    n_members=args.members)
    members = res["members"]
    sev = diffusion.severity_from_ensemble(members, res["cond"])

    # inverse-normalise to physical units
    vr = inv_normalise(res["cond"], dd.stats)
    vt = inv_normalise(res["target"], dd.stats)
    vm = inv_normalise(members, dd.stats)
    vmu = vm.mean(axis=0)[0]

    coarse = vr[0]
    peak_coarse = float(np.abs(coarse).max())
    peak_true = float(np.abs(vt[0]).max())
    peak_mean = float(np.abs(vmu).max())
    peak_members = np.abs(vm[:, 0]).max(axis=(1, 2))

    print(f"\n  peak |value|: coarse {peak_coarse:.1f}  "
          f"true {peak_true:.1f}")
    print(f"  peak |ensemble mean|: {peak_mean:.1f}  "
          f"({100 * peak_mean / max(peak_true, 1e-9):.0f}% of truth)")
    print(f"  per-member peaks: min {peak_members.min():.1f}  "
          f"median {np.median(peak_members):.1f}  "
          f"max {peak_members.max():.1f}")
    print(f"  member peaks that reach >= coarse peak: "
          f"{(peak_members >= peak_coarse).mean() * 100:.0f}%")
    print(f"  ensemble 'sharpens the peak' if peak_mean > peak_coarse: "
          f"{'YES' if peak_mean > peak_coarse else 'NO'}")

    print(f"\n  severity probability map: max {sev['severity_prob'].max():.2f}  "
          f"area>0.5 {np.mean(sev['severity_prob'] > 0.5) * 100:.1f}%")

    for tag, arr in (("coarse", coarse), ("truth", vt[0]),
                     ("ensemble_mean", vmu),
                     ("severity", sev["severity_prob"])):
        plot_mod.plot_grid_field(
            dd.lon, dd.lat, arr,
            f"[{tag} (downscale)] idx={args.index}",
            str(OUTPUTS / f"phase3_{tag}.png"))

    out = OUTPUTS / "phase3_ensemble.npz"
    np.savez(out, members=vm, coarse=coarse, truth=vt[0],
             severity=sev["severity_prob"], stat_vmin=dd.stats["vmin"],
             stat_vmax=dd.stats["vmax"])
    print(f"\n[saved] outputs/phase3_*.png, {out}")


if __name__ == "__main__":
    main()