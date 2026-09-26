"""PHASE 4 driver: fully-automated GNN -> diffusion end-to-end pipeline.

Usage:
    python -m scripts.phase4_run [--members 24] [--level 850]

Runs the whole chain for every forecast timestep with NO manual
intervention between stages, prints the STOP-AND-VERIFY summary (centre
tracking error + peak preservation), and saves:
    outputs/phase4_alerts.json
    outputs/phase4_report.txt
    outputs/phase4_track_map.png        (GNN hotspots + downscale bboxes)
    outputs/phase4_downscale_t*.png     (coarse vs sharpened for select ts)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from weather_pipeline.config import OUTPUTS
from weather_pipeline.pipeline import ThreatPipeline, write_report
from weather_pipeline import plot as plot_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=16,
                    help="diffusion ensemble size per timestep")
    ap.add_argument("--steps", type=int, default=100,
                    help="denoising steps per member (fast-sampling mode)")
    ap.add_argument("--level", type=int, default=850)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="ignore phase4_cache and recompute")
    args = ap.parse_args()

    pipe = ThreatPipeline(level=args.level, n_steps=args.steps)
    print(f"--- Phase 4 end-to-end: GNN->diffusion, {pipe.n_time} timesteps "
          f"({args.members} members x {args.steps} steps each) ---")
    results = pipe.run(n_members=args.members, start=args.start,
                       end=args.end, force=args.force)

    det = [r for r in results if r["detected"]]
    alerts = [r["alert"] for r in det]
    json.dump(alerts, open(OUTPUTS / "phase4_alerts.json", "w"), indent=2)
    print(f"\n[saved] outputs/phase4_alerts.json ({len(alerts)} alerts)")

    # ground-truth synthetic cyclone track for verification
    track = np.column_stack([
        np.linspace(10.5, 22.5, pipe.n_time),
        np.linspace(89.5, 88.7, pipe.n_time)])
    report = write_report(results, track)
    print("\n" + report + "\n")

    # ---- plots -------------------------------------------------------
    from weather_pipeline.synthetic import build_synthetic_amphan
    build_synthetic_amphan(save=True)

    # track map: overlay detected centres + downscale region on GNN probs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from weather_pipeline import gnn as gnn_mod
    probs = gnn_mod.predict(pipe.dataset, pipe.gnn, 0, pipe.n_time,
                            device=pipe.device)
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    im = ax.scatter(pipe.dataset["node_lon"], pipe.dataset["node_lat"],
                    c=probs.mean(axis=0), s=3, cmap="magma",
                    vmin=0, vmax=float(np.percentile(probs, 99)))
    for r in det:
        a = r["alert"]
        ax.plot(a["centre_lon"], a["centre_lat"], "c.",
                ms=6)
        ax.plot([a["bbox"][2], a["bbox"][3]], [a["centre_lat"]] * 2, "c-",
                lw=1, alpha=0.7)
        ax.plot([a["centre_lon"]] * 2, [a["bbox"][0], a["bbox"][1]], "c-",
                lw=1, alpha=0.7)
    ax.plot(track[:, 1], track[:, 0], "w-", lw=1.5, alpha=0.8,
            label="true Amphan track")
    ax.plot(track[:, 1], track[:, 0], "w.", ms=4)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title("Phase 4: GNN hotspot mean prob + downscale bboxes")
    fig.colorbar(im, ax=ax, label="mean GNN prob")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "phase4_track_map.png", dpi=120)
    plt.close(fig)

    for t in (8, 16, 22):
        for r in det:
            if r["alert"]["timestep_index"] == t:
                sub = r["crop"]
                lat, lon = sub["lat"], sub["lon"]
                dmean = r["members"].mean(axis=0)[0]
                plot_mod.plot_grid_field(
                    lat, lon, sub["coarse"],
                    f"coarse (v.{args.level}) t={t} [GNN-centred]",
                    str(OUTPUTS / f"phase4_coarse_t{t}.png"))
                plot_mod.plot_grid_field(
                    lat, lon, dmean,
                    f"downscaled ensemble-mean t={t}",
                    str(OUTPUTS / f"phase4_downscale_t{t}.png"))
                plot_mod.plot_grid_field(
                    lat, lon, r["severity"]["severity_prob"],
                    f"severity probability t={t}",
                    str(OUTPUTS / f"phase4_severity_t{t}.png"),
                    vmin=0, vmax=1, cmap="YlOrRd")
    print("[saved] outputs/phase4_track_map.png, "
          "outputs/phase4_{coarse,downscale,severity}_t*.png")


if __name__ == "__main__":
    main()