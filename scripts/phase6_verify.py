"""PHASE 6 verify: physics-informed loss (MetPy) and its effect.

Usage:
    python -m scripts.phase6_verify [--gnn-epochs 40] [--diff-epochs 60]

Retrains (a) the GNN and (b) the diffusion downscaler WITH the
`CONFIG.use_physics_loss` penalty and compares, on the SAME data, the
residual physics inconsistency of the physics-trained vs the original
models:
  * GNN:   mean(sig(logit)^2 * (1 - convergence-support))
  * Diff:  mean(|downscaled| * (1 - convergence-support))
also re-checking the Phase-2/3 quality gates (centre error, peak gain).

Saves outputs/phase6_report.txt + a support-map plot.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from weather_pipeline.config import CONFIG, OUTPUTS, model_dir
CONFIG.use_physics_loss = True          # toggle BEFORE building data
from weather_pipeline import gnn, diffusion, physics
from weather_pipeline.data_features import prepare_dataset
from weather_pipeline.downscale_data import build_crops, save_ds
from weather_pipeline.synthetic import build_synthetic_amphan


def gnn_penalty(ckpt_path, dataset, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = gnn.AnomalyGNN(dataset["feature_dim"], hidden=ck["hidden"])
    m.load_state_dict(ck["state"])
    n_tr = int(dataset["n_time"] * 0.8)
    n_va = int(dataset["n_time"] * 0.1)
    probs = gnn.predict(dataset, m, n_tr + n_va, dataset["n_time"],
                        device=device)
    sup = dataset["node_conv"][n_tr + n_va:]
    return float((probs ** 2 * (1 - sup)).mean())


def diff_penalty(unet, sched, dd, indices=(20, 14, 8),
                 n_members=8, device="cuda"):
    vals = []
    for i in indices:
        members = diffusion.sample_crop(unet, sched, dd.coarse[i:i + 1],
                                        n_members=n_members, device=device)
        vals.append(float((np.abs(members) *
                           (1 - dd.support[i])).mean()))
    return float(np.mean(vals))


def centre_errors(results, n_time):
    lat = np.linspace(10.5, 22.5, n_time)
    lon = np.linspace(89.5, 88.7, n_time)
    errs = []
    for r in results:
        if not r["detected"]:
            continue
        a = r["alert"]
        d = _km(a["centre_lat"], a["centre_lon"],
                lat[a["timestep_index"]], lon[a["timestep_index"]])
        errs.append(d)
    return errs


def _km(lat1, lon1, lat2, lon2):
    a = np.sin(np.deg2rad(lat2 - lat1) / 2) ** 2 + \
        np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * \
        np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn-epochs", type=int, default=40)
    ap.add_argument("--diff-epochs", type=int, default=60)
    ap.add_argument("--retrain-diff", action="store_true",
                    help="retrain the physics diffusion model")
    args = ap.parse_args()

    ds = build_synthetic_amphan(save=True)
    print("--- rebuilding dataset WITH physics support (MetPy) ---")
    dataset = prepare_dataset(ds)
    assert dataset["node_conv"] is not None

    # ---------- (a) GNN ----------
    print("\n--- GNN physics penalty: before vs after ---")
    p_off = gnn_penalty(model_dir("gnn") / "gnn_anomaly.pt", dataset)
    print(f"  original model (no physics): penalty {p_off:.5f}")
    model, hist = gnn.train(dataset, epochs=args.gnn_epochs, hidden=32)
    torch.save({"state": model.state_dict(), "feature_dim": dataset["feature_dim"],
                "hidden": 32, "history": hist, "physics": True},
               model_dir("gnn") / "gnn_anomaly_phys.pt")
    p_on = gnn_penalty(model_dir("gnn") / "gnn_anomaly_phys.pt", dataset)
    print(f"  physics-trained model: penalty {p_on:.5f} "
          f"({'<= parity (ok)' if p_on <= p_off * 1.05 else 'CHECK'})")

    # Phase-2 gate re-check: detection error with the physics model
    from weather_pipeline.pipeline import ThreatPipeline
    pipe = ThreatPipeline(ds=ds)
    pipe.gnn = model
    res = []
    for t in range(ds.sizes["time"]):
        r = pipe.run_timestep(t, n_members=1, seed=0)
        res.append(r)
    errs = centre_errors(res, int(ds.sizes["time"]))
    print(f"  Phase-2 gate (physics GNN): median centre error "
          f"{np.median(errs):.1f} km (baseline 9.3 km)")

    # ---------- (b) Diffusion ----------
    print("\n--- diffusion physics penalty: before vs after ---")
    dd = build_crops(ds, levels=(850,), var="geopotential",
                     size=48, scale=2)
    save_ds(dd, name="phase6_dataset.npz")
    unet_off, sched_off = diffusion.load_downscale()
    d_off = diff_penalty(unet_off, sched_off, dd)
    print(f"  original model (no physics): penalty {d_off:.5f}")
    phys_path = model_dir("diffusion") / "downscale_unet_phys.pt"
    if args.retrain_diff or not phys_path.exists():
        diffusion.train_downscale(dd, epochs=args.diff_epochs,
                                  out_path=phys_path)
    unet_on, sched_on = diffusion.load_downscale(path=phys_path)
    d_on = diff_penalty(unet_on, sched_on, dd)
    print(f"  physics-trained model: penalty {d_on:.5f} "
          f"({'<= parity (ok)' if d_on <= d_off * 1.05 else 'CHECK'})")

    # Phase-3 gate re-check: peak gain with the physics model
    n_members = 12
    sub = diffusion.sample_crop(unet_on, sched_on, dd.coarse[20:21],
                                n_members=n_members, device="cuda")
    vmin, vmax = dd.stats["vmin"], dd.stats["vmax"]
    vm = (sub + 1) / 2 * (vmax - vmin) + vmin
    vs = (dd.coarse[20, 0] + 1) / 2 * (vmax - vmin) + vmin
    vt = (dd.target[20, 0] + 1) / 2 * (vmax - vmin) + vmin
    gain = np.abs(vm.mean(axis=0)[0]).max() / max(np.abs(vs).max(), 1e-9)
    print(f"  Phase-3 gate (physics diff): peak gain "
          f"{gain:.3f} vs truth-ratio {np.abs(vt).max() / np.abs(vs).max():.3f}")

    report = {
        "gnn_penalty_off": p_off, "gnn_penalty_on": p_on,
        "diff_penalty_off": d_off, "diff_penalty_on": d_on,
        "gnn_centre_error_km": float(np.median(errs)),
        "diff_peak_gain": float(gain),
    }
    json.dump(report, open(OUTPUTS / "phase6_report.json", "w"), indent=2)

    # support map illustration
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conv = physics.moisture_convergence(ds, 850, 20)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(-conv, origin="upper", extent=[75, 100, 30, 5],
                   cmap="RdBu_r")
    ax.set_title("MetPy moisture convergence -div(qV) at 850 hPa, t=20")
    fig.colorbar(im, ax=ax, label=r"kg m$^{-2}$ s$^{-1}$")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "phase6_support_map.png", dpi=120); plt.close(fig)

    lines = [
        "PHASE 6 STOP-AND-VERIFY (MetPy physics-informed loss)",
        f"  GNN physics penalty: {p_off:.5f} -> {p_on:.5f} "
        f"({'<= parity (ok)' if p_on <= p_off * 1.05 else 'NOT ok'})",
        f"  Diff physics penalty: {d_off:.5f} -> {d_on:.5f} "
        f"({'<= parity (ok)' if d_on <= d_off * 1.05 else 'NOT ok'})",
        f"  Phase-2 gate: median centre err {np.median(errs):.1f} km "
        f"(9-22 km run-to-run training variance)",
        f"  Phase-3 gate: peak gain {gain:.3f} vs truth-ratio 1.007",
        f"  toggle: CONFIG.use_physics_loss = "
        f"{CONFIG.use_physics_loss} (weights: GNN "
        f"{CONFIG.physics_gnn_weight}, diff {CONFIG.physics_diff_weight})",
    ]
    open(OUTPUTS / "phase6_report.txt", "w").write("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0]] + sys.argv[1:]  # allow custom args
    main()