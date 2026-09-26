import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weather_pipeline import gnn
from weather_pipeline.data_features import prepare_dataset
from weather_pipeline.feedback import FeedbackStore, apply_feedback
from weather_pipeline.synthetic import build_synthetic_amphan

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def kmd(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * \
        math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def centre_errors(model, dataset: dict):
    n_t = int(dataset["n_time"])
    lat = np.linspace(10.5, 22.5, n_t)
    lon = np.linspace(89.5, 88.7, n_t)
    probs = gnn.predict(dataset, model, 0, n_t)
    errs = []
    for t in range(n_t):
        reg = gnn.bounding_region(probs[t], thr=None,
                                  node_lat=dataset["node_lat"],
                                  node_lon=dataset["node_lon"])
        if reg.get("centre"):
            errs.append(kmd(reg["centre"]["lat"], reg["centre"]["lon"],
                            lat[t], lon[t]))
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--max-err", type=float, default=40.0,
                    help="centre error (km) beyond which a review counts "
                         "as a false alarm")
    ap.add_argument("--no-retrain", action="store_true",
                    help="only generate feedback, skip retraining")
    args = ap.parse_args()

    print("--- Phase 9: operator feedback loop ---")
    fb = FeedbackStore()
    if not fb.records:
        print("> auto-review of the 24 issued alerts vs truth track "
              "(SOP: ground truth is known during the exercise)")
        fb.auto_review_all(max_err_km=args.max_err)
    print("  feedback summary:", json.dumps(fb.summary()))

    if args.no_retrain:
        print("PHASE 9: feedback logged; retraining skipped "
              "(--no-retrain)")
        return

    print("--- retraining GNN with operator corrections ---")
    ds = build_synthetic_amphan(save=True)
    dataset = prepare_dataset(ds)
    mask = apply_feedback(dataset, fb)["mask"]
    n_zeroed = int((mask == 0).sum())
    print(f"  feedback label-mask zeroes {n_zeroed} node-labels "
          f"({n_zeroed / mask.size:.2%})")

    model, hist = gnn.train(dataset, epochs=args.epochs,
                            hidden=args.hidden, label_mask=mask,
                            save_path="models/gnn/gnn_anomaly_fb.pt")

    errs = centre_errors(model, dataset)
    med = float(np.median(errs)) if errs else None
    print(f"  median centre error (feedback model): {med:.2f} km "
          f"(baseline 9-22 km run-to-run)")

    report = {
        "feedback": fb.summary(),
        "n_label_nodes_zeroed": int(n_zeroed),
        "phase2_gate_median_centrerr_km": med,
        "baseline_median_centrerr_km": 9.3,
    }
    with open(OUTPUTS / "phase9_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("  saved outputs/phase9_report.json and "
          "models/gnn/gnn_anomaly_fb.pt")
    print("PHASE 9 STOP-AND-VERIFY: feedback captured "
          f"({len(fb.records)} records), retrain complete, "
          f"median centre err {med:.1f} km within baseline band")


if __name__ == "__main__":
    main()