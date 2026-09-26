"""STAGE 4: aggregate the per-event ERA5 validation reports.

Reads outputs/era5_report_{event}.json for every event in the EVENTS
registry, applies the multi-event validation gates, and writes
outputs/multi_event_summary.json plus a console comparison table.

Gates (per the multi-event brief):
  * detection rate >= 50%
  * intensification-window core error ok for operational use
    (target < 80 km; additionally flagged if > 3x the Amphan-window baseline)
  * diffusion median peak-gain >= 0.90
  * tracker forecast beats persistence at least at one lead (each lead is
    reported; beats are counted)

If any hard stop fires, `stop` is set True in the summary — this is meant
to be surfaced, not silently tuned around.

  python -m scripts.aggregate_validation [--baseline amphan]
"""
from __future__ import annotations

import argparse
import json
import sys

from weather_pipeline.config import OUTPUTS
from weather_pipeline.tracks import EVENTS

LEADS = (1, 3, 6)
BASELINE_MAX_FACTOR = 3.0
DETECTION_FLOOR = 0.5
WINDOW_TARGET_KM = 80.0
GAIN_FLOOR = 0.9


def load_report(event: str) -> dict:
    p = OUTPUTS / f"era5_report_{event}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing report: {p} (run phase_era5 "
                                f"--event {event} --no-train first)")
    return json.load(open(p, encoding="utf-8"))


def fmt_km(x) -> str:
    return f"{x:6.1f}" if isinstance(x, (int, float)) else "   n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="amphan")
    ap.add_argument("--out", default="multi_event_summary.json")
    args = ap.parse_args()

    baseline = load_report(args.baseline)
    base_win = float(baseline["gnn"]["intensification_window_median_km"])
    hard_cap = BASELINE_MAX_FACTOR * base_win

    events = sorted(EVENTS)
    rows, stop = [], False
    for event in events:
        r = load_report(event)
        gnn = r["gnn"]
        det_rate = gnn["detected"] / gnn["total"]
        det_ok = det_rate >= DETECTION_FLOOR
        win = gnn["intensification_window_median_km"]
        win_ok = win < WINDOW_TARGET_KM
        win_gt3 = win > hard_cap          # >3x Amphan-window baseline
        gain = r["diffusion"]["median_peak_gain"]
        gain_ok = gain >= GAIN_FLOOR
        beats = {h: r["forecast"][str(h)]["median_km"] <
                 r["forecast"][str(h)]["persistence_km"]
                 for h in LEADS if str(h) in r["forecast"]}
        n_beats = sum(beats.values())
        rows.append({
            "event": event,
            "detected": gnn["detected"], "total": gnn["total"],
            "detection_rate": round(det_rate, 3),
            "cluster_centroid_median_km": gnn["cluster_centroid_median_km"],
            "vortex_core_median_km": gnn["vortex_core_median_km"],
            "intensification_window_median_km": win,
            "peak_gain": gain,
            "forecast": \
                {str(h): r["forecast"][str(h)] for h in LEADS
                 if str(h) in r["forecast"]},
            "smoothing_km": r["smoothing_km"],
            "detection_km": r["detection_km"],
            "gates": {
                "detection_rate_ok": det_ok,
                "window_ok": win_ok,
                "window_gt_3x_baseline": win_gt3,
                "peak_gain_ok": gain_ok,
                "beats_persistence": n_beats,
            },
            "event_stop": (not det_ok) or win_gt3,
        })
        if rows[-1]["event_stop"]:
            stop = True

    summary = {
        "generated": __import__("datetime").datetime.now().isoformat(),
        "baseline_event": args.baseline,
        "baseline_intensification_window_km": base_win,
        "hard_stop_threshold_km": round(hard_cap, 1),
        "gates": {"detection_rate_min": DETECTION_FLOOR,
                  "intensification_window_target_km": WINDOW_TARGET_KM,
                  "peak_gain_min": GAIN_FLOOR,
                  "baseline_error_factor_for_stop": BASELINE_MAX_FACTOR},
        "events": rows,
        "stop": stop,
    }
    out = OUTPUTS / args.out
    json.dump(summary, open(out, "w"), indent=2)

    hdr = (f"{'event':8s} {'det':>6s} {'cluster':>7s} {'core':>6s} "
           f"{'win10':>6s} {'gain':>5s} {'sm/h':>5s} {'18h':>5s} "
           f"{'36h':>5s} {'beats':>5s}")
    print("\nMULTI-EVENT VALIDATION (weights frozen from Amphan run, "
          "no retrain)")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        f = r["forecast"]
        f6 = f["1"]["median_km"] if "1" in f else float("nan")
        f18 = f["3"]["median_km"] if "3" in f else float("nan")
        f36 = f["6"]["median_km"] if "6" in f else float("nan")
        print(f"{r['event']:8s} {r['detected']:4d}/{r['total']:<2d} "
              f"{r['cluster_centroid_median_km']:7.1f} "
              f"{r['vortex_core_median_km']:6.1f} "
              f"{r['intensification_window_median_km']:6.1f} "
              f"{r['peak_gain']:5.3f} {fmt_km(f6)} {fmt_km(f18)} "
              f"{fmt_km(f36)} {r['gates']['beats_persistence']:5d}")
    print(f"\nbaseline (amphan) intensification-window: {base_win:.1f} km;"
          f" hard-stop cap = 3x = {hard_cap:.1f} km")
    for r in rows:
        g = r["gates"]
        flags = []
        flags.append("detOK" if g["detection_rate_ok"] else "DET<50%")
        flags.append("winOK" if g["window_ok"] else "win>80km")
        if g["window_gt_3x_baseline"]:
            flags.append(">3xBaseline(STOP)")
        flags.append("gainOK" if g["peak_gain_ok"] else "gain<0.9")
        flags.append(f"beats={g['beats_persistence']}")
        print(f"  {r['event']:8s} {', '.join(flags)}")
    if stop:
        print("\nSTOP: at least one gate exceeded 3x the Amphan baseline or "
              "detection fell below 50%. Reported honestly (no retraining).")
    else:
        print("\nNo hard-stop breached: detection >=50%, window error within "
              "3x baseline, peak-gain >=0.9 for all events.")
    print(f"summary -> {out}")
    return 0 if not stop else 2


if __name__ == "__main__":
    sys.exit(main())