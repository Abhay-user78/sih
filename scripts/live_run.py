"""STAGE 1: run the pipeline on LIVE (non-historical) ERA5T data.

Pulls the most recent available ERA5T window (no hardcoded dates, no
best-track truth) and runs GNN detection -> diffusion downscaling ->
temporal tracking, writing whatever alerts/forecasts the models produce
to outputs/live_run_{timestamp}/ exactly as in real operation.

  python -m scripts.live_run [--n-days 5] [--members 8] [--steps 50]
                             [--warmup 10] [--area 30 75 5 100]
                             [--refresh] [--no-track]
"""
from __future__ import annotations

import argparse
import sys

from weather_pipeline.config import OUTPUTS
from weather_pipeline.download import download_live, cds_configured
from weather_pipeline.operational import run_live_cycle, new_run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-days", type=int, default=5,
                    help="ERA5T window length in days")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--area", nargs=4, type=float,
                    default=[30, 75, 5, 100], metavar=("N", "W", "S", "E"))
    ap.add_argument("--refresh", action="store_true",
                    help="re-download even if a live cache file exists")
    ap.add_argument("--no-track", action="store_true",
                    help="skip the temporal forecast stage")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    if not cds_configured():
        print("CDS credentials not configured (see ~/.cdsapirc); "
              "cannot pull live data.", file=sys.stderr)
        return 1

    path = download_live(n_days=args.n_days, area=args.area,
                         force=args.refresh)
    if path is None:
        print("live download returned nothing; aborting.", file=sys.stderr)
        return 1
    if args.download_only:
        print(f"Stage 1 download-only OK: {path}")
        return 0

    out = new_run_dir("live")
    summary = run_live_cycle(path, out, members=args.members,
                             steps=args.steps, warmup=args.warmup,
                             do_track=not args.no_track)
    print("\nSTAGE 1 STOP-AND-VERIFY (live, no truth):")
    print(f"  data: {path}")
    print(f"  timesteps {summary['timesteps']}, detected "
          f"{summary['detected']}, alerts {summary['n_alerts']}")
    print(f"  diffusion median peak-gain: {summary['median_peak_gain']}")
    print(f"  forecast boxes: {summary['n_forecast_boxes']}")
    print(f"  artefacts: {out.relative_to(OUTPUTS)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())