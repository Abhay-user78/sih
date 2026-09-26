"""STAGE 2: automated operational trigger for the live pipeline cycle.

One call runs the whole Stage-1 chain (download -> GNN -> diffusion ->
temporal -> alerts) and appends an audit record to outputs/run_log.jsonl.

  python -m scripts.run_pipeline_cycle --once          # single cycle, exit
  python -m scripts.run_pipeline_cycle --forever --every 6  # every 6 h (00/06/12/18Z)

For cron-based scheduling instead of the internal APScheduler loop, drop
`--once` into a crontab line (documented in README):

  20 */6 * * *  cd /path/to/Spatio-Temporal && \
      .venv/bin/python -m scripts.run_pipeline_cycle --once \
      >> outputs/cron.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime

from weather_pipeline.config import OUTPUTS
from weather_pipeline.download import download_live, cds_configured
from weather_pipeline.operational import run_live_cycle, new_run_dir

RUN_LOG = OUTPUTS / "run_log.jsonl"


def _log(record: dict) -> None:
    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def run_cycle(n_days: int = 5, members: int = 8, steps: int = 50,
              warmup: int = 10, area=None, refresh: bool = False) -> dict:
    """One unattended live cycle; raises on any failure. Reusable (not a
    subprocess), so the API/other services can call it directly."""
    if not cds_configured():
        raise RuntimeError("CDS credentials not configured (~/.cdsapirc)")
    path = download_live(n_days=n_days, area=area, force=refresh)
    if path is None:
        raise RuntimeError("live download returned no data")
    out = new_run_dir("cycle")
    summary = run_live_cycle(path, out, members=members, steps=steps,
                             warmup=warmup, label="cycle")
    return summary


def execute_once(args: argparse.Namespace) -> int:
    started = datetime.now()
    try:
        summary = run_cycle(n_days=args.n_days, members=args.members,
                            steps=args.steps, warmup=args.warmup,
                            area=args.area, refresh=args.refresh)
        _log({
            "ts": started.strftime("%Y-%m-%dT%H:%M:%S"),
            "success": True,
            "out_dir": summary.get("out_dir") if "out_dir" in summary
            else None,
            "timesteps": summary["timesteps"],
            "detected": summary["detected"],
            "n_alerts": summary["n_alerts"],
            "n_forecast_boxes": summary["n_forecast_boxes"],
            "median_peak_gain": summary["median_peak_gain"],
        })
        print(f"[cycle] OK ts={started} alerts={summary['n_alerts']} "
              f"detected={summary['detected']}/{summary['timesteps']}")
        return 0
    except Exception as exc:                              # noqa: BLE001
        _log({"ts": started.strftime("%Y-%m-%dT%H:%M:%S"),
              "success": False,
              "error": f"{exc.__class__.__name__}: {exc}",
              "traceback": traceback.format_exc().splitlines()[-3:]})
        print(f"[cycle] FAILED ({exc.__class__.__name__}): {exc}",
              file=sys.stderr)
        return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true",
                    help="run a single cycle and exit (default)")
    ap.add_argument("--forever", action="store_true",
                    help="stay resident and run every --every hours "
                         "(APScheduler)")
    ap.add_argument("--every", type=int, default=6,
                    help="cycle interval in hours when --forever")
    ap.add_argument("--n-days", type=int, default=5)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--area", nargs=4, type=float,
                    default=[30, 75, 5, 100], metavar=("N", "W", "S", "E"))
    ap.add_argument("--refresh", action="store_true",
                    help="force a live re-download this cycle")
    args = ap.parse_args()

    if args.forever:
        from apscheduler.schedulers.blocking import BlockingScheduler

        def job():
            execute_once(args)

        print(f"[cycle] resident scheduler: one run every {args.every}h "
              f"(matches 00/06/12/18Z NCUM/NEPS-G cycles)")
        print(f"[cycle] audit trail: {RUN_LOG}")
        print("[cycle] cron alternative: python -m scripts.run_pipeline_cycle"
              " --once  (see README)")
        sched = BlockingScheduler()
        sched.add_job(job, "interval", hours=args.every)
        try:
            sched.start()
        except (KeyboardInterrupt, SystemExit):
            print("[cycle] scheduler stopped")
        return 0

    return execute_once(args)


if __name__ == "__main__":
    sys.exit(main())