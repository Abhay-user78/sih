"""Phase 1 download script.

Usage:
    python -m scripts.phase1_download [--event amphan|yaas|fani] [--dry-run]

Real CDS credentials must be present at ~/.cdsapirc.
"""
from __future__ import annotations

import argparse
import sys

from weather_pipeline.download import download_era5
from weather_pipeline.tracks import EVENTS


def main() -> int:
    ap = argparse.ArgumentParser(description="Download ERA5 Amphan test data")
    ap.add_argument("--event", default="amphan",
                    choices=sorted(EVENTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = download_era5(event=args.event, dry_run=args.dry_run)
    return 0 if out is not None or args.dry_run else 1


if __name__ == "__main__":
    sys.exit(main())