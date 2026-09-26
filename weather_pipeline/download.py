"""Phase 1a: download ERA5 sample data via the official CDS API.

Requires credentials configured at ~/.cdsapirc (see agent.txt section 4).
The event (amphan | yaas | fani) determines the date range and area via
`weather_pipeline.tracks.EVENTS`; each event is saved to
`data/raw/era5_{event}_test.nc` so files never overwrite each other.
"""
from __future__ import annotations

import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import xarray as xr

from .config import CONFIG, DATA_RAW
from .tracks import EVENTS

XMIT_GLOB = list(DATA_RAW.glob("era5_*_test.nc"))


def _cdsapirc_path() -> Path:
    return Path.home() / ".cdsapirc"


def cds_configured() -> bool:
    """True if ~/.cdsapirc exists and looks valid."""
    p = _cdsapirc_path()
    if not p.exists():
        return False
    text = p.read_text()
    return "url:" in text and ("key:" in text) and len(text.strip()) > 20


def _date_groups(start: str, end: str) -> list[tuple[str, str, list[str]]]:
    """Split the [start, end] inclusive range into (year, month, days)
    groups. CDS joins 'month' x 'day' as a cross product, so a request
    spanning a month boundary must be split into per-month retrieves."""
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    groups: dict[tuple[int, int], list[str]] = {}
    d = d0
    while d <= d1:
        groups.setdefault((d.year, d.month), []).append(f"{d.day:02d}")
        d += timedelta(days=1)
    return [(f"{y}", f"{m:02d}", days)
            for (y, m), days in groups.items()]


def _require_cds() -> bool:
    if cds_configured():
        return True
    print(
        "[download] CDS credentials not found at ~/.cdsapirc.\n"
        "  Register at https://cds.climate.copernicus.eu, copy the API key\n"
        "  from account settings, and create ~/.cdsapirc containing:\n"
        "    url: https://cds.climate.copernicus.eu/api\n"
        "    key: <uid>:<api-key>",
        file=sys.stderr,
    )
    return False


def _retrieve_merge(groups: list[tuple[str, str, list[str]]],
                    area, out: Path, tag: str) -> Path:
    """Retrieve one group per month into a temp dir, merge by the leading
    time dim, and write the merged netCDF to `out`.

    Windows note: `xr.load_dataset` loads into memory and closes each part
    file, so the temp dir can always be cleaned up.
    """
    import cdsapi

    c = cdsapi.Client()
    datasets = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for year, month, days in groups:
                params = {
                    "product_type": CONFIG.era5.product_type,
                    "variable": CONFIG.era5.variables,
                    "pressure_level": CONFIG.era5.pressure_levels,
                    "year": year,
                    "month": month,
                    "day": days,
                    "time": CONFIG.era5.times,
                    "area": list(area),
                    "grid": CONFIG.era5.grid,
                    "format": CONFIG.era5.fmt,
                }
                part = Path(tmp) / f"{tag}_{year}{month}.nc"
                print(f"[download] requesting {tag}: "
                      f"{year}-{month} days={days}")
                c.retrieve(CONFIG.era5.dataset, params, str(part))
                size_mb = part.stat().st_size / 1e6
                print(f"[download]   got {part.name} ({size_mb:.1f} MB)")
                datasets.append(xr.load_dataset(part))
            dim = next(iter(datasets[0].dims))     # leading dim name
            ds = xr.concat(datasets, dim=dim)
        ds.load()                                  # detach from file backends
        ds.to_netcdf(out)
    except Exception:
        out.unlink(missing_ok=True)                # no partial files
        raise
    print(f"[download] saved merged file to {out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    return out


def download_range(area, start: str, end: str, tag: str = "range",
                   force: bool = False,
                   out: Path | None = None) -> Path | None:
    """Generic ERA5 download for an explicit [start, end] ISO window.

    The single shared entry point behind every data path: used by the
    event downloads, the live pull, and the `Era5Source` adapter (Stage 3).
    """
    if not _require_cds():
        return None
    out = out or (DATA_RAW / f"era5_{tag}_test.nc")
    if out.exists() and out.stat().st_size > 0 and not force:
        print(f"[download] already present: {out}")
        return out
    groups = _date_groups(start, end)
    return _retrieve_merge(groups, area, out, tag=tag)


def download_era5(event: str = "amphan", dry_run: bool = False,
                  force: bool = False) -> Path | None:
    """Download the ERA5 test file for one event from the EVENTS registry.

    Returns the path to the merged per-event file, or None on dry/placeholder
    run.
    """
    if event not in EVENTS:
        print(f"[download] unknown event {event!r}; "
              f"choose from {sorted(EVENTS)}", file=sys.stderr)
        return None

    if not _require_cds():
        return None

    meta = EVENTS[event]
    out = DATA_RAW / f"era5_{event}_test.nc"
    if out.exists() and out.stat().st_size > 0 and not force:
        print(f"[download] already present: {out}")
        return out

    groups = _date_groups(meta["start"], meta["end"])
    if dry_run:
        print(f"[download] dry-run: would request {event} ({len(groups)} "
              f"month-group(s)):")
        for year, month, days in groups:
            print(f"  year={year} month={month} day={days} "
                  f"area={meta['area']} -> {out.name}")
        return None
    return download_range(meta["area"], meta["start"], meta["end"],
                          tag=event, force=force, out=out)


# ---------------------------------------------------------------------------
# Live (no fixed date) pull
# ---------------------------------------------------------------------------
LIVE_LAG_DAYS = 5           # ERA5 final ~5-day latency; serve ERA5T for these
LIVE_MAX_LAG_DAYS = 21      # walk back this far if CDS refuses the newest days
LIVE_WINDOW_DAYS = 5        # default number of days in a live window
LIVE_AREA = [30, 75, 5, 100]   # [N, W, S, E] Bay of Bengal / NIO


def latest_window(n_days: int = LIVE_WINDOW_DAYS,
                  lag_days: int = LIVE_LAG_DAYS) -> tuple[str, str]:
    """Inclusive [start, end] ISO dates ending `lag_days` before today.

    ERA5T (the near-real-time product) covers the most recent ~5 days; this
    returns a window entirely inside the T-support so `reanalysis-era5-
    pressure-levels` serves data for a live, unattended run.
    """
    end = date.today() - timedelta(days=lag_days)
    start = end - timedelta(days=n_days - 1)
    return start.isoformat(), end.isoformat()


def download_live(n_days: int = LIVE_WINDOW_DAYS, area=None,
                  dry_run: bool = False, force: bool = False,
                  max_lag_days: int = LIVE_MAX_LAG_DAYS,
                  out: Path | None = None) -> Path | None:
    """Pull the most recent available ERA5T window for the AOI.

    No hardcoded dates: the window is computed from today, then walked back
    (up to `max_lag_days`) if CDS has no data for the newest dates yet.
    Returns the merged dataset path; raises RuntimeError if no window works.
    """
    if not _require_cds():
        return None
    area = list(area) if area else list(LIVE_AREA)
    out = out or (DATA_RAW / "era5_live_test.nc")
    if out.exists() and out.stat().st_size > 0 and not force:
        print(f"[download] live data already present: {out}")
        return out

    for lag in range(LIVE_LAG_DAYS, max_lag_days + 1):
        start, end = latest_window(n_days, lag)
        groups = _date_groups(start, end)
        if dry_run:
            print(f"[download] dry-run: live window {start}..{end} "
                  f"(lag {lag}d, {len(groups)} month-group(s)) -> {out.name}")
            return None
        print(f"[download] trying live window {start}..{end} "
              f"(lag {lag} day(s))")
        try:
            return download_range(area, start, end, tag="live",
                                  force=force, out=out)
        except Exception as exc:                      # noqa: BLE001
            print(f"[download]   failed for lag {lag}d: "
                  f"{exc.__class__.__name__}: {exc}")
            out.unlink(missing_ok=True)              # retry clean
    raise RuntimeError(
        f"live download failed for all windows in the last {max_lag_days} "
        f"days against {CONFIG.era5.dataset}; "
        f"check CDS availability/credentials and report upstream.")