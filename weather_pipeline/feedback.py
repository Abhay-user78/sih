import json
import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
FEEDBACK_FILE = OUTPUT_DIR / "phase9_feedback.jsonl"


def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * \
        math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _truth_track(n_time=24):
    lat = np.linspace(10.5, 22.5, n_time)
    lon = np.linspace(89.5, 88.7, n_time)
    return list(zip(lat.tolist(), lon.tolist()))


class FeedbackStore:
    def __init__(self, path: Path | None = None,
                 lookup: Optional[Callable[[str], dict]] = None,
                 alerts: list | None = None):
        self.path = path or FEEDBACK_FILE
        self.lookup = lookup or (lambda _id: {})
        self.alerts = alerts or []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = self.load()

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            for r in self.records:
                fh.write(json.dumps(r) + "\n")

    def add(self, alert_id: str, correct: bool,
            true_lat: float | None = None,
            true_lon: float | None = None,
            note: str = "") -> dict:
        a = self.lookup(alert_id)
        err = None
        if true_lat is not None and true_lon is not None:
            err = haversine_km(a.get("lat", 0.0), a.get("lon", 0.0),
                               true_lat, true_lon)
        rec = {
            "id": alert_id,
            "correct": bool(correct),
            "true_lat": true_lat,
            "true_lon": true_lon,
            "centre_err_km": round(err, 1) if err is not None else None,
            "severity": a.get("severity", "unknown"),
            "confidence": a.get("confidence", 0.0),
            "note": note,
            "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.records = [r for r in self.records if r["id"] != alert_id]
        self.records.append(rec)
        self.save()
        return rec

    def summary(self) -> dict:
        tot = len(self.records)
        pos = [r for r in self.records if r["correct"]]
        neg = [r for r in self.records if not r["correct"]]
        errs = [r["centre_err_km"] for r in self.records
                if r.get("centre_err_km") is not None]
        return {
            "reviewed": tot,
            "confirmed": len(pos),
            "false_alarms": len(neg),
            "confirmed_rate": (len(pos) / tot) if tot else 0.0,
            "median_centre_err_km": round(float(np.median(errs)), 1)
            if errs else None,
            "max_centre_err_km": round(float(np.max(errs)), 1)
            if errs else None,
        }

    def auto_review_all(self, max_err_km: float = 40.0,
                        n_time: int = 24,
                        radius_km: float = 5.0) -> list[dict]:
        truth = _truth_track(n_time)
        added = []
        for a in self.alerts:
            t = a["timestep_index"]
            if t >= len(truth):
                continue
            tlat, tlon = truth[t]
            err = haversine_km(a["lat"], a["lon"], tlat, tlon)
            added.append(self.add(
                a["id"], correct=(err <= max_err_km + radius_km),
                true_lat=tlat, true_lon=tlon,
                note=f"auto-review vs truth track (err {err:.1f} km)"))
        return added


def apply_feedback(dataset: dict, feedback_store: FeedbackStore) -> dict:
    """Return a label mask that zeroes labels near false-alarm centres."""
    n_time = int(dataset["n_time"])
    labels = np.asarray(dataset["labels"], dtype=np.float32)
    mask = np.ones_like(labels, dtype=np.float32)
    node_lat = dataset["node_lat"]
    node_lon = dataset["node_lon"]
    deg_per_mesh = 0.25 / 2.0 ** 2
    for r in feedback_store.records:
        if r["correct"] or r.get("true_lat") is None:
            continue
        t = _timestep_of(r["id"])
        if t is None or t >= n_time:
            continue
        tl, to = r["true_lat"], r["true_lon"]
        d = np.sqrt((node_lat - tl) ** 2 + (node_lon - to) ** 2)
        mask[t][d < 3.0 * deg_per_mesh] = 0.0
    return {"labels": labels, "mask": mask}


def _timestep_of(alert_id: str):
    parts = alert_id.split("-")
    for p in parts:
        if p.isdigit() and len(p) <= 3:
            return int(p)
    return None