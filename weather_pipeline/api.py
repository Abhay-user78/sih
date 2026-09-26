import json
import math
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from weather_pipeline.feedback import FeedbackStore

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
ALERTS_FILE = OUTPUT_DIR / "phase4_alerts.json"
FORECAST_FILE = OUTPUT_DIR / "phase5_amphan.json"

SEVERITY_WEIGHT = {"low": 1.0, "moderate": 2.0, "severe": 3.0}
PINPOINT_RADIUS_KM = 6.0
EXPOSURE_RADIUS_CAP_KM = 300.0
REVIEW_STATE_FILE = OUTPUT_DIR / "review_state.json"

STATUS_PENDING = "pending_review"
STATUS_CONFIRMED = "confirmed"
STATUS_REJECTED = "rejected"
STATUSES = (STATUS_PENDING, STATUS_CONFIRMED, STATUS_REJECTED)


class DownscaleDetail(BaseModel):
    coarse_peak: float
    downscaled_peak: float
    peak_gain: float
    n_ensemble: int


class ConfidenceBreakdown(BaseModel):
    severity_prob_mean: float
    n_ensemble: int
    gnn_mean_score: float
    gnn_max_score: float


class FeedbackRequest(BaseModel):
    correct: bool
    true_lat: Optional[float] = None
    true_lon: Optional[float] = None
    note: str = ""


class ReviewRequest(BaseModel):
    decision: str = Field(pattern="^(confirmed|rejected)$")
    note: str = ""


class AlertRecord(BaseModel):
    id: str
    type: str = "alert"
    timestamp: str
    timestep_index: int
    lat: float
    lon: float
    radius_km: float
    downscaled_radius_km: float = PINPOINT_RADIUS_KM
    severity: str
    confidence: float = 0.0
    confidence_breakdown: Optional[ConfidenceBreakdown] = None
    downscale: Optional[DownscaleDetail] = None
    bbox: Optional[list] = None
    level_hPa: Optional[int] = 850
    lead_hours: Optional[int] = 0
    track_error_km: Optional[float] = None
    status: str = STATUS_CONFIRMED
    reviewed_note: str = ""


class ExposureRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_km: float = Field(default=150.0, ge=0.0)


class ExposureResult(BaseModel):
    queried_at: str
    risk: str
    risk_score: float
    nearest: Optional[dict] = None
    contributing_alerts: list[dict]


class AlertStore:
    def __init__(self, alerts_file: Path | None = None,
                 forecast_file: Path | None = None):
        self.alerts_file = alerts_file or ALERTS_FILE
        self.forecast_file = forecast_file or FORECAST_FILE
        self.records: dict[str, AlertRecord] = {}
        self._review_state: dict[str, dict] = {}
        if REVIEW_STATE_FILE.exists():
            self._review_state = json.load(
                open(REVIEW_STATE_FILE, encoding="utf-8")) or {}
        self.load()
        self._apply_review_state()

    def _apply_review_state(self) -> None:
        """Persist review decisions across restarts."""
        for alert_id, stmeta in self._review_state.items():
            if alert_id in self.records:
                rec = self.records[alert_id]
                rec.status = stmeta.get("status", rec.status)
                rec.reviewed_note = stmeta.get("note", "")
                rec.timestamp = stmeta.get("at", rec.timestamp) + \
                    " (reviewed)"
                self.records[alert_id] = rec

    def load(self) -> None:
        if not self.alerts_file.exists():
            raise FileNotFoundError(f"missing alerts file: {self.alerts_file}")
        with open(self.alerts_file, encoding="utf-8") as fh:
            raw_alerts = json.load(fh)
        for a in raw_alerts:
            rec = self._normalize_alert(
                a, status=a.get("status", STATUS_CONFIRMED))
            self.records[rec.id] = rec
        if self.forecast_file.exists():
            with open(self.forecast_file, encoding="utf-8") as fh:
                raw_boxes = json.load(fh)
            for b in raw_boxes:
                rec = self._normalize_box(b)
                self.records[rec.id] = rec
        self._ingest_live_runs()

    def _ingest_live_runs(self) -> None:
        """Automated-cycle alerts enter the system as pending_review and are
        NOT visible via /api/alerts/active until a forecaster reviews them.
        Only the newest live run is ingested (older runs are superseded by
        the next cycle)."""
        live_dirs = sorted(OUTPUT_DIR.glob("live_run_*"),
                           key=lambda p: p.name)
        if not live_dirs:
            return
        newest = live_dirs[-1]
        alerts_file = newest / "live_alerts.json"
        forecast_file = newest / "live_forecast.json"
        if alerts_file.exists():
            with open(alerts_file, encoding="utf-8") as fh:
                raw = json.load(fh)
            for a in raw:
                rec = self._normalize_alert(
                    a, status=a.get("status", STATUS_PENDING))
                self.records[rec.id] = rec
        if forecast_file.exists():
            with open(forecast_file, encoding="utf-8") as fh:
                raw = json.load(fh)
            for b in raw:
                rec = self._normalize_box(b, prefix="live")
                self.records[rec.id] = rec

    @staticmethod
    def _normalize_alert(a: dict,
                         status: str = STATUS_CONFIRMED) -> AlertRecord:
        return AlertRecord(
            id=a["id"],
            type="alert",
            timestamp=a.get("timestep", str(a["timestep_index"])),
            timestep_index=int(a["timestep_index"]),
            lat=float(a["centre_lat"]),
            lon=float(a["centre_lon"]),
            radius_km=float(a.get("radius_km", 0.0)),
            downscaled_radius_km=PINPOINT_RADIUS_KM,
            severity=a.get("severity", "unknown"),
            confidence=float(a.get("confidence", 0.0)),
            confidence_breakdown=ConfidenceBreakdown(
                severity_prob_mean=float(a.get("severity_prob_mean", 0.0)),
                n_ensemble=int(a.get("n_ensemble", 0)),
                gnn_mean_score=float(a.get("gnn_mean_score", 0.0)),
                gnn_max_score=float(a.get("gnn_max_score", 0.0)),
            ),
            downscale=DownscaleDetail(
                coarse_peak=float(a.get("coarse_peak", 0.0)),
                downscaled_peak=float(a.get("downscaled_peak", 0.0)),
                peak_gain=float(a.get("peak_gain", 0.0)),
                n_ensemble=int(a.get("n_ensemble", 0)),
            ),
            bbox=list(a["bbox"]) if "bbox" in a else None,
            level_hPa=a.get("level_hPa", 850),
            lead_hours=0,
            status=status,
            reviewed_note=a.get("reviewed_note", ""),
        )

    @staticmethod
    def _normalize_box(b: dict, prefix: str = "amphan") -> AlertRecord:
        """Normalise a forecast/observed track box.

        Track records carry no detection score: ``error_km`` is a position
        error against the best track (km), so it is exposed as
        ``track_error_km``. ``confidence`` stays 0.0 here — the 0-1
        detection confidence only means something for alert records.
        """
        c = b.get("centre", {})
        kind = b.get("kind", "observed")
        lead = int(b.get("lead_hours") or 0)
        err = b.get("error_km")
        return AlertRecord(
            id=f"{prefix}-{kind}-{int(b['timestep_index'])}",
            type=kind,
            timestamp=b.get("time", ""),
            timestep_index=int(b["timestep_index"]),
            lat=float(c["lat"]),
            lon=float(c["lon"]),
            radius_km=PINPOINT_RADIUS_KM,
            downscaled_radius_km=PINPOINT_RADIUS_KM,
            severity="severe" if kind == "forecast" else "unknown",
            confidence=0.0,
            level_hPa=b.get("level_hPa", 850),
            lead_hours=lead,
            track_error_km=float(err) if err is not None else None,
        )

    def by_id(self, alert_id: str) -> AlertRecord:
        if alert_id not in self.records:
            raise HTTPException(status_code=404,
                                detail=f"alert not found: {alert_id}")
        return self.records[alert_id]

    def latest_index(self) -> int:
        return max(r.timestep_index for r in self.records.values())

    def alerts(self, active: bool = False,
               min_severity: str | None = None,
               status: str | None = None,
               limit: int = 500) -> list[AlertRecord]:
        recs = [r for r in self.records.values() if r.type == "alert"]
        if active:
            t = self.latest_index()
            recs = [r for r in recs
                    if r.timestep_index == t and r.status == STATUS_CONFIRMED]
        elif status is None:
            recs = [r for r in recs if r.status != STATUS_REJECTED]
        if status:
            recs = [r for r in recs if r.status == status]
        if min_severity:
            recs = [r for r in recs
                    if SEVERITY_WEIGHT.get(r.severity, 0) >=
                    SEVERITY_WEIGHT.get(min_severity, 0)]
        recs.sort(key=lambda r: (-r.timestep_index,
                                 -SEVERITY_WEIGHT.get(r.severity, 0)))
        return recs[:limit]

    def pending(self) -> list[AlertRecord]:
        return self.alerts(status=STATUS_PENDING)

    def review(self, alert_id: str, decision: str, note: str = "") -> AlertRecord:
        """Forecaster gate: confirm or reject an alert.

        Confirmed alerts become visible through /api/alerts (and hence the
        /active endpoint); rejected ones are hidden from public listings.
        Decisions are persisted to outputs/review_state.json.
        """
        rec = self.by_id(alert_id)
        rec.status = decision
        rec.reviewed_note = note
        at = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec.timestamp = at + " (reviewed)"
        self.records[alert_id] = rec
        self._review_state[alert_id] = {"status": decision, "note": note,
                                        "at": at}
        with open(REVIEW_STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(self._review_state, fh, indent=2)
        return rec

    def forecast(self) -> list[AlertRecord]:
        return sorted((r for r in self.records.values()
                       if r.type == "forecast"),
                      key=lambda r: r.timestep_index)

    @staticmethod
    def _reach_km(r: AlertRecord) -> float:
        """Radius used by exposure scoring, capped.

        ``radius_km`` comes from the coarse GNN cluster and can exceed
        1000 km; without the cap ``reach * 3.0`` swallows the whole
        domain and points far from any storm score as at-risk.
        """
        return min(r.radius_km, EXPOSURE_RADIUS_CAP_KM)

    def exposure(self, req: ExposureRequest) -> ExposureResult:
        lat, lon = req.lat, req.lon
        hits = []
        for r in self.records.values():
            if r.type == "alert" and r.status == STATUS_CONFIRMED:
                d = haversine_km(lat, lon, r.lat, r.lon)
                reach = max(req.radius_km, self._reach_km(r))
                if d <= reach * 3.0:
                    hits.append((d, r))
        score = 0.0
        contrib = []
        for d, r in sorted(hits, key=lambda x: x[0]):
            base = max(self._reach_km(r), 10.0)
            near = max(0.0, 1.0 - (d / (base * 3.0)))
            w = SEVERITY_WEIGHT.get(r.severity, 1.0)
            score += near * w
            contrib.append({
                "id": r.id, "distance_km": round(d, 1),
                "severity": r.severity, "radius_km": r.radius_km,
                "confidence": r.confidence,
            })
        if score <= 0:
            risk, score = "none", 0.0
        elif score < 0.6:
            risk = "low"
        elif score < 1.4:
            risk = "medium"
        else:
            risk = "high"
        nearest = None
        if hits:
            d, r = hits[0]
            nearest = {"id": r.id, "distance_km": round(d, 1),
                       "severity": r.severity,
                       "radius_km": r.radius_km}
        return ExposureResult(queried_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                              risk=risk, risk_score=round(score, 3),
                              nearest=nearest, contributing_alerts=contrib)


def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * \
        math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def create_app() -> FastAPI:
    app = FastAPI(title="Amplicast Weather Intelligence API",
                  version="0.7.0",
                  description="Amplicast spatio-temporal anomaly detection, "
                              "forecast tracking, exposure, and human review.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    store = AlertStore()

    @app.get("/")
    def root():
        latest = store.latest_index()
        return {
            "service": "amplicast-weather-api",
            "version": "0.7.0",
            "timesteps_processed": latest + 1,
            "alerts_total": len([r for r in store.records.values()
                                 if r.type == "alert"]),
            "forecast_boxes": len(store.forecast()),
            "endpoints": ["/api/alerts", "/api/alerts/{id}",
                          "/api/alerts/pending",
                          "/api/alerts/{id}/review",
                          "/api/forecast",
                          "/api/exposure/{lat}/{lon}"],
        }

    @app.get("/api/alerts")
    def list_alerts(active: bool = False,
                    min_severity: Optional[str] = Query(
                        None, pattern="^(low|moderate|severe)$"),
                    status: Optional[str] = Query(
                        None, pattern="^(pending_review|confirmed|rejected)$"),
                    limit: int = 500):
        return store.alerts(active=active, min_severity=min_severity,
                            status=status, limit=limit)

    @app.get("/api/alerts/active")
    def active_alerts():
        return store.alerts(active=True)

    @app.get("/api/alerts/pending")
    def pending_alerts():
        return store.pending()

    @app.get("/api/alerts/{alert_id}")
    def get_alert(alert_id: str):
        return store.by_id(alert_id)

    @app.post("/api/alerts/{alert_id}/review")
    def review_alert(alert_id: str, req: ReviewRequest):
        """Human-in-the-loop gate. Only `confirmed` alerts are visible via
        /api/alerts/active (and the dashboard's confirmed listings)."""
        rec = store.review(alert_id, req.decision, req.note)
        return {"record": rec,
                "active_now": store.alerts(active=True)}

    @app.get("/api/forecast")
    def get_forecast():
        return store.forecast()

    @app.get("/api/exposure/{lat}/{lon}")
    def exposure_get(lat: float, lon: float,
                     radius_km: float = 150.0):
        return store.exposure(ExposureRequest(
            lat=lat, lon=lon, radius_km=radius_km))

    @app.post("/api/exposure")
    def exposure_post(req: ExposureRequest):
        return store.exposure(req)

    @app.get("/api/health")
    def health():
        return {"status": "ok",
                "sources": {"alerts": str(store.alerts_file),
                            "forecast": str(store.forecast_file)},
                "alerts_loaded": len(store.records),
                "pending_review": len(store.pending()),
                "confirmed": len([r for r in store.records.values()
                                  if r.status == STATUS_CONFIRMED])}

    app.state.store = store
    fb = FeedbackStore(
        lookup=lambda _id: store.by_id(_id).model_dump(),
        alerts=[a.model_dump() for a in store.alerts()])
    app.state.feedback = fb

    @app.get("/api/feedback")
    def feedback_summary():
        return {"summary": fb.summary(), "records": fb.records}

    @app.post("/api/alerts/{alert_id}/feedback")
    def submit_feedback(alert_id: str, req: FeedbackRequest):
        rec = fb.add(alert_id, req.correct, req.true_lat, req.true_lon,
                     req.note)
        return {"record": rec, "summary": fb.summary()}

    @app.post("/api/feedback/auto-review")
    def auto_review():
        added = fb.auto_review_all()
        return {"reviewed": len(added), "summary": fb.summary()}

    return app


app = create_app()