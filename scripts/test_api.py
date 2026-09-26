import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from weather_pipeline.api import create_app
import weather_pipeline.api as api_mod

OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    app = create_app()

    # Keep this test hermetic: redirect review + feedback persistence to
    # temp files so running it never mutates operational outputs.
    tmp_review = OUT / "_test_review_state.json"
    tmp_feedback = OUT / "_test_feedback.jsonl"
    api_mod.REVIEW_STATE_FILE = tmp_review
    for p in (tmp_review, tmp_feedback):
        p.unlink(missing_ok=True)
    app.state.store._review_state = {}
    fb = app.state.feedback
    fb.path = tmp_feedback
    fb.records = []

    client = TestClient(app)
    checks = []

    def check(name, cond, extra=""):
        checks.append(all([cond]))
        mark = "PASS" if cond else "FAIL"
        print(f"  [{mark}] {name} {extra}")

    print("--- Phase 7 API checks (Stage-5 review gating) ---")
    r = client.get("/api/health")
    h = r.json()
    check("health", r.status_code == 200 and h["status"] == "ok",
          f"(code {r.status_code})")

    r = client.get("/")
    body = r.json()
    check("root summary", r.status_code == 200
          and body["alerts_total"] == 40
          and body["forecast_boxes"] == 26
          and body["timesteps_processed"] == 24,
          f"(alerts={body['alerts_total']}, fc={body['forecast_boxes']})")

    r = client.get("/api/alerts")
    alerts = r.json()
    check("default listing = non-rejected, incl pending",
          len(alerts) == 40
          and all(a["status"] != "rejected" for a in alerts),
          f"({len(alerts)} alerts)")
    required = {"id", "timestamp", "lat", "lon", "radius_km",
                "downscaled_radius_km", "severity", "confidence",
                "confidence_breakdown", "downscale", "status"}
    check("all alerts have required fields", all(
        required.issubset(a) for a in alerts), f"({len(alerts)} alerts)")

    r = client.get("/api/alerts", params={"status": "confirmed"})
    confirmed = r.json()
    check("confirmed listing = historical (n_ensemble 16)",
          len(confirmed) == 24
          and all(a["status"] == "confirmed" for a in confirmed),
          f"({len(confirmed)} confirmed)")

    r = client.get("/api/alerts", params={"status": "pending_review"})
    pend = r.json()
    check("pending listing = automated live cycle alerts",
          len(pend) == 16
          and all(a["status"] == "pending_review" for a in pend),
          f"({len(pend)} pending)")

    r = client.get("/api/alerts/pending")
    check("pending endpoint == pending listing",
          r.json() == pend, f"({len(r.json())} pending)")

    r = client.get("/api/alerts/active")
    active = r.json()
    t = max(a["timestep_index"] for a in active)
    check("active = latest-step, confirmed only",
          len(active) == 1 and t == max(a["timestep_index"] for a in alerts)
          and active[0]["status"] == "confirmed",
          f"(n={len(active)}, t={t})")

    aid = active[0]["id"]
    r = client.get(f"/api/alerts/{aid}")
    detail = r.json()
    check("alert detail by id", r.status_code == 200
          and detail["id"] == aid
          and "peak_gain" in detail["downscale"],
          f"({aid})")

    r = client.get("/api/alerts/amphan-forecast-7")
    check("forecast box endpoint", r.status_code == 200
          and r.json()["type"] == "forecast"
          and r.json()["lead_hours"] == 6)

    r = client.get("/api/forecast")
    fc = r.json()
    leads = sorted(f["lead_hours"] for f in fc)
    ok = len(fc) == 26 and all(f["type"] == "forecast" for f in fc) \
        and leads == sorted(leads) and leads[-1] == 102
    check("forecast list", ok, f"(n={len(fc)}, leads={leads[0]}..{leads[-1]})")

    r = client.get("/api/exposure/13.5/90.8")
    ex = r.json()
    check("exposure near storm -> risk",
          r.status_code == 200 and ex["risk"] in
          ("low", "medium", "high") and ex["nearest"] is not None,
          f"(risk={ex['risk']}, score={ex['risk_score']})")

    r = client.get("/api/exposure/5.0/75.0")
    far = r.json()
    check("exposure far away -> none",
          far["risk"] == "none", f"(risk={far['risk']})")

    r = client.post("/api/exposure",
                    json={"lat": 13.5, "lon": 90.8, "radius_km": 100.0})
    check("exposure POST", r.status_code == 200
          and r.json()["risk"] == ex["risk"])

    r = client.get("/api/alerts/does-not-exist")
    check("404 on unknown alert", r.status_code == 404)

    r = client.get("/api/feedback")
    fl = r.json()
    check("feedback summary endpoint", r.status_code == 200
          and "summary" in fl and "records" in fl)

    aid = alerts[0]["id"]
    t_idx = alerts[0]["timestep_index"]
    t_lat = 10.5 + t_idx * (22.5 - 10.5) / 23.0
    t_lon = 89.5 - t_idx * (89.5 - 88.7) / 23.0
    r = client.post(f"/api/alerts/{aid}/feedback",
                    json={"correct": True, "true_lat": round(t_lat, 3),
                          "true_lon": round(t_lon, 3), "note": "operator check"})
    rec = r.json()["record"]
    check("submit alert feedback", r.status_code == 200
          and rec["id"] == aid and rec["centre_err_km"] is not None,
          f"(err={rec['centre_err_km']} km)")

    r = client.post("/api/feedback/auto-review")
    s = r.json()["summary"]
    check("auto-review all alerts", s["reviewed"] >= 40
          and 0 <= s["confirmed_rate"] <= 1,
          f"(reviewed={s['reviewed']} confirmed={s['confirmed']} "
          f"fa={s['false_alarms']})")

    # Stage-5 gate: reviewing a pending alert promotes it to the confirmed
    #	pending list, and decisions persist to the (redirected) state file.
    target = pend[0]["id"]
    r = client.post(f"/api/alerts/{target}/review",
                    json={"decision": "confirmed", "note": "test review"})
    promoted = r.json()["record"]
    check("review endpoint promotes pending -> confirmed",
          r.status_code == 200 and promoted["id"] == target
          and promoted["status"] == "confirmed",
          f"({target})")
    r = client.get("/api/alerts", params={"status": "pending_review"})
    check("reviewed alert leaves the pending queue",
          target not in [a["id"] for a in r.json()])
    check("review decision persisted",
          json.loads(tmp_review.read_text(encoding="utf-8"))
          .get(target, {}).get("status") == "confirmed")

    for p in (tmp_review, tmp_feedback):
        p.unlink(missing_ok=True)

    print()
    passed = sum(checks)
    print(f"Phase 7: {passed}/{len(checks)} checks passed")
    print("PHASE 7 STOP-AND-VERIFY: FastAPI alerting service (Stage-5 "
          "review gate) is ready; run `python -m scripts.serve_api` "
          "to serve on :8000")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())