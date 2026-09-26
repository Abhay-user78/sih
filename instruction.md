# Spatio-Temporal Cyclone Early-Warning Dashboard

Averaged-over-everything, this repo runs a **hybrid AI pipeline** that watches ERA5
weather fields for the Bay of Bengal, finds organised cyclonic rotation (a GNN),
sharpens the core peak (a diffusion ensemble), projects the track forward (a
temporal transformer), and puts every detection in a **human-reviewed alert feed**
served by a FastAPI backend + Streamlit dashboard.

The included pre-computed run (`outputs/live_run_20260924_221709`) gives you a
fully working demo out of the box: 16 alerts waiting in the Review Queue,
8 forecast boxes, and the two historical sources (Amphan detections + forecast)
that feed the Overview/Alerts/Exposure pages.

---

## 1. What's inside

```
spatio-temporal/
  app/dashboard.py              Streamlit UI (6 pages)
  weather_pipeline/             core package (detection, diffusion, temporal,
                                pipeline, live operational cycle, FastAPI app)
  scripts/                      drivers: live_run, run_pipeline_cycle,
                                phase_era5, aggregate_validation, serve_api,
                                test_api
  models/                       frozen inference weights (ERA5 variants only)
    gnn/gnn_anomaly_era5.pt
    gnn/gnn_anomaly_fb.pt
    gnn/dataset_meta.json
    diffusion/downscale_unet_era5.pt
    temporal/temporal_tracker_era5_ft.pt
  outputs/
    phase4_alerts.json          historical Amphan detections (API source)
    phase5_amphan.json          historical Amphan forecast boxes (API source)
    live_run_20260924_221709/   latest automated run (pending-review alerts)
  requirements.txt              pinned Python dependencies
  Dockerfile / docker-compose.yml / .dockerignore   container deployment
  instruction.md                this file
```

Not included (tiny by design): raw ERA5 netCDFs (~250 MB in the full project) and
non-ERA5 model variants. See section 7 to re-fetch ERA5 data for a fresh live run.

---

## 2. Run it (Windows / Linux, 5 minutes)

Requires Python 3.10-3.13. A GPU is not required — everything runs on CPU if
needed (slower diffusion sampling).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate ; pip install -r requirements.txt
# Linux / macOS
source .venv/bin/activate      ; pip install -r requirements.txt
```

Start the API (port 8000):

```bash
python -m scripts.serve_api
# or: uvicorn weather_pipeline.api:app --host 0.0.0.0 --port 8000
```

In a second terminal, start the dashboard (port 8501):

```bash
streamlit run app/dashboard.py
```

Open **http://localhost:8501**. The API is at **http://localhost:8000**, docs at
**http://localhost:8000/docs** (Swagger), health at `/api/health`.

If port 8000 is already in use (e.g. an earlier API instance is running), stop it
first, or serve on another port and adjust nothing locally except the port in your
URLs.

---

## 3. What you will see

| Page          | Shows |
|---------------|-------|
| Overview      | most recent confirmed alert, severity mix, confidence timeline, forecast boxes |
| Alerts        | full alert list (filter by severity/status), per-alert detail + confidence breakdown, exposure risk |
| Review Queue  | pending-alert triage: **Confirm / Reject** with an optional note |
| Forecast      | temporal-transformer track vs truth (historical) |
| Exposure      | point-risk score for any lat/lon |
| API           | raw JSON via the FastAPI backend |

### Confirming alerts (the human-in-the-loop step)

Automated cycles land in the Review Queue as `pending_review`. Nothing pending is
ever served by the "active alert" endpoint — a human must confirm it first.

- **Confirm** -> promoted to the active feed and visible on Overview.
- **Reject** -> hidden from all default listings.
- Decisions persist to `outputs/review_state.json` and survive restarts.

API equivalent:

```bash
curl -X POST http://localhost:8000/api/alerts/<id>/review \
     -H "Content-Type: application/json" \
     -d '{"decision":"confirmed","note":"looks good"}'
curl http://localhost:8000/api/alerts/pending          # queue
curl http://localhost:8000/api/alerts/active           # confirmed, latest step
```

### Reading the confidence field (important)

`confidence` is a **detection confidence**, defined identically everywhere as:

```
confidence = 0.5 * (GNN cluster mean anomaly score) +
             0.5 * (ensemble agreement)
```

where *ensemble agreement* is the fraction of diffusion members whose downscaled
peak stays at/above the coarse peak. It is **NOT an inverted severity
probability** — a weak, unsharpened detection shows a LOW confidence. The danger
signal is `severity` + `severity_prob_mean` + `peak_gain` (separate fields). The
included live run demonstrates this: its detections are soft (~0.07-0.68
confidence), which is honest.

Full alert field list: `id`, `timestep(_index)`, `centre_lat/lon`, `bbox`,
`radius_km`, `severity`, `confidence`, `coarse_peak`, `downscaled_peak`,
`peak_gain`, `n_ensemble`, `severity_prob_mean`, `gnn_mean_score`,
`gnn_max_score`, `status`.

Forecast/observed **track** records (`/api/forecast`) are a different animal:
they carry no detection score, so their `confidence` is always `0.0`. Their
position error against the best track is reported separately as
`track_error_km` (km, from the source `error_km`, `null` for a live forecast
with no truth yet) — never read it as a 0-1 confidence.

---

## 4. Run a fresh automated cycle (new data)

1. Create a free Copernicus account and save a `.cdsapirc` (uid + key) to your
   home directory.
2. Fetch the latest ERA5T window and run the full chain:

```bash
python -m scripts.live_run --n-days 5
# scheduled mode (every 6 h):
python -m scripts.run_pipeline_cycle --once
# or as cron:
# 20 */6 * * * cd /path/to/spatio-temporal && .venv/bin/python -m scripts.run_pipeline_cycle --once >> outputs/cron.log 2>&1
```

Each cycle writes a new `outputs/live_run_<timestamp>/` whose alerts land in the
Review Queue as pending. `python -m scripts.test_api` is a 21-check hermetic test
of the whole API contract.

---

## 5. Multi-event validation (frozen weights, no retraining)

```bash
python -m scripts.phase_era5 --event amphan --no-train
python -m scripts.phase_era5 --event yaas  --no-train
python -m scripts.phase_era5 --event fani   --no-train
python -m scripts.aggregate_validation
```

Requires `data/raw/era5_{amphan,yaas,fani}_test.nc` (see section 7). Results,
scale-free against the Amphan baseline (hard-stop cap = 3x 28.9 km):

| event  | det  | intensif. window err | peak-gain | tracker beats persistence |
|--------|------|----------------------|-----------|---------------------------|
| amphan | 23/24 | 28.9 km              | 0.958     | 3/3                       |
| yaas   | 18/20 | 22.9 km              | 1.002     | 1/3                       |
| fani   | 23/24 | 23.5 km              | 0.989     | 0/3                       |

Honest note: detection and core downscaling pass all gates on all three storms;
the **short-lead track forecast does not beat persistence for Yaas/Fani**.
The models were NOT tuned to hide this — it is a known weakness of this tracker.

---

## 6. Docker

```bash
docker compose up --build
```

- API on :8000, dashboard on :8501.
- `./outputs` and `./data` are bind-mounted; an optional `./cdsapirc` is mounted
  as a read-only secret so live cycles work inside the container.

Status note: the Docker files are complete and validated (compose parses, all COPY
targets exist, a clean venv equivalent to the container boots both services), but
the literal `docker compose up --build` step could not be executed on the machine
that produced this archive (no Docker engine available). It is expected to work on
any Docker-capable host (Docker Desktop/WSL2, Linux, or a build server).

---

## 7. Fetching the ERA5 data (optional, for fresh/validation runs)

The raw fields are excluded from this archive to keep it small. To re-fetch:

```python
from weather_pipeline.download import download_era5
path = download_era5("amphan")   # writes data/raw/era5_amphan_test.nc
# live (needs ~/.cdsapirc): download_live() / scripts.live_run
```

Variables used: geopotential, temperature, u/v wind, specific humidity at
multi-level ERA5 (850 hPa focused), 0.25-deg, 6-hourly, Bay of Bengal box
(~5-30N, 75-100E).

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `[Errno 10048]` port in use | an API instance is already running — stop it, or use `--port` |
| `CDS credentials not configured` | create `~/.cdsapirc` (free Copernicus account) |
| torch-geometric import error | CPU wheels for the graph layers are optional; the package auto-falls back to pure-torch scatter (works, slightly slower) |
| no active alerts on Overview | nothing confirmed yet — confirm something in the Review Queue |
| dashboard shows 0 pending | all alerts already reviewed; run a fresh cycle or reset `outputs/review_state.json` |