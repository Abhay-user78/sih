"""Phase 4: wire Phase 2 (GNN anomaly detection + bounding regions) and
Phase 3 (conditional diffusion downscaling) together end-to-end, fully
automated, on the Amphan test case.

For every forecast timestep:
  1. GNN scores the icosahedral-mesh nodes -> anomaly probabilities.
  2. `bounding_region` clusters hotspots into a macro bbox + centre.
  3. The coarse-resolution field is re-cropped *centred on the GNN
     centre* (no manual intervention) and fed as conditioning into the
     diffusion downscaler (mode: emulate the 12-km forecast by block-mean
     + bilinear upsample of the 0.25-deg field).
  4. An ensemble of stochastic sampling passes gives a sharpened 5-km-
     class field + a severity probability, WITHOUT smoothing the peak.
  5. An alert record is assembled (lat/lon centre, bbox, radius, severity,
     confidence) and written to outputs/phase4_alerts.json.

Verification inside the driver: the pipeline-reported centre still tracks
the true cyclone (low km error), and the downscaled peak preserves (or
exceeds) the coarse peak.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, asdict

import numpy as np
import torch

from .config import CONFIG, DATA_PROCESSED, OUTPUTS
from . import gnn, diffusion
from .data_features import prepare_dataset
from .downscale_data import _block_mean, _bilinear2
from .synthetic import build_synthetic_amphan


@dataclass
class Alert:
    id: str
    timestep_index: int
    timestep: str
    centre_lat: float
    centre_lon: float
    bbox: list
    radius_km: float
    severity: str
    confidence: float
    coarse_peak: float
    downscaled_peak: float
    peak_gain: float
    n_ensemble: int
    severity_prob_mean: float
    gnn_mean_score: float
    gnn_max_score: float


SEVERITY_ORDER = ["low", "moderate", "severe"]


def detection_confidence(gnn_mean_score, members_phys, coarse_peak):
    """Detection confidence — the single definition used by every alert
    builder (pipeline, operational live cycle, phase_era5 validation).

    confidence = 0.5 * GNN cluster mean anomaly score
               + 0.5 * ensemble agreement,
    where agreement is the fraction of downscaled ensemble members whose
    peak stays at or above the coarse peak. This is a *detection*
    magnitude on [0, 1] — deliberately NOT an inverted severity
    probability, so a weak anomaly shows a low confidence and a strong,
    sharpened core shows a high one.
    """
    frac_agreement = float(
        (np.abs(members_phys[:, 0]).max(axis=(1, 2)) >= coarse_peak).mean())
    return float(np.clip(
        0.5 * float(gnn_mean_score) + 0.5 * np.clip(frac_agreement, 0, 1),
        0, 1))


def grid_crop(field: np.ndarray, lat: np.ndarray, lon: np.ndarray,
              centre: tuple, size: int = 48) -> dict:
    """(y0,x0)-clamped 2-D crop centred as close to (lat,lon) as the grid
    allows; returns the crop plus its true centre."""
    ci_lat = int(np.argmin(np.abs(lat - centre[0])))
    ci_lon = int(np.argmin(np.abs(lon - centre[1])))
    y0 = int(min(max(ci_lat - size // 2, 0), len(lat) - size))
    x0 = int(min(max(ci_lon - size // 2, 0), len(lon) - size))
    return {
        "crop": field[y0:y0 + size, x0:x0 + size].astype(np.float32),
        "lat": lat[y0:y0 + size], "lon": lon[x0:x0 + size],
        "y0": y0, "x0": x0,
        "crop_centre": (float(lat[y0 + size // 2]),
                        float(lon[x0 + size // 2])),
    }


def _norm(coarse_np: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    tr = vmax - vmin
    return np.clip((np.asarray(coarse_np) - vmin) / (tr + 1e-9) * 2 - 1,
                   -1, 1)


def inv_norm(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return (np.asarray(x) + 1) / 2 * (vmax - vmin) + vmin


def _radius_km(centre_lat: float, centre_lon: float, bbox) -> float:
    """Approx radius from centre to farthest bbox corner (km)."""
    lat = np.deg2rad(np.array([centre_lat, bbox[0], bbox[1]]))
    lon = np.deg2rad(np.array([centre_lon, bbox[2], bbox[3]]))
    pts = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                    np.sin(lat)], axis=-1) * 6371.0
    return float(np.max(np.linalg.norm(pts[1:] - pts[0], axis=1)))


def severity_label(sp: np.ndarray, base: float, thr_mid: float = 0.55,
                   thr_hi: float = 0.80) -> str:
    val = min(max(sp, base), 1.0)
    return "severe" if val >= thr_hi else \
        ("moderate" if val >= thr_mid else "low")


class ThreatPipeline:
    """End-to-end GNN -> diffusion detector, one instance per Dataset."""

    def __init__(self, ds=None, variable="geopotential", level=850,
                 size: int = 48, scale: int = 2, n_steps: int = 100,
                 device: str | None = None):
        ds = ds or build_synthetic_amphan(save=True)
        self.ds = ds
        self.var, self.level = variable, level
        self.size, self.scale = size, scale
        self.n_steps = n_steps
        self.device = (device or CONFIG.device)
        self.device = self.device if torch.cuda.is_available() else "cpu"
        self.lat = ds.latitude.values.astype(float)
        self.lon = np.mod(ds.longitude.values.astype(float), 360.0)
        self.n_time = int(ds.sizes["time"])

        self.dataset = prepare_dataset(ds)
        ck = torch.load(model_dir_gnn(), map_location="cpu",
                        weights_only=False)
        self.gnn = gnn.AnomalyGNN(self.dataset["feature_dim"],
                                  hidden=ck["hidden"])
        self.gnn.load_state_dict(ck["state"])

        self.unet, self.sched = diffusion.load_downscale(self.device)

        dl = np.load(DATA_PROCESSED / "phase3_dataset.npz")
        self._vmin = float(dl["stats_vmin"])
        self._vmax = float(dl["stats_vmax"])

    def field_t(self, t: int) -> np.ndarray:
        da = self.ds[self.var].sel(level=self.level)
        if "time" in da.dims:
            return np.asarray(da.isel(time=t).values, dtype=float)
        return np.asarray(da.values, dtype=float)

    def run_timestep(self, t: int, n_members: int = 24,
                     seed: int = 0) -> dict:
        """Full automated chain for one timestep -> alert + diagnostics."""
        probs = gnn.predict(self.dataset, self.gnn, t, t + 1,
                            device=self.device)[0]
        reg = gnn.bounding_region(probs,
                                  node_lat=self.dataset["node_lat"],
                                  node_lon=self.dataset["node_lon"])
        ts = str(self.ds.time.values[t]) if "time" in self.ds.dims else f"t{t}"

        if not reg.get("centre"):
            return {"timestep_index": t, "timestep": ts,
                    "detected": False, "n_hits": reg["n_hits"],
                    "thr": reg["thr"]}

        field = self.field_t(t)
        gc = grid_crop(field, self.lat, self.lon,
                       (reg["centre"]["lat"], reg["centre"]["lon"]),
                       self.size)
        coarse = _bilinear2(_block_mean(gc["crop"], self.scale),
                            self.size, self.size)
        cond = _norm(coarse, self._vmin, self._vmax)[None, None]

        members = diffusion.sample_crop(self.unet, self.sched, cond,
                                        n_members=n_members,
                                        n_steps=self.n_steps,
                                        seed=seed, device=self.device)
        m_phys = inv_norm(members, self._vmin, self._vmax)
        sev = diffusion.severity_from_ensemble(members, cond[0, 0])

        peak_coarse = float(np.abs(coarse).max())
        peak_mean = float(np.abs(m_phys.mean(axis=0)[0]).max())
        peak_gain = peak_mean / max(peak_coarse, 1e-9)

        gnn_ms = float(np.asarray(reg["clusters"][0]["mean_score"]))
        gnn_mx = float(np.asarray(reg["clusters"][0]["max_score"]))
        conf = detection_confidence(gnn_ms, m_phys, peak_coarse)
        spm = float(sev["severity_prob"].mean())

        bbox = [float(v) for v in reg["bbox"]]
        centre = (float(reg["centre"]["lat"]),
                  float(reg["centre"]["lon"]))
        radius = _radius_km(centre[0], centre[1], bbox)

        alert = Alert(
            id=f"amphan-2020-{ts[:19].replace(':', '')}",
            timestep_index=t, timestep=ts,
            centre_lat=centre[0], centre_lon=centre[1],
            bbox=bbox, radius_km=round(radius, 1),
            severity=severity_label(
                float(sev["severity_prob"].max()), base=0.5),
            confidence=round(conf, 3),
            coarse_peak=round(peak_coarse, 1),
            downscaled_peak=round(peak_mean, 1),
            peak_gain=round(peak_gain, 3),
            n_ensemble=n_members,
            severity_prob_mean=round(spm, 4),
            gnn_mean_score=round(gnn_ms, 3),
            gnn_max_score=round(gnn_mx, 3),
        )
        return {
            "detected": True, "alert": asdict(alert),
            "probs": probs, "region": reg,
            "members": m_phys, "severity": sev,
            "crop": {"lat": gc["lat"], "lon": gc["lon"],
                     "coarse": coarse, "fine": gc["crop"],
                     "centre": gc["crop_centre"]},
        }

    def run(self, n_members: int = 24, start: int = 0,
            end: int | None = None, cache: bool = True,
            force: bool = False) -> list[dict]:
        end = end or self.n_time
        cachedir = OUTPUTS / "phase4_cache"
        cachedir.mkdir(exist_ok=True)
        results = []
        for t in range(start, end):
            cp = cachedir / f"t{t:03d}.pkl"
            if cache and cp.exists() and not force:
                with open(cp, "rb") as fh:
                    r = pickle.load(fh)
            else:
                r = self.run_timestep(t, n_members=n_members)
                for k in ("probs", "region"):
                    r.pop(k, None)          # keep cache lean
                if cache:
                    with open(cp, "wb") as fh:
                        pickle.dump(r, fh)
            results.append(r)
            if r["detected"]:
                a = r["alert"]
                print(f"  t={t:2d} detected  centre "
                      f"({a['centre_lat']:.2f},{a['centre_lon']:.2f})  "
                      f"sev={a['severity']:<8s}  conf={a['confidence']:.2f}  "
                      f"peak {a['coarse_peak']:.0f}->{a['downscaled_peak']:.0f}")
            else:
                print(f"  t={t:2d} not detected")
        return results


def model_dir_gnn():
    from .config import model_dir
    return model_dir("gnn") / "gnn_anomaly.pt"


def write_report(results: list[dict], track_centre: np.ndarray,
                 path=None) -> str:
    """Assemble a STOP-AND-VERIFY report with centre-error + peak stats."""
    det = [r for r in results if r["detected"]]
    lines = ["PHASE 4 STOP-AND-VERIFY (GNN -> diffusion end-to-end)",
             f"  timesteps: {len(det)}/{len(results)} auto-detected"]
    if det:
        errs = []
        for r in det:
            a = r["alert"]
            d = _km(a["centre_lat"], a["centre_lon"],
                    float(track_centre[r["alert"]["timestep_index"]][0]),
                    float(track_centre[r["alert"]["timestep_index"]][1]))
            errs.append(d)
        gains = [a["alert"]["peak_gain"] for a in det]
        lines += [
            f"  median true-centre error: {np.median(errs):.1f} km "
            f"(phase2 was 9.3 km)",
            f"  peak-gain (downscale/coarse): median "
            f"{np.median(gains):.3f} "
            f"(>=0.95 ok: the ensemble-mean peak must not be smoothed away)",
            f"  alerts: {len(det)}",
        ]
        for a in det:
            al = a["alert"]
            lines.append(
                f"    t={al['timestep_index']:2d} {al['timestep']}  "
                f"({al['centre_lat']:.2f},{al['centre_lon']:.2f})  "
                f"sev={al['severity']:<8s} conf={al['confidence']:.2f} "
                f"r={al['radius_km']:.0f}km")
    text = "\n".join(lines)
    p = path or OUTPUTS / "phase4_report.txt"
    open(p, "w").write(text + "\n")
    return text


def _km(lat1, lon1, lat2, lon2):
    a = np.sin(np.deg2rad(lat2 - lat1) / 2) ** 2 + \
        np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * \
        np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))