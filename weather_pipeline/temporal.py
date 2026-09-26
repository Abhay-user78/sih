"""Phase 5: Temporal transformer for dynamic 4D bounding-box tracking.

Consumes the per-timestep GNN outputs (a sequence of detected anomaly
regions) and produces a *dynamic 4D bounding box* (lat, lon, level, time)
that tracks how the anomaly moves — outputing both a smoothed current
box and a forecast of the future track.

Architecture (torch.nn.TransformerEncoder, per the brief):
  * token feature = [lat, lon, level, score, span, miss-flag,
                     hour-of-run] (normalised)
  * positional encoding = forecast offset (sinusoidal)
  * causal self-attention, then two heads:
      - `smooth` head: refined current box      (target = true box)
      - `forecast` head: one-step-ahead box     (target = box at t+1)
  * multi-step rollouts at inference produce the full track forecast, from
    which the operator keeps the first K steps as the "tracked" 4D box
    sequence and the rest as the forecast horizon.

Training data: parameterised synthetic cyclone events over the Bay of
Bengal that mirror Amphan-like physics (NW drift ~3-6 m/s, recurvature,
intensity envelope) PLUS simulated GNN detection noise, so the layer can
be validated against truth. On deployment it runs unchanged on the real
GNN detection sequence (e.g. outputs/phase4_alerts.json).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CONFIG, model_dir


# ---------------------------------------------------------------- data --
def generate_events(n_events: int = 60, n_steps: int = 24,
                    seed: int = 0, dt_h: float = 6.0) -> np.ndarray:
    """Synthetic cyclone tracks as (E, T, 2) lat/lon arrays.

    Each event starts in the SE Bay (lat 7.5-12.5, lon 88-94), moves
    north-west with speed 3-6 m/s and slight recurvature, wrapped to the
    domain [5, 30] x [75, 100].
    """
    rng = np.random.default_rng(seed)
    E = n_events
    sec_per_step = dt_h * 3600.0
    out = np.zeros((E, n_steps, 2))
    for e in range(E):
        lat0 = rng.uniform(7.5, 12.5)
        lon0 = rng.uniform(88.0, 94.0)
        speed = rng.uniform(3.0, 6.0)                      # m/s
        head0 = rng.uniform(300.0, 330.0)                  # deg (toward NW)
        turn = rng.uniform(0.4, 2.2)                       # deg/step^2
        turn_dir = rng.choice([-1.0, 1.0])
        wob = rng.uniform(0.4, 1.2)                        # deg amplitude
        wob_p = rng.uniform(0, 2 * np.pi)                  # phase
        lat, lon = lat0, lon0
        for t in range(n_steps):
            out[e, t] = (lat, lon)
            heading = head0 + turn_dir * turn * t
            dist = speed * sec_per_step                      # metres
            d_lat = dist / 111320.0 * np.cos(np.deg2rad(heading))
            d_lon = dist / (111320.0 * max(np.cos(np.deg2rad(lat)), 0.1)) \
                * np.sin(np.deg2rad(heading))
            lat = np.clip(lat + d_lat + wob * np.sin(t / 4.0 + wob_p),
                          5.0, 30.0)
            lon = np.clip(np.mod(lon + d_lon, 360.0), 75.0, 100.0)
    return out


def add_detection_noise(tracks: np.ndarray, km: float = 15.0,
                        miss_p: float = 0.0,
                        seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt truth track with GNN-like detection noise (±km) and misses.

    Returns (detections (E,T,2), miss_mask (E,T) in {0,1}).
    """
    rng = np.random.default_rng(seed)
    dlat = km / 111.0
    dlon = km / (111.0 * np.cos(np.deg2rad(np.clip(tracks[..., 0], -60, 60)))
                 + 1e-9)
    det = tracks.copy()
    det[..., 0] += rng.normal(0, dlat, tracks.shape[:-1])
    det[..., 1] += rng.normal(0, dlon, tracks.shape[:-1])
    miss = (rng.random(tracks.shape[:-1]) < miss_p).astype(np.float32)
    return det, miss


class TrackSet:
    """Normalised sequences from detection streams."""

    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 5.0, 30.0, 75.0, 100.0

    def __init__(self, detections: np.ndarray, miss: np.ndarray,
                 scores: np.ndarray | None = None,
                 missing_score: float = 0.0):
        self.det = detections
        self.miss = miss
        self.scores = scores
        self.missing_score = missing_score

    def lat_n(self, x) -> np.ndarray:
        return (x[..., 0] - self.LAT_MIN) / (self.LAT_MAX - self.LAT_MIN) * 2 - 1

    def lon_n(self, x) -> np.ndarray:
        return (x[..., 1] - self.LON_MIN) / (self.LON_MAX - self.LON_MIN) * 2 - 1

    def tokens(self) -> np.ndarray:
        """(E, T, D) token features incl. velocity deltas."""
        E, T, _ = self.det.shape
        D = 8
        tok = np.zeros((E, T, D), dtype=np.float32)
        tok[..., 0] = self.lat_n(self.det)
        tok[..., 1] = self.lon_n(self.det)
        tok[..., 2] = 850.0 / 1000.0                    # level (constant)
        if self.scores is None:
            tok[..., 3] = 1.0 - self.miss               # detection confidence
        else:
            tok[..., 3] = self.scores                   # mean GNN score
        tok[..., 4] = self.miss                          # missing flag
        tok[..., 5] = np.arange(T)[None] / float(T)      # hour-of-run
        d = np.diff(self.det, axis=1, prepend=self.det[:, :1])
        tok[..., 6] = d[..., 0] / (self.LAT_MAX - self.LAT_MIN) * 2
        tok[..., 7] = d[..., 1] / (self.LON_MAX - self.LON_MIN) * 2
        return tok

    def targets(self, tracks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(smooth_target, forecast_target) as (E,T,2) normalised."""
        st = np.stack([self.lat_n(tracks), self.lon_n(tracks)], axis=-1)
        ft = np.zeros_like(st)
        ft[:, :-1] = st[:, 1:]
        ft[:, -1] = st[:, -1]
        return st, ft


# -------------------------------------------------------------- model ---
def _pos_emb(T: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(T, d_model)
    idx = torch.arange(T)[:, None].float()
    k = torch.pow(10000.0, -torch.arange(0, d_model, 2).float() / d_model)
    pe[:, 0::2] = torch.sin(idx * k)
    pe[:, 1::2] = torch.cos(idx * k)
    return pe


class TemporalTracker(nn.Module):
    def __init__(self, in_dim: int = 8, d_model: int = 64, nhead: int = 4,
                 n_layers: int = 2, ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.drop = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                         batch_first=True)
        self.enc = nn.TransformerEncoder(enc, n_layers)
        self.smooth = nn.Linear(d_model, 2)       # residual denoise correction
        self.forecast = nn.Linear(d_model, 2)     # one-step-ahead box

    def _mask(self, T: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device) * float("-inf"),
                          diagonal=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (E, T, in_dim) -> (smooth_box, forecast_box), each (E,T,2).

        `smooth` is a residual correction added to the observed (noisy)
        position, so an uninformative model falls back exactly to the raw
        detection instead of degrading it.
        """
        E, T, _ = x.shape
        h = self.embed(x)
        h = h + _pos_emb(T, self.embed.out_features).to(x.device)
        h = self.drop(h)
        h = self.enc(h, mask=self._mask(T, x.device))
        sm = self.smooth(h) + x[..., :2]
        return sm, self.forecast(h)


# ----------------------------------------------------------- training ---
def train_tracker(events_train: TrackSet, tracks_train: np.ndarray,
                  events_val: TrackSet, tracks_val: np.ndarray,
                  epochs: int = 120, lr: float = 3e-4, wd: float = 1e-4,
                  α: float = 0.45, d_model: int = 96, n_layers: int = 3,
                  device: str | None = None,
                  print_every: int = 20) -> tuple[object, dict]:
    device = (device or CONFIG.device)
    device = device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    xt = torch.tensor(events_train.tokens(), device=device)
    st_t, ft_t = events_train.targets(tracks_train)
    st_t = torch.tensor(st_t, device=device)
    ft_t = torch.tensor(ft_t, device=device)
    xv = torch.tensor(events_val.tokens(), device=device)
    st_v, ft_v = events_val.targets(tracks_val)
    st_v = torch.tensor(st_v, device=device)
    ft_v = torch.tensor(ft_v, device=device)

    model = TemporalTracker(d_model=d_model, n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    hist = {"tr": [], "va": []}
    for ep in range(1, epochs + 1):
        model.train()
        sm, fc = model(xt)
        lft = F.l1_loss(fc[:, :-1], ft_t[:, :-1])
        lst = F.l1_loss(sm, st_t)
        loss = α * lft + (1 - α) * lst
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            smv, fcv = model(xv)
            lftv = F.l1_loss(fcv[:, :-1], ft_v[:, :-1])
            lstv = F.l1_loss(smv, st_v)
            lossv = α * lftv + (1 - α) * lstv
        hist["tr"].append(float(loss.item())); hist["va"].append(float(lossv))
        if ep % print_every == 0 or ep == epochs:
            print(f"  ep {ep:3d}  train {loss:.4f}  val {lossv:.4f}")

    torch.save({"state": model.state_dict(), "α": α, "n_epochs": epochs,
                "in_dim": 8, "d_model": d_model, "nhead": 4,
                "n_layers": n_layers,
                "hist": hist}, model_dir("temporal") / "temporal_tracker.pt")
    return model, hist


def rollout_forecast(model: TemporalTracker, det: np.ndarray,
                     miss: np.ndarray, scores=None, warmup: int = 6,
                     max_delta_norm: float = 0.10,
                     device: str | None = None) -> np.ndarray:
    """Autoregressive multi-step forecast of the track.

    Uses the first `warmup` detections (smoothed in-place via the `smooth`
    head), then iteratively predicts the next position with the `forecast`
    head, feeding outputs back as inputs. Per-step motion is clamped to
    `max_delta_norm` (normalised units) to keep rollouts physical.
    Returns (T, 2) deg lat/lon boxes; first `warmup` entries are
    smooth-box refinements, the rest are forecasts.
    """
    device = (device or CONFIG.device)
    device = device if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    LMIN, LMAX = TrackSet.LAT_MIN, TrackSet.LAT_MAX
    LN_MIN, LN_MAX = TrackSet.LON_MIN, TrackSet.LON_MAX
    det = np.asarray(det, dtype=float).copy()
    miss = np.asarray(miss, dtype=float).copy()
    if scores is None:
        scores = 1.0 - miss
    T = det.shape[0]
    tok = np.zeros((T, 8), dtype=np.float32)
    tok[:, 0] = (det[:, 0] - LMIN) / (LMAX - LMIN) * 2 - 1
    tok[:, 1] = (det[:, 1] - LN_MIN) / (LN_MAX - LN_MIN) * 2 - 1
    tok[:, 2] = 850.0 / 1000.0
    tok[:, 3] = np.asarray(scores, dtype=float)
    tok[:, 4] = miss
    tok[:, 5] = np.arange(T) / float(T)
    tok[1:, 6] = np.diff(tok[:, 0])
    tok[1:, 7] = np.diff(tok[:, 1])
    boxes = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        tok[:t + 1, 3] = scores[:t + 1]        # keep score/confidence current
        tok[:t + 1, 4] = miss[:t + 1]
        x = torch.tensor(tok[:t + 1][None], device=device)
        with torch.no_grad():
            sm, fc = model(x)
        if t < warmup:
            b_n = sm[0, t].cpu().numpy()
        else:
            b_n = fc[0, t - 1].cpu().numpy()
            # clamp per-step motion so the rollout cannot diverge
            dlat = np.clip(b_n[0] - tok[t, 0], -max_delta_norm,
                           max_delta_norm)
            dlon = np.clip(b_n[1] - tok[t, 1], -max_delta_norm,
                           max_delta_norm)
            b_n = np.array([tok[t, 0] + dlat, tok[t, 1] + dlon])
            # write predicted position into the next token input
            if t + 1 < T:
                tok[t + 1, 0] = b_n[0]; tok[t + 1, 1] = b_n[1]
                tok[t + 1, 6] = b_n[0] - tok[t, 0]
                tok[t + 1, 7] = b_n[1] - tok[t, 1]
        boxes[t] = b_n

    lat = (boxes[:, 0] + 1) / 2 * (LMAX - LMIN) + LMIN
    lon = (boxes[:, 1] + 1) / 2 * (LN_MAX - LN_MIN) + LN_MIN
    return np.stack([lat, lon], axis=1)


def _km(lat1, lon1, lat2, lon2):
    a = np.sin(np.deg2rad(lat2 - lat1) / 2) ** 2 + \
        np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * \
        np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def evaluate(model, events: TrackSet, tracks: np.ndarray, warmup: int = 6,
             device=None, horizons=(1, 3, 6)) -> dict:
    """Forecast MAE (km) at several horizons vs truth + persistence.

    Persistence = hold the *last known* (warmup) detection position.
    Also reports the tracker's per-step smoothing error vs raw detection
    noise during the observed window.
    """
    model.to(device or "cpu").eval()
    E, T, _ = tracks.shape
    ah = {}
    ap = {}
    for h in horizons:
        ah[h], ap[h] = [], []
    sm_ok, det_ok = [], []
    for e in range(E):
        pred = rollout_forecast(model, events.det[e], events.miss[e],
                                events.scores[e] if events.scores is not None
                                else None, warmup=warmup, device=device)
        for h in horizons:
            j = warmup + h
            if j >= T:
                continue
            d_pred = _km(pred[j, 0], pred[j, 1], tracks[e, j, 0],
                         tracks[e, j, 1])
            d_per = _km(events.det[e, warmup, 0], events.det[e, warmup, 1],
                        tracks[e, j, 0], tracks[e, j, 1])
            ah[h].append(d_pred); ap[h].append(d_per)
        for j in range(warmup):
            d_sm = _km(pred[j, 0], pred[j, 1], tracks[e, j, 0],
                       tracks[e, j, 1])
            d_det = _km(events.det[e, j, 0], events.det[e, j, 1],
                        tracks[e, j, 0], tracks[e, j, 1])
            sm_ok.append(d_sm); det_ok.append(d_det)
    return {
        h: {"median_km": float(np.median(ah[h])),
            "persistence_km": float(np.median(ap[h]))}
        for h in horizons if ah[h]
    } | {"smooth_median_km": float(np.median(sm_ok)),
         "detection_median_km": float(np.median(det_ok))}


def load_tracker(device: str | None = None, path=None):
    ck = torch.load(path or (model_dir("temporal") / "temporal_tracker.pt"),
                    map_location="cpu", weights_only=False)
    m = TemporalTracker(in_dim=ck["in_dim"], d_model=ck["d_model"],
                        nhead=ck["nhead"], n_layers=ck["n_layers"])
    m.load_state_dict(ck["state"])
    return m, ck