"""Real cyclone observations.

`AMPHAN_TRACK` is the IMD best-track for Cyclone Amphan (IMD RSMC New Delhi
"Super Cyclonic Storm AMPHAN over the Bay of Bengal 16-21 May 2020"),
resampled to the same 24 x 6-hourly grid as the ERA5 download
(16 May 00Z = index 0 ... 21 May 18Z = index 23).

Knots are the IMD best-track fix positions on 6-hourly boundaries
(00/06/12/18Z), matching the ERA5 sample grid; a cubic spline fills any
intermediate steps. Landfall occurred 20 May ~12Z (≈21.7N 88.3E, Sundarbans).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

KNOTS = [
    # (hours since 16-May-00Z, lat N, lon E)
    # IMD best-track rows for Super Cyclonic Storm Amphan (16-21 May 2020),
    # taken on the 6-hourly boundaries that match the ERA5 sample grid.
    (0, 10.4, 87.0),
    (6, 10.9, 86.3),
    (12, 10.9, 86.3),
    (18, 11.1, 86.1),
    (24, 11.4, 86.0),
    (30, 11.5, 86.0),
    (36, 12.0, 86.0),
    (42, 12.5, 86.1),
    (48, 13.2, 86.3),
    (54, 13.4, 86.2),
    (60, 14.0, 86.3),
    (66, 14.9, 86.5),
    (72, 15.6, 86.7),
    (78, 16.5, 86.9),
    (84, 17.4, 87.0),
    (90, 18.4, 87.2),
    (96, 19.1, 87.5),
    (102, 20.6, 88.0),
    (108, 21.9, 88.4),   # landfall near Sundarbans 20 May 12Z
    (114, 23.3, 89.0),
    (120, 24.2, 89.3),
    (126, 25.0, 89.6),
    (132, 25.7, 89.8),
    (138, 26.0, 89.9),
]


def amphan_track(n_steps: int = 24, step_hours: int = 6) -> np.ndarray:
    """(n_steps, 2) lat/lon best-track on the 6-hourly ERA5 grid.

    The ERA5 test download samples 4x/day over 16-21 May (24 steps), so
    step_hours = 6 by default.
    """
    h = np.asarray([k[0] for k in KNOTS], dtype=float)
    lat = np.asarray([k[1] for k in KNOTS], dtype=float)
    lon = np.asarray([k[2] for k in KNOTS], dtype=float)
    target = np.arange(n_steps) * step_hours
    lat_q = CubicSpline(h, lat)(target)
    lon_q = CubicSpline(h, lon)(target)
    return np.stack([lat_q, lon_q], axis=1)


AMPHAN_TRACK = amphan_track(24)


# --------------------------------------------------------------------------
# Additional events (multi-event validation)
# Knots are IMD RSMC New Delhi best-track fixes on 6-hourly boundaries
# (00/06/12/18Z) matching the ERA5 sample grid, from the published
# "Summary of Cyclonic Disturbances over the North Indian Ocean" PDFs.
# --------------------------------------------------------------------------

YAAS_KNOTS = [
    # Very Severe Cyclonic Storm YAAS, 23-28 May 2021 (landfall near
    # Balasore, Odisha, 26 May ~06Z). IMD best-track Table (6-hourly rows).
    (0, 15.5, 90.0),     # 23/00Z
    (6, 16.1, 90.2),     # 23/06Z
    (12, 16.2, 89.9),
    (18, 16.3, 89.7),
    (24, 16.3, 89.7),
    (30, 16.4, 89.6),
    (36, 17.1, 89.3),
    (42, 17.6, 89.0),
    (48, 18.0, 88.6),
    (54, 18.7, 88.0),
    (60, 19.5, 88.0),
    (66, 20.1, 87.8),
    (72, 20.8, 87.3),
    (78, 21.4, 86.9),    # 26/06Z: crossed coast near 21.35N 86.95E
    (84, 21.8, 86.6),
    (90, 22.5, 86.0),
    (96, 22.8, 85.8),
    (102, 23.5, 85.6),
    (108, 24.3, 85.3),
    (114, 24.7, 84.8),   # 27/18Z (weakened to D)
]


def yaas_track(n_steps: int = 20, step_hours: int = 6) -> np.ndarray:
    """(n_steps, 2) lat/lon IMD best-track for Cyclone Yaas (May 2021)."""
    h = np.asarray([k[0] for k in YAAS_KNOTS], dtype=float)
    lat = np.asarray([k[1] for k in YAAS_KNOTS], dtype=float)
    lon = np.asarray([k[2] for k in YAAS_KNOTS], dtype=float)
    target = np.arange(n_steps) * step_hours
    return np.stack([CubicSpline(h, lat)(target),
                     CubicSpline(h, lon)(target)], axis=1)


YAAS_TRACK = yaas_track(20)


FANI_KNOTS = [
    # Extremely Severe Cyclonic Storm FANI, 26 Apr-04 May 2019 (landfall
    # near Puri, Odisha, 03 May ~03Z). IMD best-track rows, sampled from
    # 28/00Z when the system was already a CS inside the AOI.
    (0, 7.3, 87.9),      # 28/00Z
    (6, 7.4, 87.8),
    (12, 8.2, 87.0),
    (18, 8.4, 86.9),
    (24, 8.6, 86.9),     # 29/00Z
    (30, 9.2, 86.9),
    (36, 10.1, 86.7),    # 29/12Z SCS
    (42, 10.8, 86.6),
    (48, 11.7, 86.5),    # 30/00Z VSCS
    (54, 12.6, 85.7),
    (60, 13.3, 84.7),    # 30/12Z ESCS
    (66, 13.5, 84.4),
    (72, 13.9, 84.0),    # 01/00Z
    (78, 14.2, 83.9),
    (84, 14.9, 84.1),
    (90, 15.5, 84.2),
    (96, 15.9, 84.5),    # 02/00Z
    (102, 16.7, 84.8),
    (108, 17.5, 84.8),
    (114, 18.2, 85.0),
    (120, 19.1, 85.5),   # 03/00Z
    (126, 20.2, 85.9),   # 03/06Z: crossed near Puri (19.75N 85.7E)
    (132, 21.1, 86.5),
    (138, 21.9, 87.1),
    (144, 23.1, 88.2),   # 04/00Z
]


def fani_track(n_steps: int = 24, step_hours: int = 6) -> np.ndarray:
    """(n_steps, 2) lat/lon IMD best-track for Cyclone Fani (May 2019)."""
    h = np.asarray([k[0] for k in FANI_KNOTS], dtype=float)
    lat = np.asarray([k[1] for k in FANI_KNOTS], dtype=float)
    lon = np.asarray([k[2] for k in FANI_KNOTS], dtype=float)
    target = np.arange(n_steps) * step_hours
    return np.stack([CubicSpline(h, lat)(target),
                     CubicSpline(h, lon)(target)], axis=1)


FANI_TRACK = fani_track(24)


# Event registry: event name -> IMD best track (already sampled on the
# 6-hourly ERA5 grid), requested ERA5 date range and [N, W, S, E] bbox.
EVENTS = {
    "amphan": {"track": AMPHAN_TRACK, "start": "2020-05-16",
               "end": "2020-05-21", "area": [30, 75, 5, 100],
               "knots": len(KNOTS)},
    "yaas": {"track": YAAS_TRACK, "start": "2021-05-23",
             "end": "2021-05-27", "area": [26, 84, 15, 91],
             "knots": len(YAAS_KNOTS)},
    "fani": {"track": FANI_TRACK, "start": "2019-04-28",
             "end": "2019-05-03", "area": [26, 82, 7, 91],
             "knots": len(FANI_KNOTS)},
}