"""Phase 1d: Extreme Forecast Index (EFI).

Definition (ECMWF, Anderson-Darling weighted variant):

    EFI = (2/pi) * INT_0^1 [ p - F_f(p) ] / sqrt(p(1-p)) dp

where F_f(p) is the proportion of forecast members below the p-quantile of
the reference climate record, and 1/sqrt(p(1-p)) downweights the centre of
the distribution (giving maximum weight to the tails).

For a DETERMINISTIC forecast occupying climate percentile p0 (the case we
need for single-scenario ERA5 fields) the integral closes analytically:

    EFI(p0) = (4/pi)*asin(sqrt(p0)) - 1,     p0 in [0,1], EFI in [-1,1]

Sanity checks: p0=0.5 -> 0; p0=1.0 -> +1 (all-time record high);
p0=0.0 -> -1 (all-time record low).

Threshold convention (ECMWF): |EFI| in [0.5, 0.8] unusual,
|EFI| > 0.8 very unusual/extreme.

Reference: Lalaurette (2002) "Early detection of abnormal weather using a
probabilistic Extreme Forecast Index"; ECMWF Forecast User Guide 8.1.9.2.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def efi_from_percentile(p0) -> np.ndarray:
    """EFI for a deterministic value located at climate percentile p0."""
    p0 = np.clip(np.asarray(p0, dtype=float), 0.0, 1.0)
    return (4.0 / np.pi) * np.arcsin(np.sqrt(p0)) - 1.0


def efi_numerical(Ff_p: np.ndarray, p: np.ndarray) -> float:
    """Numerical EFI via the general integral (for ensemble forecasts later).

    Ff_p[i] is the proportion of forecast members below the p[i] quantile of
    climate. Used when the forecast is an ensemble (NEPS-G).
    """
    p = np.asarray(p, dtype=float)
    Ff = np.asarray(Ff_p, dtype=float)
    ok = (p > 0) & (p < 1)
    integrand = (p - Ff)[ok] / np.sqrt((p * (1 - p))[ok])
    return float((2.0 / np.pi) * np.trapezoid(integrand, p[ok]))


def compute_efi(current: np.ndarray, climatology: np.ndarray,
                axis: int = 0) -> np.ndarray:
    """EFI field comparing `current` (shape S...) against `climatology`
    (shape (..., N) along `axis`) for every spatial point.

    current shape must broadcast over the climatology spatial dims. Uses the
    deterministic closed form with p0 = empirical percentile of the current
    value within the climate record at that location.
    """
    c = np.moveaxis(np.asarray(climatology, dtype=float), axis, -1)  # (S..., N)
    cur = np.asarray(current, dtype=float)                           # (S...)
    nitems = c.shape[-1]

    combined = np.concatenate([c, cur[..., None]], axis=-1)  # (S..., N+1)
    ranks = rankdata(combined, axis=-1, method="average")[..., -1]  # (S...)
    p0 = (ranks - 1.0) / nitems   # fraction of climate samples below current
    return efi_from_percentile(p0)


def severity(efi: np.ndarray,
             unusual: float = 0.5,
             extreme: float = 0.8) -> np.ndarray:
    """Map |EFI| to categorical severity per point (same shape as input).

    Returns 0=normal, 1=unusual, 2=severe. Sign of EFI preserved separately
    by callers if needed (+ above climatology, - below).
    """
    a = np.abs(efi)
    out = np.zeros(np.shape(efi), dtype=np.int8)
    out[a >= extreme] = 2
    out[(a >= unusual) & (a < extreme)] = 1
    return out