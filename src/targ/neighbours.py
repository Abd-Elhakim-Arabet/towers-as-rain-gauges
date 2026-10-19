"""Neighbour check: a link may only report rain if a nearby link also sees rain.

Rain cells are kilometres wide, so real rain on one link almost always shows on
another link close by within minutes. A link that reports rain while every one of
its neighbours that has data is dry is more likely broken (equipment, a sudden
level jump) than wet. This is the idea behind RAINLINK's nearby-link filter
(Overeem et al. 2016), simplified.

Rule (fixed in advance, not tuned on test data):
  neighbours   other paths whose midpoints lie within RADIUS_KM (the two directions
               of one physical link do not count as each other's neighbours)
  support      at least one neighbour reports rain (>= 0.1 mm/h) within
               +-WINDOW_MIN minutes
  action       rain without support is set to 0. If no neighbour has data at that
               moment, the link cannot be checked and is left unchanged.

The +-15 min window means a warning can need up to 15 minutes to be confirmed.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.ndimage import maximum_filter1d

RADIUS_KM = 10.0
WINDOW_MIN = 15
WET_MM_H = 0.1


def _path_key(lat0, lon0, lat1, lon1) -> str:
    a, b = (round(lat0, 4), round(lon0, 4)), (round(lat1, 4), round(lon1, 4))
    return str(sorted([a, b]))


def neighbour_matrix(lat0, lon0, lat1, lon1, radius_km: float = RADIUS_KM) -> sparse.csr_matrix:
    """(n, n) 0/1 matrix: j is a neighbour of i (midpoints within radius, different path)."""
    lat0, lon0, lat1, lon1 = (np.asarray(v, float) for v in (lat0, lon0, lat1, lon1))
    mlat, mlon = (lat0 + lat1) / 2, (lon0 + lon1) / 2
    y = mlat * 111.32
    x = mlon * 111.32 * np.cos(np.radians(mlat.mean()))
    d = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    keys = np.array([_path_key(*v) for v in zip(lat0, lon0, lat1, lon1)])
    A = (d <= radius_km) & (keys[:, None] != keys[None])
    return sparse.csr_matrix(A.astype(np.float32))


def nearest_matrix(lat0, lon0, lat1, lon1, k: int, radius_km: float = RADIUS_KM) -> sparse.csr_matrix:
    """Like neighbour_matrix, but only each link's k nearest neighbours (within the radius)."""
    lat0, lon0, lat1, lon1 = (np.asarray(v, float) for v in (lat0, lon0, lat1, lon1))
    A = neighbour_matrix(lat0, lon0, lat1, lon1, radius_km).toarray()
    mlat, mlon = (lat0 + lat1) / 2, (lon0 + lon1) / 2
    d = np.hypot((mlat[:, None] - mlat[None]) * 111.32,
                 (mlon[:, None] - mlon[None]) * 111.32 * np.cos(np.radians(mlat.mean())))
    d[A == 0] = np.inf
    K = np.zeros_like(A)
    order = np.argsort(d, axis=1)[:, :k]
    for i in range(len(K)):
        j = order[i][np.isfinite(d[i, order[i]])]
        K[i, j] = 1
    return sparse.csr_matrix(K)


def neighbour_check(R: np.ndarray, A: sparse.csr_matrix, step_min: float,
                    window_min: float = WINDOW_MIN):
    """R: (n_links, T) rain rate [mm/h], NaN = missing. Returns (R_checked, info).

    info: share of the method's rainy steps removed, and how many rainy steps could
    not be checked because no neighbour had data.
    """
    R = np.asarray(R, float)
    present = np.isfinite(R)
    wet = present & (np.nan_to_num(R) >= WET_MM_H)
    w = max(1, int(round(window_min / step_min)))
    size = 2 * w + 1
    wet_win = maximum_filter1d(wet.astype(np.float32), size=size, axis=1, mode="constant")
    pres_win = maximum_filter1d(present.astype(np.float32), size=size, axis=1, mode="constant")
    support = (A @ wet_win) > 0
    checkable = (A @ pres_win) > 0
    remove = wet & checkable & ~support
    out = R.copy()
    out[remove] = 0.0
    info = {"rainy_steps": int(wet.sum()), "removed": int(remove.sum()),
            "removed_share": float(remove.sum() / max(1, wet.sum())),
            "unchecked_rainy_steps": int((wet & ~checkable).sum()),
            "radius_km": RADIUS_KM, "window_min": window_min,
            "median_neighbours": float(np.median(np.asarray(A.sum(axis=1)).ravel()))}
    return out, info
