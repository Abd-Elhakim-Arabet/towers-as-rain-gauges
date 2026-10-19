"""Rain maps from links: inverse-distance weighting (IDW) of path-average rain.

Each path is represented by its midpoint (the two directions of one physical link
are averaged first). For every target point (grid cell or gauge), its value is a
weighted average of the NNEAR nearest paths within MAX_KM, with weights 1/d^POWER:
closer links count more. Points with no path within MAX_KM get no value (NaN), so
the map never invents rain far from any link.

Weights are computed once for fixed link positions and reused for every time
step, so a whole season of maps is one sparse matrix product. Missing link values
at a time step are skipped and the remaining weights renormalised.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

NNEAR = 8
POWER = 2.0
MAX_KM = 10.0
MIN_KM = 0.5   # distances below this count as 0.5 km (avoid one link swamping a cell)


def merge_directions(R: np.ndarray, lat0, lon0, lat1, lon1):
    """Average the sub-links of each physical path. Returns (R_paths, mid_lat, mid_lon)."""
    keys = [str(sorted([(round(a, 4), round(b, 4)), (round(c, 4), round(d, 4))]))
            for a, b, c, d in zip(lat0, lon0, lat1, lon1)]
    uniq, inv = np.unique(keys, return_inverse=True)
    n_paths = len(uniq)
    M = sparse.csr_matrix((np.ones(len(keys)), (inv, np.arange(len(keys)))), shape=(n_paths, len(keys)))
    R = np.asarray(R, float)
    ok = np.isfinite(R)
    num = M @ np.where(ok, R, 0.0)
    den = M @ ok.astype(float)
    with np.errstate(invalid="ignore"):
        Rp = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    cnt = np.asarray(M.sum(axis=1)).ravel()
    mlat = (M @ ((np.asarray(lat0) + np.asarray(lat1)) / 2)) / cnt
    mlon = (M @ ((np.asarray(lon0) + np.asarray(lon1)) / 2)) / cnt
    return Rp, mlat, mlon


def to_km(lat, lon, lat0: float, lon0: float):
    """Local flat-earth km coordinates (fine at city-to-country scale)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return (lon - lon0) * 111.32 * np.cos(np.radians(lat0)), (lat - lat0) * 111.32


def idw_weights(src_xy: np.ndarray, dst_xy: np.ndarray, nnear: int = NNEAR, power: float = POWER,
                max_km: float = MAX_KM) -> sparse.csr_matrix:
    """(n_dst, n_src) sparse IDW weights (rows not yet normalised)."""
    tree = cKDTree(src_xy)
    k = min(nnear, len(src_xy))
    d, j = tree.query(dst_xy, k=k, distance_upper_bound=max_km)
    d, j = np.atleast_2d(d), np.atleast_2d(j)
    if k == 1:
        d, j = d.T, j.T
    ok = np.isfinite(d)
    w = np.where(ok, 1.0 / np.maximum(d, MIN_KM) ** power, 0.0)
    rows = np.repeat(np.arange(len(dst_xy)), k)
    return sparse.csr_matrix((w.ravel()[ok.ravel()], (rows[ok.ravel()], j.ravel()[ok.ravel()])),
                             shape=(len(dst_xy), len(src_xy)))


def apply_idw(W: sparse.csr_matrix, values: np.ndarray) -> np.ndarray:
    """values (n_src, T) -> (n_dst, T); NaN where no source with data is in range."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    num = W @ np.where(ok, v, 0.0)
    den = W @ ok.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)
