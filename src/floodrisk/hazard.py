"""From water depth and speed to danger, exposure, alarms and scores.

Hazard to people: UK DEFRA/Environment Agency FD2320 rating  HR = d (v + 0.5) + DF
  d depth (m), v speed (m/s), DF debris factor (0 below 0.25 m, 0.5 to 0.75 m, 1 above).
  Classes: low < 0.75 <= moderate ("danger for some") < 1.25 <= significant ("danger for most") < 2.0 <= extreme
  ("danger for all"). Cells that never get 10 cm of water are "dry".
Exposure: built-up land (WorldCover class 50 share of each cell).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLASSES = ("dry", "low", "moderate", "significant", "extreme")
HR_EDGES = (0.75, 1.25, 2.0)
WET_M = 0.3                     # "knee-deep": the depth used for flooded area and for alarms


def hazard_class(hrmax: np.ndarray, hmax: np.ndarray) -> np.ndarray:
    c = 1 + np.digitize(hrmax, HR_EDGES)
    return np.where(hmax < 0.1, 0, c).astype(np.uint8)


def built_area_ha(built: np.ndarray, cell_m2: float) -> np.ndarray:
    return built * cell_m2 / 1e4


def wet_area_curve(t_wet: np.ndarray, weight_ha: np.ndarray, times_s: np.ndarray, mask=None) -> np.ndarray:
    """Built-up hectares that have been under >= WET_M of water by each time."""
    tw = t_wet if mask is None else np.where(mask, t_wet, np.inf)
    w = weight_ha if mask is None else np.where(mask, weight_ha, 0.0)
    flat_t, flat_w = tw.ravel(), w.ravel()
    o = np.argsort(flat_t)
    cum = np.cumsum(flat_w[o])
    k = np.searchsorted(flat_t[o], times_s, side="right")
    return np.where(k > 0, cum[np.maximum(k - 1, 0)], 0.0)


def first_crossing(times_s: np.ndarray, values: np.ndarray, level: float) -> float:
    """First time the series reaches `level` (linear between samples); nan if never."""
    v = np.asarray(values)
    i = np.flatnonzero(v >= level)
    if i.size == 0:
        return float("nan")
    i = i[0]
    if i == 0:
        return float(times_s[0])
    f = (level - v[i - 1]) / max(v[i] - v[i - 1], 1e-12)
    return float(times_s[i - 1] + f * (times_s[i] - times_s[i - 1]))


def csi(a: np.ndarray, b: np.ndarray, mask=None) -> float:
    """Critical success index of two flooded masks: overlap / union."""
    if mask is not None:
        a, b = a & mask, b & mask
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def commune_raster(path: Path, lat: np.ndarray, lon: np.ndarray):
    """Label every grid cell with its commune (geoBoundaries ADM3). Returns (labels int32, names list); 0 = none."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from shapely.geometry import box, shape
    feats = json.loads(Path(path).read_text(encoding="utf-8"))["features"]
    res = lon[1] - lon[0]
    tr = from_origin(lon[0] - res / 2, lat[0] + res / 2, res, res)
    B = box(lon[0], lat[-1], lon[-1], lat[0])
    shapes, names = [], []
    for f in feats:
        if not f["geometry"]:
            continue
        g = shape(f["geometry"])
        if g.intersects(B):
            names.append(f["properties"].get("shapeName", "?"))
            shapes.append((g, len(names)))
    lab = rasterize(shapes, out_shape=(lat.size, lon.size), transform=tr, fill=0, dtype="int32")
    return lab, names


def commune_risk(labels, names, hclass, built_ha, min_class: int = 3):
    """Built-up hectares at hazard class >= min_class (default 'significant'), per commune, sorted."""
    at = np.where(hclass >= min_class, built_ha, 0.0)
    ha = np.bincount(labels.ravel(), weights=at.ravel(), minlength=len(names) + 1)
    tot = np.bincount(labels.ravel(), weights=built_ha.ravel(), minlength=len(names) + 1)
    rows = [{"commune": names[i - 1], "built_ha_at_risk": float(ha[i]), "built_ha": float(tot[i]),
             "share": float(ha[i] / tot[i]) if tot[i] > 0 else 0.0}
            for i in range(1, len(names) + 1) if tot[i] > 0]
    return sorted(rows, key=lambda r: -r["built_ha_at_risk"])
