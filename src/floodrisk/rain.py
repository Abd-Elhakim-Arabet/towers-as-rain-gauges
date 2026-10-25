"""Rain sources for the flood model, all put on the flood grid as hourly rain-rate fields (mm/h).

  GriddedRain   any hourly lat/lon rain field (the general case)
  Phase4Rain    THE LINK TO PHASE 4: reads data/processed/phase4/sim_<city>.npz (read-only) and returns one of its
                hourly 500 m maps (truth / ai / classical / ai_sparse / classical_sparse), or the satellite (imerg).
                Outside the 500 m tower box, and where a tower map has no value, the satellite fills in, because
                that is what a real system would have there.
  DesignStorm   a made-up storm (uniform in space), for tests and for rain thresholds
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PHASE4

SOURCES = ("truth", "ai", "classical", "ai_sparse", "classical_sparse", "imerg")


def _nearest_index(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest cell centre on a regular axis, and whether the value falls inside the axis cells."""
    step = axis[1] - axis[0]
    i = np.round((values - axis[0]) / step).astype(int)
    inside = (i >= 0) & (i < axis.size)
    return np.clip(i, 0, axis.size - 1), inside


@dataclass
class GriddedRain:
    lat: np.ndarray            # cell centres (any order, regular)
    lon: np.ndarray
    hours: pd.DatetimeIndex
    rate: np.ndarray           # (T, lat, lon) mm/h, hour-mean

    def on(self, lat: np.ndarray, lon: np.ndarray):
        """Nearest-cell values on the target grid: ((T, ny, nx) rates, inside mask (ny, nx))."""
        i, ii = _nearest_index(self.lat, lat)
        j, jj = _nearest_index(self.lon, lon)
        inside = ii[:, None] & jj[None, :]
        return self.rate[:, i][:, :, j], inside


def _imerg_hourly(z) -> GriddedRain:
    t = pd.DatetimeIndex(z["imerg_time"])
    r = np.asarray(z["imerg"], np.float32)
    hrs = t.floor("h")
    uh = pd.DatetimeIndex(sorted(set(hrs)))
    rate = np.stack([np.nanmean(r[hrs == h], 0) for h in uh])
    return GriddedRain(np.asarray(z["imerg_lat"], float), np.asarray(z["imerg_lon"], float), uh, np.nan_to_num(rate))


class Phase4Rain:
    """Hourly rain on the flood grid from the Phase 4 simulation of one city."""

    def __init__(self, city: str, path=None):
        self.path = path or PHASE4 / f"sim_{city}.npz"
        z = np.load(self.path, allow_pickle=True)
        self.hours = pd.DatetimeIndex(z["hours"])
        self.imerg = _imerg_hourly(z)
        self.maps = {k: GriddedRain(np.asarray(z["lat"], float), np.asarray(z["lon"], float), self.hours,
                                    np.asarray(z[k], np.float32)) for k in SOURCES[:-1] if k in z.files}

    def fields(self, source: str, lat: np.ndarray, lon: np.ndarray):
        """(T, ny, nx) mm/h on the flood grid for the Phase 4 hours, and the share of cells the source itself covers."""
        sat, _ = self.imerg.on(lat, lon)
        sat = sat[[self.imerg.hours.get_loc(h) for h in self.hours]]
        if source == "imerg":
            return sat, 0.0
        m, inside = self.maps[source].on(lat, lon)
        own = inside[None] & np.isfinite(m)
        out = np.where(own, m, sat)
        return np.clip(out, 0, None).astype(np.float32), float(own.mean())


def design_storm(total_mm: float, duration_h: int, shape: tuple[int, int], peak_at: float = 0.5,
                 hours_after: int = 6) -> np.ndarray:
    """Uniform storm with a triangular hyetograph (hourly), then dry hours. (T, ny, nx) mm/h."""
    t = np.arange(duration_h) + 0.5
    tp = peak_at * duration_h
    w = np.where(t <= tp, t / max(tp, 1e-9), (duration_h - t) / max(duration_h - tp, 1e-9)).clip(min=0.05)
    hy = total_mm * w / w.sum()
    hy = np.concatenate([hy, np.zeros(hours_after)])
    return np.broadcast_to(hy[:, None, None], (hy.size, *shape)).astype(np.float32).copy()
