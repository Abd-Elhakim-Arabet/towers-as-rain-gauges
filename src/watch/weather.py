"""Live rain from a weather service (Open-Meteo: free, no key, worldwide, 15-minute steps).

This is the "real weather" of the simulation mode, and the filler outside tower coverage in every mode. It is
weather-service rain (radar-fed where a country has radar, model-based elsewhere), not a measurement by us.

`Coarse` holds rain on a small grid of points over the box, in mm/h, one value per 15 minutes; a value at time T is
the mean over the 15 minutes that END at T (Open-Meteo's convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests

API = "https://api.open-meteo.com/v1/forecast"
STEP_S = 900


@dataclass
class Coarse:
    t: np.ndarray            # (T,) seconds since 1970 (UTC), END of each 15-minute interval
    lat: np.ndarray          # (ny,) south -> north
    lon: np.ndarray          # (nx,) west -> east
    mmh: np.ndarray          # (T, ny, nx)

    def at(self, t_s: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Rain (mm/h) at times t_s (seconds, any length T) on the regular grid lat x lon -> (T, len(lat), len(lon)).
        Bilinear in space, constant within each 15-minute interval; zero outside the time range."""
        k = np.searchsorted(self.t, np.asarray(t_s, float), side="left")
        ok = (k < self.t.size) & (np.asarray(t_s, float) > self.t[0] - STEP_S)
        fi = np.interp(lat, self.lat, np.arange(self.lat.size))
        fj = np.interp(lon, self.lon, np.arange(self.lon.size))
        i0, j0 = np.clip(np.floor(fi).astype(int), 0, max(self.lat.size - 2, 0)), np.clip(np.floor(fj).astype(int), 0, max(self.lon.size - 2, 0))
        i1, j1 = np.minimum(i0 + 1, self.lat.size - 1), np.minimum(j0 + 1, self.lon.size - 1)
        wi, wj = (fi - i0)[:, None], (fj - j0)[None, :]
        out = np.zeros((len(k), lat.size, lon.size), np.float32)
        for n in np.flatnonzero(ok):
            f = self.mmh[k[n]]
            if not f.any():
                continue
            out[n] = ((1 - wi) * (1 - wj) * f[np.ix_(i0, j0)] + (1 - wi) * wj * f[np.ix_(i0, j1)]
                      + wi * (1 - wj) * f[np.ix_(i1, j0)] + wi * wj * f[np.ix_(i1, j1)])
        return out

    def wet(self, t0: float, t1: float) -> bool:
        """Is there any rain on the grid between t0 and t1 (seconds)?"""
        m = (self.t > t0) & (self.t - STEP_S < t1)
        return bool(self.mmh[m].any())


def fetch(box, past_days: int = 4, n: int = 3, timeout: float = 30) -> Coarse:
    """The last `past_days` days and the next 24 h on an n x n grid of points over the box."""
    lat0, lon0, lat1, lon1 = box
    lat, lon = np.linspace(lat0, lat1, n), np.linspace(lon0, lon1, n)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    r = requests.get(API, params={"latitude": ",".join(f"{v:.4f}" for v in LA.ravel()),
                                  "longitude": ",".join(f"{v:.4f}" for v in LO.ravel()),
                                  "minutely_15": "precipitation", "past_days": past_days, "forecast_days": 2,
                                  "timezone": "UTC", "timeformat": "unixtime"}, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    d = d if isinstance(d, list) else [d]
    t = np.asarray(d[0]["minutely_15"]["time"], float)
    v = np.array([[np.nan if x is None else x for x in p["minutely_15"]["precipitation"]] for p in d], np.float32)
    mmh = np.nan_to_num(v).T.reshape(t.size, n, n) * 4.0          # mm per 15 min -> mm/h
    return Coarse(t, lat, lon, mmh)


def design_storm(box, start_s: float, total_mm: float = 90.0, hours: float = 3.0, n: int = 3) -> Coarse:
    """A made-up storm for drills: `total_mm` at the centre over `hours` (triangular in time), weaker at the edges."""
    lat0, lon0, lat1, lon1 = box
    lat, lon = np.linspace(lat0, lat1, n), np.linspace(lon0, lon1, n)
    steps = int(hours * 4)
    t = start_s + STEP_S * np.arange(1, steps + 9)                 # two dry hours after the rain
    shape = np.interp(np.arange(steps) + 0.5, [0, steps / 2, steps], [0, 1, 0])
    shape = shape / shape.sum() * total_mm * 4.0                   # mm/h per step at the centre
    yy, xx = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    space = 1.0 - 0.35 * (yy ** 2 + xx ** 2)
    mmh = np.zeros((t.size, n, n), np.float32)
    mmh[:steps] = shape[:, None, None] * space[None]
    return Coarse(t, lat, lon, mmh)
