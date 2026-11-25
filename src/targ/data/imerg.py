"""NASA GPM IMERG V07 Final, half-hourly, 0.1 deg: small boxes via OPeNDAP.

Each global half-hourly file (~7.5 MB) is downloaded, the box is cut out, and only
the box is kept. (The OPeNDAP subset service would send only the box, but it
answered "503 busy" to most requests in October 2026; whole files download in ~5 s.)
Needs EARTHDATA_TOKEN in the environment or .env.
Each half-hour is cached as data/raw/imerg/<box>/<YYYYmmddTHHMM>.npy.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
BASE = "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07"
# IMERG grid: cell centres lon -179.95..179.95, lat -89.95..89.95, step 0.1
LON0, LAT0, STEP = -179.95, -89.95, 0.1


def token() -> str:
    t = os.environ.get("EARTHDATA_TOKEN")
    if not t:
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("EARTHDATA_TOKEN="):
                t = line.split("=", 1)[1].strip()
    if not t:
        raise RuntimeError("EARTHDATA_TOKEN missing (.env)")
    return t


def box_index(lat0, lon0, lat1, lon1):
    """Inclusive grid indices for a lat/lon box."""
    i0, i1 = int(round((lon0 - LON0) / STEP)), int(round((lon1 - LON0) / STEP))
    j0, j1 = int(round((lat0 - LAT0) / STEP)), int(round((lat1 - LAT0) / STEP))
    return i0, i1, j0, j1


def file_url(t: pd.Timestamp) -> str:
    s = t.strftime("%H%M%S")
    e = (t + pd.Timedelta("29min59s")).strftime("%H%M%S")
    minutes = t.hour * 60 + t.minute
    name = f"3B-HHR.MS.MRG.3IMERG.{t:%Y%m%d}-S{s}-E{e}.{minutes:04d}.V07B.HDF5"
    return f"{BASE}/{t:%Y}/{t.dayofyear:03d}/{name}"


def fetch(name: str, box, start: str, end: str) -> xr.DataArray:
    """Half-hourly rain rate [mm/h] in box=(lat0, lon0, lat1, lon1) for [start, end)."""
    i0, i1, j0, j1 = box_index(*box)
    out = ROOT / "data" / "raw" / "imerg" / name
    out.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(start, end, freq="30min", inclusive="left")
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token()}"
    frames = []
    for t in times:
        f = out / f"{t:%Y%m%dT%H%M}.npy"
        if not f.exists():
            for attempt in range(5):
                try:
                    r = s.get(file_url(t), timeout=300)
                    r.raise_for_status()
                    with h5py.File(io.BytesIO(r.content), "r") as h:
                        v = h["Grid/precipitation"][0, i0:i1 + 1, j0:j1 + 1]    # (lon, lat)
                    break
                except Exception:      # network hiccup or server busy: back off
                    if attempt == 4:
                        raise
                    time.sleep(10 * (attempt + 1))
            v = np.where(v < 0, np.nan, v)                       # fill value -9999.9
            np.save(f, v.T.astype(np.float32))                   # (lat, lon)
        frames.append(np.load(f))
    lat = LAT0 + STEP * np.arange(j0, j1 + 1)
    lon = LON0 + STEP * np.arange(i0, i1 + 1)
    # label each half-hour by its start time
    return xr.DataArray(np.stack(frames), dims=("time", "lat", "lon"),
                        coords={"time": times, "lat": np.round(lat, 2), "lon": np.round(lon, 2)},
                        name="rain", attrs={"units": "mm/h", "source": "GPM IMERG V07B Final half-hourly"})
