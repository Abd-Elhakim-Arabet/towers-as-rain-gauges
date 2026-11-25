"""Terrain, land cover and soil on one geographic grid per city, downloaded once and cached.

Grid: regular lat/lon, rows north->south, cell = agg arc-seconds (agg=1 ~ 30 m). Cell sizes in metres come with it.

Sources (all public; see docs/phase5/research.md):
  terrain    Copernicus GLO-30 DEM (AWS open data, no key)  -- or FABDEM if a local copy exists
  land cover ESA WorldCover 2021 v200, 10 m (AWS open data, no key)
  soil       HYSOGs250m hydrologic soil groups (ORNL DAAC; NASA Earthdata token, the one IMERG uses)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import RAW, ROOT, City

DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_DEM/"
           "Copernicus_DSM_COG_10_{t}_DEM.tif")
WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_{t}_Map.tif")
HYSOGS_URL = "https://daac.ornl.gov/daacdata/global_soil/Global_Hydrologic_Soil_Group/data/HYSOGs250m.tif"
FABDEM_DIR = RAW / "fabdem"          # optional: unzipped FABDEM tiles (*.tif) put here by hand or by a script
ARCSEC = 1 / 3600
R_EARTH = 6371008.8


@dataclass
class Ground:
    city: str
    lat: np.ndarray            # cell-centre latitudes (north -> south)
    lon: np.ndarray
    z: np.ndarray              # terrain height, m
    landcover: np.ndarray      # WorldCover majority class per cell (10..100)
    built: np.ndarray          # fraction of the cell that is built-up (WorldCover class 50)
    hsg: np.ndarray            # HYSOGs code (1-4 = A-D, 11-14 = A/D..D/D, 255 none)
    dx: float                  # east-west cell size, m (at the box centre)
    dy: float                  # north-south cell size, m
    dem_source: str

    @property
    def sea(self) -> np.ndarray:
        """Open sea: WorldCover water at or below 1 m. (Inland water keeps its water.)"""
        return (self.landcover == 80) & (self.z <= 1.0)

    def index(self, lat: float, lon: float) -> tuple[int, int]:
        r = int(np.clip(np.round((self.lat[0] - lat) / (self.lat[0] - self.lat[1])), 0, self.lat.size - 1))
        c = int(np.clip(np.round((lon - self.lon[0]) / (self.lon[1] - self.lon[0])), 0, self.lon.size - 1))
        return r, c


def earthdata_token() -> str | None:
    """NASA Earthdata token from the environment or .env (the same one the IMERG download uses)."""
    t = os.environ.get("EARTHDATA_TOKEN")
    env = ROOT / ".env"
    if not t and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("EARTHDATA_TOKEN="):
                t = line.split("=", 1)[1].strip()
    return t or None


def dem_tiles(box) -> tuple[str, ...]:
    """Names of the 1-degree Copernicus tiles that cover a (lat0, lon0, lat1, lon1) box."""
    lat0, lon0, lat1, lon1 = box
    return tuple(f"{'N' if la >= 0 else 'S'}{abs(la):02d}_00_{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00"
                 for la in range(int(np.floor(lat0)), int(np.floor(lat1 - 1e-9)) + 1)
                 for lo in range(int(np.floor(lon0)), int(np.floor(lon1 - 1e-9)) + 1))


def _grid(box, agg: int):
    lat0, lon0, lat1, lon1 = box
    res = agg * ARCSEC
    ny, nx = int(round((lat1 - lat0) / res)), int(round((lon1 - lon0) / res))
    from rasterio.transform import from_origin
    tr = from_origin(lon0, lat1, res, res)
    lat = lat1 - res * (np.arange(ny) + 0.5)
    lon = lon0 + res * (np.arange(nx) + 0.5)
    phi = np.deg2rad(0.5 * (lat0 + lat1))
    dy = np.deg2rad(res) * R_EARTH
    dx = dy * np.cos(phi)
    return tr, (ny, nx), lat, lon, dx, dy


def _warp(paths, tr, shape, resampling, dtype, nodata, skip_missing: bool = False):
    """skip_missing: a tile that does not exist is left as nodata (open sea has no terrain or land-cover tile)."""
    import rasterio
    from rasterio.warp import reproject
    out = np.full(shape, nodata, dtype)
    for p in paths:
        try:
            src = rasterio.open(p)
        except rasterio.errors.RasterioIOError:
            if skip_missing:
                continue
            raise
        with src:
            tmp = np.full(shape, nodata, dtype)
            reproject(rasterio.band(src, 1), tmp, dst_transform=tr, dst_crs="EPSG:4326",
                      dst_nodata=nodata, resampling=resampling)
        ok = tmp != nodata if not np.isnan(nodata) else np.isfinite(tmp)
        out[ok] = tmp[ok]
    return out


def _wc_tiles(box):
    lat0, lon0, lat1, lon1 = box
    tiles = []
    for la in range(int(np.floor(lat0 / 3) * 3), int(np.floor(lat1 / 3) * 3) + 1, 3):
        for lo in range(int(np.floor(lon0 / 3) * 3), int(np.floor(lon1 / 3) * 3) + 1, 3):
            tiles.append(f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}")
    return tiles


def build(city: City, force: bool = False) -> Ground:
    """Download/crop the three layers onto the city grid; cached at data/raw/flood/<city>/ground.npz."""
    return build_box(city.key, city.box, city.agg, city.dem_tiles, RAW / city.key / "ground.npz", force)


def build_box(key: str, box, agg: int = 1, tiles=None, cache: Path | None = None, force: bool = False,
              soil_default: int = 3) -> Ground:
    """The same three layers for any (lat0, lon0, lat1, lon1) box in the world.

    If the soil server cannot be reached, every cell gets `soil_default` (3 = group C, slow to soak: the cautious
    choice for a flood watch) and `dem_source` says so.
    """
    from rasterio.enums import Resampling
    cache = cache or RAW / key / "ground.npz"
    tiles = tiles or dem_tiles(box)
    tr, shape, lat, lon, dx, dy = _grid(box, agg)
    if cache.exists() and not force:
        g = np.load(cache)
        return Ground(key, lat, lon, g["z"], g["landcover"], g["built"], g["hsg"], dx, dy, str(g["dem_source"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")

    fab = sorted(FABDEM_DIR.glob("*.tif")) if FABDEM_DIR.exists() else []
    if fab:
        dem_paths, dem_source = [str(p) for p in fab], "FABDEM (bare earth)"
    else:
        dem_paths, dem_source = [DEM_URL.format(t=t) for t in tiles], "Copernicus GLO-30 (surface)"
    z = _warp(dem_paths, tr, shape, Resampling.average if agg > 1 else Resampling.bilinear, np.float32, np.nan,
              skip_missing=True)
    z = np.where(np.isfinite(z) & (z > -100), z, 0.0).astype(np.float32)

    # land cover: read at ~10 m (cell/3/agg), then majority and built fraction per model cell
    f = 3 * agg
    from rasterio.transform import from_origin
    tr_f = from_origin(tr.c, tr.f, tr.a / f, tr.a / f)
    wc = _warp([WC_URL.format(t=t) for t in _wc_tiles(box)], tr_f, (shape[0] * f, shape[1] * f),
               Resampling.nearest, np.uint8, 0, skip_missing=True)
    wc[wc == 0] = 80                           # no land-cover tile = open water
    blocks = wc.reshape(shape[0], f, shape[1], f).transpose(0, 2, 1, 3).reshape(shape[0], shape[1], f * f)
    built = (blocks == 50).mean(-1).astype(np.float32)
    classes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], np.uint8)
    counts = np.stack([(blocks == k).sum(-1) for k in classes], -1)
    landcover = classes[np.argmax(counts, -1)]

    # soil groups: one window read through the Earthdata token
    tok = earthdata_token()
    if tok:
        os.environ["GDAL_HTTP_HEADERS"] = f"Authorization: Bearer {tok}"
    try:
        hsg = _warp(["/vsicurl/" + HYSOGS_URL], tr, shape, Resampling.nearest, np.uint8, 255)
    except Exception:                          # server down or no token: fall back, and say so
        hsg = np.full(shape, 255, np.uint8)
    os.environ.pop("GDAL_HTTP_HEADERS", None)
    if (hsg == 255).all():
        hsg[:] = soil_default
        dem_source += "; soil: default group (soil map unavailable)"

    np.savez_compressed(cache, z=z, landcover=landcover, built=built, hsg=hsg, dem_source=dem_source)
    return Ground(key, lat, lon, z, landcover, built, hsg, dx, dy, dem_source)
