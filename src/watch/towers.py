"""Tower sites for any city, from three sources, best first.

  1. OpenCelliD   real sites, where a sweep of the area is already on disk (scripts/fetch_opencellid_area.py;
                  the API is quota-limited, so a new city cannot be swept on the spot)
  2. OpenStreetMap  real mast and antenna locations (Overpass API; few of them in most cities)
  3. estimated    sites placed on built-up land to reach a realistic density, so the simulated network has the
                  shape of a real one. These are NOT real towers and are marked `real = False`.

Density target: Algiers has 20 and Ghardaia 9 sites per built-up km2 (all operators, OpenCelliD + WorldCover).
"""

from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OPENCELLID = ROOT / "data" / "processed" / "algeria_sites.csv"
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter")
UA = "towers-as-rain-gauges/0.7 (flood watch research demo)"
SITES_PER_BUILT_KM2 = 12.0
MAX_SITES = 3000
MIN_GAP_M = 150.0            # two sites of one operator are never closer than this (same rule as the OpenCelliD build)
OPERATORS = ("Operator A", "Operator B", "Operator C")


def opencellid(box) -> pd.DataFrame:
    if not OPENCELLID.exists():
        return pd.DataFrame(columns=["lon", "lat", "operator"])
    s = pd.read_csv(OPENCELLID)
    lat0, lon0, lat1, lon1 = box
    s = s[s.lat.between(lat0, lat1) & s.lon.between(lon0, lon1)]
    return s[["lon", "lat", "operator"]].reset_index(drop=True)


def osm(box, cache: Path | None = None) -> pd.DataFrame:
    """Communication masts and mobile-phone antennas mapped in OpenStreetMap. Empty if every server is busy."""
    if cache and cache.exists():
        els = json.loads(cache.read_text(encoding="utf-8"))
    else:
        b = ",".join(f"{v:.5f}" for v in box)
        q = (f'[out:json][timeout:60];(node["man_made"~"^(mast|tower)$"]["tower:type"="communication"]({b});'
             f'node["communication:mobile_phone"="yes"]({b});node["man_made"="mast"]["operator"]({b}););out body;')
        els = None
        for attempt in range(2):
            for url in OVERPASS:
                try:
                    r = requests.post(url, data={"data": q}, headers={"User-Agent": UA}, timeout=90)
                    if r.ok and r.headers.get("content-type", "").startswith("application/json"):
                        els = r.json()["elements"]
                        break
                except requests.RequestException:
                    pass
            if els is not None:
                break
            time.sleep(8)
        if els is None:
            return pd.DataFrame(columns=["lon", "lat", "operator"])
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(els), encoding="utf-8")
    rows = [{"lon": e["lon"], "lat": e["lat"], "operator": (e.get("tags", {}).get("operator") or "").split(";")[0].strip()}
            for e in els if "lat" in e]
    return pd.DataFrame(rows, columns=["lon", "lat", "operator"])


def estimated(g, town: np.ndarray, n: int, seed: int, avoid: pd.DataFrame | None = None) -> pd.DataFrame:
    """n made-up sites: more where more of the land is built, never two of one operator within MIN_GAP_M."""
    if n <= 0:
        return pd.DataFrame(columns=["lon", "lat", "operator"])
    rng = np.random.default_rng(seed)
    w = np.where(town, g.built, 0).astype(float).ravel()
    if w.sum() <= 0:
        return pd.DataFrame(columns=["lon", "lat", "operator"])
    dlat, dlon = abs(g.lat[1] - g.lat[0]), abs(g.lon[1] - g.lon[0])
    kx, ky = 111_320 * np.cos(np.radians(g.lat.mean())), 110_540
    placed = {op: [] for op in OPERATORS}
    for _, r in (avoid if avoid is not None else pd.DataFrame()).iterrows():
        placed.setdefault(r.operator, []).append((r.lon * kx, r.lat * ky))
    rows = []
    for idx in rng.choice(w.size, size=min(w.size, 6 * n), p=w / w.sum(), replace=True):
        if len(rows) >= n:
            break
        r, c = divmod(int(idx), g.lon.size)
        lon = g.lon[c] + rng.uniform(-0.5, 0.5) * dlon
        lat = g.lat[r] + rng.uniform(-0.5, 0.5) * dlat
        op = OPERATORS[len(rows) % len(OPERATORS)]
        x, y = lon * kx, lat * ky
        if placed[op] and min((x - a) ** 2 + (y - b) ** 2 for a, b in placed[op][-400:]) < MIN_GAP_M ** 2:
            continue
        placed[op].append((x, y))
        rows.append({"lon": round(float(lon), 5), "lat": round(float(lat), 5), "operator": op})
    return pd.DataFrame(rows, columns=["lon", "lat", "operator"])


def build(area, S, cache_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """All sites for the area (lon, lat, operator, real, source) and a count per source."""
    cache_dir = cache_dir or area.dir
    f = cache_dir / "sites.csv"
    if f.exists():
        sites = pd.read_csv(f)
    else:
        g = S["g"]
        real = opencellid(area.box).assign(source="OpenCelliD")
        if len(real) < 20:
            o = osm(area.box, cache_dir / "osm_masts.json").assign(source="OpenStreetMap")
            real = pd.concat([real, o], ignore_index=True)
        # a real site with no operator tag is shared out, so every operator's network gets some real anchors
        blank = real.operator.fillna("").eq("")
        real.loc[blank, "operator"] = [OPERATORS[i % len(OPERATORS)] for i in range(int(blank.sum()))]
        target = int(min(MAX_SITES, SITES_PER_BUILT_KM2 * float(S["built_ha"][S["town"]].sum()) / 100))
        est = estimated(g, S["town"], target - len(real), zlib.crc32(area.key.encode()), real).assign(source="estimated")
        sites = pd.concat([real, est], ignore_index=True)
        sites["real"] = sites.source != "estimated"
        cache_dir.mkdir(parents=True, exist_ok=True)
        sites.to_csv(f, index=False)
    return sites, {k: int(v) for k, v in sites.source.value_counts().items()}
