"""Sweep the OpenCelliD API over a city, tile by tile.

The downloadable country file (603.csv.gz) is a trimmed export: in a test box in
central Algiers it had 32 cells while the API had 745, and it had none at all in
Ghardaïa. The API only accepts boxes under 4 km² and returns 50 cells per
request, so we tile the area and page through each tile. Every tile is cached as
JSON, so the sweep resumes where it stopped (e.g. after hitting the daily quota).

    python scripts/fetch_opencellid_area.py ghardaia
    python scripts/fetch_opencellid_area.py algiers
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://opencellid.org/cell"

# lat_min, lon_min, lat_max, lon_max
AREAS = {
    "ghardaia": (32.40, 3.60, 32.55, 3.78),   # M'zab valley: Ghardaïa, Bounoura, El Atteuf, Beni Isguen
    "algiers": (36.65, 2.90, 36.82, 3.25),    # central Algiers incl. Bab El Oued
}
TILE_KM = 1.8        # 1.8 × 1.8 = 3.24 km², under the API's 4 km² limit
PAGE = 50
PAUSE_S = 0.5


def load_token() -> str:
    tok = os.environ.get("OPENCELLID_TOKEN")
    if not tok:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("OPENCELLID_TOKEN="):
                tok = line.split("=", 1)[1].strip()
    if not tok:
        sys.exit("OPENCELLID_TOKEN missing (.env)")
    return tok


def get(session, endpoint, params):
    for attempt in range(5):
        try:
            r = session.get(f"{API}/{endpoint}", params=params, timeout=60)
            d = r.json()
        except (requests.RequestException, ValueError):
            time.sleep(5 * (attempt + 1))
            continue
        if "error" in d:
            raise RuntimeError(f"API error: {d}")
        return d
    raise RuntimeError("API unreachable")


def main(area: str) -> None:
    lat0, lon0, lat1, lon1 = AREAS[area]
    out = ROOT / "data" / "raw" / "opencellid" / "api" / area
    out.mkdir(parents=True, exist_ok=True)
    dlat = TILE_KM / 111.0
    dlon = TILE_KM / (111.0 * math.cos(math.radians((lat0 + lat1) / 2)))
    ny, nx = math.ceil((lat1 - lat0) / dlat), math.ceil((lon1 - lon0) / dlon)
    token, s = load_token(), requests.Session()
    total = requests_made = 0
    print(f"{area}: {ny}×{nx} = {ny * nx} tiles")
    for i in range(ny):
        for j in range(nx):
            f = out / f"tile_{i:02d}_{j:02d}.json"
            if f.exists():
                total += len(json.loads(f.read_text())["cells"])
                continue
            bbox = (f"{lat0 + i * dlat:.5f},{lon0 + j * dlon:.5f},"
                    f"{min(lat1, lat0 + (i + 1) * dlat):.5f},{min(lon1, lon0 + (j + 1) * dlon):.5f}")
            try:
                n = get(s, "getInAreaSize", {"key": token, "BBOX": bbox, "format": "json"})["count"]
                requests_made += 1
                cells = []
                for off in range(0, n, PAGE):
                    d = get(s, "getInArea", {"key": token, "BBOX": bbox, "format": "json",
                                             "limit": PAGE, "offset": off})
                    requests_made += 1
                    cells += d.get("cells", [])
                    time.sleep(PAUSE_S)
            except RuntimeError as e:
                print(f"stopped at tile {i},{j} after {requests_made} requests: {e}")
                print("re-run later to resume")
                return
            f.write_text(json.dumps({"bbox": bbox, "count": n, "cells": cells}))
            total += len(cells)
            print(f"tile {i:02d},{j:02d}: {len(cells):4d} cells (total {total}, requests {requests_made})",
                  flush=True)
    print(f"{area}: done, {total} cells, {requests_made} new requests")


if __name__ == "__main__":
    main(sys.argv[1])
