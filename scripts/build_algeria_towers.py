"""Merge OpenCelliD cells for Algeria and group them into estimated tower sites.

Sources:
  data/raw/opencellid/603.csv.gz            country file (a trimmed export: ~4% of cells in central Algiers)
  data/raw/opencellid/api/<area>/tile_*.json city sweeps from the API (scripts/fetch_opencellid_area.py)

Outputs:
  data/processed/algeria_cells.csv  one row per cell
  data/processed/algeria_sites.csv  one row per estimated site (cells of one operator within 150 m)
  site/algeria/data.json            compact data for the map page

    python scripts/build_algeria_towers.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import mapping, shape
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OPERATORS = {1: "Mobilis", 2: "Djezzy", 3: "Ooredoo"}
SITE_RADIUS_M = 150   # cells of one operator closer than this are treated as one site
EARTH_R = 6_371_000.0

FILE_COLS = ["radio", "mcc", "net", "area", "cell", "unit", "lon", "lat", "range", "samples",
             "changeable", "created", "updated", "averageSignal"]


def load_file_cells() -> pd.DataFrame:
    df = pd.read_csv(RAW / "opencellid" / "603.csv.gz", names=FILE_COLS)
    return pd.DataFrame({"radio": df.radio, "mnc": df.net, "area": df.area, "cell": df.cell,
                         "lon": df.lon, "lat": df.lat, "samples": df.samples, "source": "file"})


def load_api_cells() -> tuple[pd.DataFrame, dict]:
    rows, coverage = [], {}
    for area_dir in sorted((RAW / "opencellid" / "api").glob("*")):
        tiles = sorted(area_dir.glob("tile_*.json"))
        boxes = []
        for t in tiles:
            d = json.loads(t.read_text())
            boxes.append([float(v) for v in d["bbox"].split(",")])
            for c in d["cells"]:
                rows.append({"radio": c["radio"], "mnc": c["mnc"],
                             "area": c.get("lac") or c.get("tac"), "cell": c["cellid"],
                             "lon": c["lon"], "lat": c["lat"], "samples": c.get("samples", 0),
                             "source": f"api:{area_dir.name}"})
        coverage[area_dir.name] = boxes
    return pd.DataFrame(rows), coverage


def cluster_sites(cells: pd.DataFrame) -> pd.DataFrame:
    out = []
    for mnc, g in cells.groupby("mnc"):
        X = np.radians(g[["lat", "lon"]].to_numpy())
        lab = DBSCAN(eps=SITE_RADIUS_M / EARTH_R, min_samples=1, metric="haversine",
                     algorithm="ball_tree").fit_predict(X)
        g = g.assign(site=lab)
        s = g.groupby("site").agg(lon=("lon", "mean"), lat=("lat", "mean"), n_cells=("cell", "size"),
                                  radios=("radio", lambda r: "/".join(sorted(set(r)))))
        s["mnc"] = mnc
        out.append(s.reset_index(drop=True))
    sites = pd.concat(out, ignore_index=True)
    sites["operator"] = sites.mnc.map(OPERATORS)
    return sites


def borders() -> list:
    gj = json.loads((RAW / "boundaries" / "dza_adm1_simplified.geojson").read_text())
    feats = []
    for f in gj["features"]:
        geom = shape(f["geometry"]).simplify(0.01, preserve_topology=True)
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        rings = [[[round(x, 3), round(y, 3)] for x, y in p.exterior.coords] for p in polys]
        feats.append({"name": f["properties"]["shapeName"], "rings": rings})
    return feats


def main():
    fc = load_file_cells()
    ac, coverage = load_api_cells()
    cells = pd.concat([ac, fc], ignore_index=True)   # API first, so its copy wins on duplicates
    n_before = len(cells)
    cells = cells.drop_duplicates(subset=["mnc", "area", "cell", "radio"], keep="first")
    cells = cells[cells.mnc.isin(OPERATORS)].reset_index(drop=True)
    sites = cluster_sites(cells)

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "algeria_cells.csv", index=False)
    sites.to_csv(out / "algeria_sites.csv", index=False)

    radios = ["GSM", "UMTS", "LTE", "NR"]
    summary = {
        "n_cells": int(len(cells)), "n_sites": int(len(sites)),
        "n_file": int((cells.source == "file").sum()), "n_api": int((cells.source != "file").sum()),
        "duplicates_removed": int(n_before - len(cells)),
        "by_operator": {OPERATORS[k]: int(v) for k, v in cells.mnc.value_counts().items()},
        "sites_by_operator": {k: int(v) for k, v in sites.operator.value_counts().items()},
        "by_radio": {k: int(v) for k, v in cells.radio.value_counts().items()},
        "api_by_area": {k: int(v) for k, v in cells[cells.source != "file"].source.value_counts().items()},
        "site_radius_m": SITE_RADIUS_M,
    }
    data = {
        "summary": summary,
        "coverage": coverage,
        # sites: [lon, lat, operator 1-3, n_cells, radio bitmask GSM=1 UMTS=2 LTE=4 NR=8, from_api]
        "sites": [[round(r.lon, 5), round(r.lat, 5), int(r.mnc), int(r.n_cells),
                   sum(1 << i for i, rd in enumerate(radios) if rd in r.radios.split("/"))]
                  for r in sites.itertuples()],
        "borders": borders(),
    }
    page = ROOT / "site" / "algeria"
    page.mkdir(parents=True, exist_ok=True)
    (page / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    print(json.dumps(summary, indent=1))
    print(f"wrote {page / 'data.json'} ({(page / 'data.json').stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
