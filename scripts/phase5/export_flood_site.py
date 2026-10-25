"""Build the Phase 5 flood page (site/flood/index.html) from the flood runs.

    python scripts/phase5/export_flood_site.py
Embeds, per city, storm version and rain source: hourly water-depth frames (PNG), the worst-danger map, the flow out
of the checked basin (10-minute means), the alarm times and the communes with most built-up land in danger.
The page streams Esri imagery (needs internet); everything else is inside the page.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from floodrisk import config, hazard, rain  # noqa: E402
import run_floods as RF  # noqa: E402

OUT = ROOT / "site" / "flood"
MAP_SOURCES = ("truth", "ai", "classical", "imerg")
EVENTS = {"algiers": "Bab El Oued flood, 9–10 Nov 2001", "ghardaia": "M'zab valley flood, 29 Sep – 1 Oct 2008"}
# depth bins (m) -> sequential blue ramp (dataviz reference, steps 100..700)
DEPTH_EDGES = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]          # thinner sheets of water are not drawn
DEPTH_RGB = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
HAZ_RGB = ["#fdd0a2", "#fd8d3c", "#e6550d", "#a63603"]


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def png(idx: np.ndarray, colours) -> str:
    """Palette PNG; index 0 transparent, 1..n the colours."""
    im = Image.fromarray(idx.astype(np.uint8), "P")
    pal = [0, 0, 0] + [c for h in colours for c in rgb(h)]
    im.putpalette(pal + [0] * (768 - len(pal)))
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True, transparency=0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def depth_png(h):
    h = np.asarray(h, np.float32)
    idx = np.digitize(h, DEPTH_EDGES)          # 0 below 10 cm
    return png(idx, DEPTH_RGB)


def outline(mask, lat, lon, simplify=None):
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    res = lon[1] - lon[0]
    tr = from_origin(lon[0] - res / 2, lat[0] + res / 2, res, res)
    polys = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask, transform=tr) if v == 1]
    if not polys:
        return {"type": "FeatureCollection", "features": []}
    u = unary_union(polys).simplify(simplify or res)
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": mapping(u)}]}


def city_data(key):
    city = config.CITIES[key]
    summ = json.loads((config.RESULTS / f"{key}.json").read_text())
    S = RF.setup(city)
    g = S["g"]
    d = config.PROC / key
    bkey = city.outlets[0].key
    c = 3                                         # frames were saved 3x coarser
    ny, nx = (g.z.shape[0] // c) * c, (g.z.shape[1] // c) * c
    res = g.lon[1] - g.lon[0]
    west, north = g.lon[0] - res / 2, g.lat[0] + res / 2
    bounds = [west, north - ny * res, west + nx * res, north]
    full_bounds = [west, g.lat[-1] - res / 2, g.lon[-1] + res / 2, north]
    hours0 = pd.Timestamp(summ["hours0"])
    runs = {}
    n_hours = None
    for scen in ("gauge", "satellite"):
        runs[scen] = {}
        for s in RF.SOURCES:
            R = np.load(d / f"{scen}_{s}.npz", allow_pickle=True)
            q = R[f"q_{bkey}"]
            q10 = q[: q.size // 10 * 10].reshape(-1, 10).mean(1)
            sm = summ["scenarios"][scen][s]
            entry = {"q": [round(float(v), 2) for v in q10],
                     "alarm": {"time": sm["alarm_time"], "lead": sm["alarm_lead_min"],
                               "flood": summ["scenarios"][scen]["truth"]["zone_flood_time"]},
                     "risk": [{"commune": r["commune"], "ha": r["built_ha_at_risk"], "share": r["share"]}
                              for r in sm["communes_top"][:8] if r["built_ha_at_risk"] > 0.05]}
            if s in MAP_SOURCES:
                snaps = R["snaps"]                 # every 30 min
                frames = [depth_png(snaps[i]) for i in range(0, snaps.shape[0], 2)]
                n_hours = len(frames)
                hcls = hazard.hazard_class(R["hrmax"].astype(np.float32), R["hmax"].astype(np.float32))
                entry["frames"] = frames
                entry["hazard"] = png(np.where(hcls >= 1, hcls, 0), HAZ_RGB)
            runs[scen][s] = entry
    labels = [str(hours0 + pd.Timedelta(hours=h))[5:16] for h in range(n_hours + 1)]
    obs = city.observed.get(bkey, {})
    peak = obs.get("peak_m3s")
    tbox = RF._tower_box(rain.Phase4Rain(key), g)
    truth_g = runs["gauge"]["truth"]["q"]
    start = max(0, int(np.argmax(truth_g) * 10 / 60) - 8)
    b = summ["basins"][bkey]
    return {
        "key": key, "name": city.name, "event": EVENTS[key], "bounds": bounds, "full_bounds": full_bounds,
        "view": full_bounds, "grid": summ["grid"], "cell": [round(x) for x in summ["cell_m"]],
        "basinName": city.outlets[0].name, "basinArea": round(b["area_km2"], 1),
        "towerShare": f"{100 * b['tower_box_share']:.0f}%",
        "sources": list(RF.SOURCES), "mapSources": list(MAP_SOURCES), "startHour": start,
        "hourLabels": labels, "q_step_h": 10 / 60, "alarmHa": config.ALARM_HA,
        "observed": {"peak": peak, "label": f"flood marks {peak:.0f} m³/s" if key == "algiers" else f"reported ~{peak:.0f} m³/s"}
        if peak else None,
        "obsNote": obs.get("src", ""),
        "gaugeNote": f"Every rain map × {summ['gauge_factor']:.2f}, so the basin storm total matches the gauge "
                     f"({city.gauge_total_mm:.0f} mm: {city.gauge_note}).",
        "basin": outline(S["basins"][bkey]["mask"], g.lat, g.lon),
        "towerbox": outline(tbox, g.lat, g.lon),
        "zone": outline(S["zone"], g.lat, g.lon, simplify=3 * res),
        "runs": runs,
    }


def main():
    cities = [city_data(k) for k in config.CITIES if (config.RESULTS / f"{k}.json").exists()]
    html = (OUT / "template.html").read_text().replace("/*DATA*/null", json.dumps(cities, separators=(",", ":")))
    (OUT / "index.html").write_text(html)
    print("wrote", OUT / "index.html", f"{len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
