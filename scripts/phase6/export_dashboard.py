"""Phase 6: build the operations dashboard (site/dashboard/) from the Phase 4-5 simulation.

    python scripts/phase6/export_dashboard.py [algiers|ghardaia ...]
Needs scripts/phase6/run_dashboard_floods.py first. Writes (all gitignored, rebuilt from the runs):
  site/dashboard/index.html                 the page (template + small JSON summary per city)
  site/dashboard/data/terrain/{z}/{x}/{y}.png  terrain tiles (Terrarium) made from the flood model's DEM, so the
                                               water sits exactly on the ground
  site/dashboard/data/<city>/water.bin      the water surface mesh (corners of every cell that is ever wet)
  site/dashboard/data/<city>/<src>/NNN.png  one frame per hour: R depth, G/B velocity east/north
  site/dashboard/data/<city>/rain.bin       hourly rain per source on a ~500 m grid (0.5 mm/h steps)
  site/dashboard/data/<city>/net.json       towers, links and hourly rain per link
"""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.merge import merge
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from floodrisk import config, hazard, rain  # noqa: E402
import run_floods as RF  # noqa: E402
from run_dashboard_floods import PROC6, SOURCES  # noqa: E402

OUT = ROOT / "site" / "dashboard"
DATA = OUT / "data"
DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_DEM/"
           "Copernicus_DSM_COG_10_{t}_DEM.tif")
PAD = {"algiers": 0.35, "ghardaia": 0.45}      # terrain beyond the flood grid, degrees (context to the horizon)
TILE_Z = range(8, 13)                          # z12 ~ 31 m pixels here, the DEM's own detail
H_MAX = 8.0                                    # depth encoding: d = H_MAX * (R/255)^2
V_SCALE = 25.0                                 # velocity encoding: G = 128 + u * V_SCALE (+-5 m/s)
EVENTS = {"algiers": "Bab El Oued flood · 9–10 Nov 2001", "ghardaia": "M'zab valley flood · 29 Sep – 1 Oct 2008"}
LABEL = {"truth": "True rain", "ai": "Towers + AI", "classical": "Towers, classical", "imerg": "Satellite only"}


# ---------------------------------------------------------------- terrain
def dem_region(key, bounds):
    f = PROC6 / key / "dem_region.npz"
    if f.exists():
        z = np.load(f)
        return z["z"], tuple(z["b"]), float(z["res"])
    lon0, lat0, lon1, lat1 = bounds
    tiles = [f"N{la:02d}_00_E{lo:03d}_00" for la in range(int(math.floor(lat0)), int(math.floor(lat1)) + 1)
             for lo in range(int(math.floor(lon0)), int(math.floor(lon1)) + 1)]
    srcs = []
    for t in tiles:
        try:
            srcs.append(rasterio.open(DEM_URL.format(t=t)))
        except Exception:                      # all-sea tiles do not exist
            print("  no DEM tile", t, flush=True)
    res = 1 / 3600
    arr, _ = merge(srcs, bounds=bounds, res=(res, res), resampling=Resampling.bilinear, nodata=0)
    z = np.nan_to_num(arr[0].astype(np.float32))
    z[z < -100] = 0
    np.savez_compressed(f, z=z, b=np.array(bounds), res=res)
    return z, bounds, res


def tile_bounds(z, x, y):
    n = 2 ** z
    def lon(xx): return xx / n * 360 - 180
    def lat(yy): return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
    return lon(x), lat(y + 1), lon(x + 1), lat(y)


def merc(lon, lat):
    R = 6378137.0
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def terrain_tiles(key, z, b, res):
    lon0, lat0, lon1, lat1 = b
    src_tr = from_origin(lon0, lat1, res, res)
    n = 0
    for zz in TILE_Z:
        N = 2 ** zz
        x0, x1 = int((lon0 + 180) / 360 * N), int((lon1 + 180) / 360 * N)
        def ty(lat): return int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * N)
        y0, y1 = ty(lat1), ty(lat0)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                w, s, e, nn = tile_bounds(zz, x, y)
                mx0, my0 = merc(w, s)
                mx1, my1 = merc(e, nn)
                dst = np.zeros((256, 256), np.float32)
                reproject(z, dst, src_transform=src_tr, src_crs="EPSG:4326",
                          dst_transform=from_bounds(mx0, my0, mx1, my1, 256, 256), dst_crs="EPSG:3857",
                          resampling=Resampling.bilinear, src_nodata=None, dst_nodata=0)
                v = np.clip(dst + 32768.0, 0, 65535.99)
                r = np.floor(v / 256)
                g = np.floor(v - r * 256)
                bb = np.floor((v - np.floor(v)) * 256)
                p = DATA / "terrain" / str(zz) / str(x)
                p.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.dstack([r, g, bb]).astype(np.uint8)).save(p / f"{y}.png", optimize=True)
                n += 1
    print(f"  terrain: {n} tiles", flush=True)


def elev_at(z, b, res):
    lon0, lat0, lon1, lat1 = b
    def f(lon, lat):
        r = np.clip(((lat1 - np.asarray(lat)) / res).astype(int), 0, z.shape[0] - 1)
        c = np.clip(((np.asarray(lon) - lon0) / res).astype(int), 0, z.shape[1] - 1)
        return z[r, c]
    return f


# ---------------------------------------------------------------- water
def water_mesh(S, wet):
    """Quads on every ever-wet cell (dilated by one), vertices at cell corners. Each vertex: lon, lat, ground (m),
    texture u, v (corner position in the frame grid). Ground at a corner = mean of the touching cells."""
    g = S["g"]
    ny, nx = wet.shape
    m = wet.copy()
    m[1:] |= wet[:-1]; m[:-1] |= wet[1:]; m[:, 1:] |= m[:, :-1].copy(); m[:, :-1] |= m[:, 1:].copy()
    zc = S["z"].astype(np.float64)
    zp = np.pad(zc, 1, mode="edge")
    zcorner = 0.25 * (zp[:-1, :-1] + zp[:-1, 1:] + zp[1:, :-1] + zp[1:, 1:])        # (ny+1, nx+1)
    rr, cc = np.nonzero(m)
    used = np.zeros((ny + 1, nx + 1), bool)
    for dr in (0, 1):
        for dc in (0, 1):
            used[rr + dr, cc + dc] = True
    vid = -np.ones((ny + 1, nx + 1), np.int64)
    vr, vc = np.nonzero(used)
    vid[vr, vc] = np.arange(vr.size)
    res = g.lon[1] - g.lon[0]
    lon = g.lon[0] - res / 2 + vc * res
    lat = g.lat[0] + res / 2 - vr * res
    pos = np.stack([lon, lat, zcorner[vr, vc]], 1)
    uv = np.stack([vc / nx, vr / ny], 1)
    a, b_, c, d = vid[rr, cc], vid[rr, cc + 1], vid[rr + 1, cc], vid[rr + 1, cc + 1]
    idx = np.stack([a, c, b_, b_, c, d], 1).reshape(-1).astype(np.uint32)
    return pos, uv, idx


def frame_png(h, uv):
    R = np.clip(np.round(np.sqrt(np.clip(h, 0, H_MAX) / H_MAX) * 255), 0, 255)
    G = np.clip(np.round(128 + uv[0] * V_SCALE), 1, 255)
    B = np.clip(np.round(128 + uv[1] * V_SCALE), 1, 255)
    dry = h < 0.02
    G[dry] = 128; B[dry] = 128
    buf = io.BytesIO()
    Image.fromarray(np.dstack([R, G, B]).astype(np.uint8), "RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def outline(mask, lat, lon, simplify=1.0):
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    res = lon[1] - lon[0]
    tr = from_origin(lon[0] - res / 2, lat[0] + res / 2, res, res)
    polys = [shape(gg) for gg, v in shapes(mask.astype(np.uint8), mask=mask, transform=tr) if v == 1]
    if not polys:
        return None
    return mapping(unary_union(polys).simplify(res * simplify))


# ---------------------------------------------------------------- one city
def city(key):
    c = config.CITIES[key]
    print(f"== {c.name}", flush=True)
    summ = json.loads((config.RESULTS / f"{key}.json").read_text())
    S = RF.setup(c)
    g = S["g"]
    d = DATA / key
    d.mkdir(parents=True, exist_ok=True)
    runs = {s: np.load(PROC6 / key / f"gauge_{s}.npz") for s in SOURCES}
    res = g.lon[1] - g.lon[0]
    grid_b = [g.lon[0] - res / 2, g.lat[-1] - res / 2, g.lon[-1] + res / 2, g.lat[0] + res / 2]
    pad = PAD[key]
    zr, rb, rres = dem_region(key, (grid_b[0] - pad, grid_b[1] - pad, grid_b[2] + pad, grid_b[3] + pad))
    terrain_tiles(key, zr, rb, rres)
    elev = elev_at(zr, rb, rres)

    # water mesh + frames
    hmax = np.max([runs[s]["hmax"].astype(np.float32) for s in SOURCES], 0)
    wet = hmax >= 0.05
    pos, uvt, idx = water_mesh(S, wet)
    lon64 = pos[:, :2]
    hi = lon64.astype(np.float32)
    lo = (lon64 - hi.astype(np.float64)).astype(np.float32)
    with open(d / "water.bin", "wb") as f:
        f.write(np.array([pos.shape[0], idx.size], np.uint32).tobytes())
        f.write(np.concatenate([hi, pos[:, 2:3].astype(np.float32), lo, uvt.astype(np.float32)], 1)
                .astype(np.float32).tobytes())
        f.write(idx.tobytes())
    print(f"  water mesh: {pos.shape[0]} vertices, {idx.size // 3} triangles", flush=True)
    nfr = min(runs[s]["h"].shape[0] for s in SOURCES)
    for s in SOURCES:
        (d / s).mkdir(exist_ok=True)
        for i in range(nfr):
            (d / s / f"{i:03d}.png").write_bytes(frame_png(runs[s]["h"][i].astype(np.float32),
                                                           runs[s]["uv"][i].astype(np.float32)))
    print(f"  {nfr} frames per source", flush=True)

    # rain on a ~500 m grid (block means of the flood-grid rain the model used)
    k = max(1, int(round(500 / g.dy)))
    ny, nx = (g.z.shape[0] // k), (g.z.shape[1] // k)
    nh = runs["truth"]["rain"].shape[0]
    rain_q = {}
    for s in SOURCES:
        r = runs[s]["rain"].astype(np.float32)[:, : ny * k, : nx * k].reshape(nh, ny, k, nx, k).mean((2, 4))
        rain_q[s] = np.clip(np.round(r * 2), 0, 255).astype(np.uint8)
    with open(d / "rain.bin", "wb") as f:
        for s in SOURCES:
            f.write(rain_q[s].tobytes())
    rain_b = [grid_b[0], grid_b[3] - ny * k * res, grid_b[0] + nx * k * res, grid_b[3]]
    rain_ground = S["z"][: ny * k, : nx * k].reshape(ny, k, nx, k).mean((1, 3))

    # series
    bkey = c.outlets[0].key
    bmask = S["basins"][bkey]["mask"]
    zone = S["zone"]
    built_ha = hazard.built_area_ha(g.built, S["cell"])
    town = zone & (g.built > 0.2)
    th = {int(T): v["threshold_mm"] for T, v in summ["thresholds"].items() if v["threshold_mm"]}
    labels, names = S["labels"], S["names"]
    gsum = summ["scenarios"]["gauge"]
    tflood = gsum["truth"]["zone_flood_s"]
    series, communes_live = {}, {}
    # communes that matter: any built-up land under 30 cm at some point, any source
    flooded_any = np.zeros_like(wet)
    for s in SOURCES:
        flooded_any |= runs[s]["hmax"].astype(np.float32) >= 0.3
    ha_per = np.bincount(labels[flooded_any].ravel(), weights=built_ha[flooded_any].ravel(), minlength=len(names) + 1)
    top = [int(i) for i in np.argsort(-ha_per) if i > 0 and ha_per[i] > 0.5][:10]
    for s in SOURCES:
        R = runs[s]
        rf = R["rain"].astype(np.float32)
        basin_rain = rf[:, bmask].mean(1)
        town_rain = rf[:, town].mean(1) if town.any() else basin_rain
        ratio = np.zeros(nh)
        for T, thr in th.items():
            for ser in (basin_rain, town_rain):
                roll = np.convolve(ser, np.ones(T), "full")[:nh]
                ratio = np.maximum(ratio, roll / thr)
        q = R["q"]
        q10 = q[: q.size // 10 * 10].reshape(-1, 10).mean(1)
        fl_ha, zone_ha, com = [], [], []
        for i in range(nfr):
            hh = R["h"][i].astype(np.float32) >= 0.3
            fl_ha.append(float(built_ha[hh].sum()))
            zone_ha.append(float(built_ha[hh & zone].sum()))
            per = np.bincount(labels[hh].ravel(), weights=built_ha[hh].ravel(), minlength=len(names) + 1)
            com.append([round(float(per[j]), 1) for j in top])
        sf = gsum[s]["zone_flood_s"]
        lat_min = config.LATENCY_MIN[s]
        series[s] = {
            "label": LABEL[s], "latency_min": lat_min,
            "basin_rain": [round(float(v), 2) for v in basin_rain],
            "ratio": [round(float(v), 3) for v in ratio],
            "q": [round(float(v), 2) for v in q10],
            "flooded_ha": [round(v, 1) for v in fl_ha], "zone_ha": [round(v, 1) for v in zone_ha],
            "communes": com,
            "peak_q": round(float(q.max()), 1),
            "alarm_model_s": None if sf is None else sf + lat_min * 60.0,
            "alarm_model_lead_min": gsum[s]["model_alarm_lead_min"],
            "alarm_rain_time": gsum[s]["alarm_time"], "alarm_rain_lead_min": gsum[s]["alarm_lead_min"],
            "risk": [{"commune": r["commune"], "ha": round(r["built_ha_at_risk"], 1), "share": round(r["share"], 3)}
                     for r in gsum[s]["communes_top"][:6] if r["built_ha_at_risk"] > 0.05],
        }

    # network (towers, links, hourly rain per link; gauge-matched like everything else)
    z4 = np.load(config.PHASE4 / f"sim_{key}.npz", allow_pickle=True)
    sites = pd.read_csv(ROOT / "data" / "processed" / "algeria_sites.csv")
    sites = sites[sites.lat.between(grid_b[1], grid_b[3]) & sites.lon.between(grid_b[0], grid_b[2])]
    ops = {"Mobilis": 0, "Djezzy": 1, "Ooredoo": 2}
    links = pd.read_csv(ROOT / "results" / f"phase4_links_{key}.csv")
    ids = sorted([f"{l}_1" for l in links.link_id] + [f"{l}_2" for l in links.link_id])
    pos_id = {x: i for i, x in enumerate(ids)}
    def per_link(a):
        return np.nanmean([[a[pos_id[f"{l}_1"]], a[pos_id[f"{l}_2"]]] for l in links.link_id], axis=1)
    fac = summ["gauge_factor"]
    lr = {"truth": per_link(z4["link_true_h"]), "ai": per_link(z4["link_ai_h"]), "classical": per_link(z4["link_cl_h"])}
    lr = {s: np.clip(np.round(np.nan_to_num(v) * fac * 2), 0, 255).astype(np.uint8) for s, v in lr.items()}
    net = {"sites": [[round(float(r.lon), 5), round(float(r.lat), 5), round(float(e), 1), ops.get(r.operator, 0)]
                     for r, e in zip(sites.itertuples(), elev(sites.lon.values, sites.lat.values))],
           "links": [[round(float(a), 5), round(float(b_), 5), round(float(e0), 1), round(float(cc), 5),
                      round(float(dd), 5), round(float(e1), 1)]
                     for a, b_, cc, dd, e0, e1 in zip(links.site_0_lon, links.site_0_lat, links.site_1_lon,
                                                      links.site_1_lat,
                                                      elev(links.site_0_lon.values, links.site_0_lat.values),
                                                      elev(links.site_1_lon.values, links.site_1_lat.values))],
           "hours": int(lr["truth"].shape[1])}
    (d / "net.json").write_text(json.dumps(net, separators=(",", ":")))
    with open(d / "linkrain.bin", "wb") as f:
        for s in ("truth", "ai", "classical"):
            f.write(np.ascontiguousarray(lr[s].T).tobytes())          # (hours, links)

    # outlines and places
    shapes_ = {"basin": outline(bmask, g.lat, g.lon), "zone": outline(zone, g.lat, g.lon, 2),
               "towerbox": outline(RF._tower_box(rain.Phase4Rain(key), g), g.lat, g.lon)}
    com_f = []
    for j in top:
        geom = outline(labels == j, g.lat, g.lon, 3)
        if geom:
            from shapely.geometry import shape as _shape
            pt = _shape(geom).representative_point()
            com_f.append({"type": "Feature", "properties": {"name": names[j - 1], "pt": [pt.x, pt.y,
                                                                                         float(elev(pt.x, pt.y))]},
                          "geometry": geom})
    # deepest built-up flooding (true rain) = the place to look at
    hb = runs["truth"]["hmax"].astype(np.float32) * (g.built > 0.3)
    r0, c0 = np.unravel_index(np.argmax(hb), hb.shape)
    ob = S["basins"][bkey]
    hours0 = pd.Timestamp(summ["hours0"])
    peak_h = int(np.argmax(series["truth"]["q"]) * 10 / 60)
    return {
        "key": key, "name": c.name, "event": EVENTS[key], "hours0": str(hours0), "nframes": nfr, "nhours": nh,
        "grid": [int(g.z.shape[0]), int(g.z.shape[1])], "cell_m": round(float(g.dy)),
        "grid_bounds": grid_b, "rain": {"ny": ny, "nx": nx, "bounds": rain_b, "nh": nh,
                                          "ground": [round(float(v), 1) for v in rain_ground.ravel()]},
        "terrain_bounds": list(rb), "h_max": H_MAX, "v_scale": V_SCALE,
        "basin": {"name": c.outlets[0].name, "area_km2": round(ob["area_km2"], 1), "outlet": [ob["lon"], ob["lat"]],
                  "tower_share": round(summ["basins"][bkey]["tower_box_share"], 3)},
        "observed": c.observed.get(bkey, {}), "gauge_mm": c.gauge_total_mm, "gauge_note": c.gauge_note,
        "gauge_factor": round(summ["gauge_factor"], 2), "alarm_ha": config.ALARM_HA,
        "alarm_communes": list(c.alarm_communes), "town_flood_s": tflood,
        "thresholds": {str(T): round(v, 1) for T, v in th.items()},
        "series": series, "q_step_min": 10, "communes_top": [names[j - 1] for j in top],
        "shapes": shapes_, "communes": {"type": "FeatureCollection", "features": com_f},
        "hotspot": [float(g.lon[c0]), float(g.lat[r0]), float(S["z"][r0, c0])], "peak_hour": peak_h,
        "town_pt": [float(g.lon[np.nonzero(town)[1]].mean()), float(g.lat[np.nonzero(town)[0]].mean())],
        "sites_n": len(net["sites"]), "links_n": len(net["links"]),
    }


def numbers(cities):
    r"""Macros for the explainer (Part 6): \Psix<City><Name>."""
    lines = ["% Phase 6 numbers, written by scripts/phase6/export_dashboard.py. Do not edit by hand."]
    for c in cities:
        C = "Alg" if c["key"] == "algiers" else "Gha"
        d = DATA / c["key"]
        nv = int(np.frombuffer((d / "water.bin").read_bytes()[:4], np.uint32)[0]) if (d / "water.bin").exists() else 0
        mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6 if d.exists() else 0
        lead = lambda s: c["series"][s]["alarm_model_lead_min"]
        vals = {"Sites": f"{c['sites_n']:,}", "Links": f"{c['links_n']:,}", "Frames": c["nframes"],
                "Cell": c["cell_m"], "Vertices": f"{round(nv / 1000):,}", "Mb": f"{mb:.0f}",
                "TowerShare": str(int(100 * c["basin"]["tower_share"] + 0.5)),
                "PeakTruth": f"{c['series']['truth']['peak_q']:.0f}"}
        for s in ("ai", "classical", "imerg"):
            v = lead(s)
            vals["Lead" + {"ai": "Ai", "classical": "Cl", "imerg": "Sat"}[s]] = "none" if v is None else f"{v:+.0f}"
        for k, v in vals.items():
            lines.append(f"\\newcommand{{\\Psix{C}{k}}}{{{v}}}%")
    tiles = sum(1 for _ in (DATA / "terrain").rglob("*.png")) if (DATA / "terrain").exists() else 0
    lines.append(f"\\newcommand{{\\PsixTiles}}{{{tiles}}}%")
    (ROOT / "docs" / "explainer" / "p6_numbers.tex").write_text("\n".join(lines) + "\n")


def main():
    """--html-only: rebuild the page from the cached city summaries (after editing the template)."""
    keys = [a for a in sys.argv[1:] if not a.startswith("--")] or list(config.CITIES)
    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA / "cities.json"
    old = {c["key"]: c for c in json.loads(cache.read_text())} if cache.exists() else {}
    if "--html-only" not in sys.argv:
        for k in keys:
            if (PROC6 / k / "gauge_truth.npz").exists():
                old[k] = json.loads(json.dumps(city(k), default=float))
        cache.write_text(json.dumps(list(old.values()), default=float))
    cities = [old[k] for k in config.CITIES if k in old]
    numbers(cities)
    html = (OUT / "template.html").read_text().replace("/*DATA*/null", json.dumps(cities, separators=(",", ":"), default=float))
    (OUT / "index.html").write_text(html)
    print("wrote", OUT / "index.html", f"{len(html) / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
