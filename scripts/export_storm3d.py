"""Data for the 3D storm view (site/storm3d/index.html): terrain, ground image, towers,
links and hourly rain for Algiers and Ghardaia, packed into one self-contained page.

    python scripts/export_storm3d.py

Terrain: Copernicus GLO-30 DEM (public, no key), read straight from its cloud-optimised
GeoTIFFs over the city box, resampled to ~60 m and encoded as a Terrarium PNG.
Ground image: hillshade + sea + coast + commune borders, rendered here (PNG).
Rain: the Phase 4 hourly fields (simulated truth and AI map, 500 m), quantised to
0.25 mm/h steps. Links: hourly rain along each link (truth and AI), mean of both
directions. Everything is embedded, so the page loads no tiles from anywhere.
Also writes towers_<city>.geojson and links_<city>.geojson (3D coordinates) for GeoLibre
or any GIS: GeoLibre (geolibre.app) is built on the same MapLibre + deck.gl engines.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.merge import merge

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "data" / "processed" / "phase4"
RES = ROOT / "results"
OUT = ROOT / "site" / "storm3d"
DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_DEM/"
           "Copernicus_DSM_COG_10_{t}_DEM.tif")
CITIES = {"algiers": {"name": "Algiers", "event": "Bab El Oued flood, 9–10 Nov 2001",
                      "tiles": ["N36_00_E002_00", "N36_00_E003_00"]},
          "ghardaia": {"name": "Ghardaïa", "event": "M'zab valley flood, 29 Sep – 1 Oct 2008",
                       "tiles": ["N32_00_E003_00"]}}
DEM_STEP = 0.0006       # ~60 m


def png_data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def b64_u8(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def read_dem(tiles, bounds):
    """bounds = (lon0, lat0, lon1, lat1). Returns elevation (rows north->south) in metres."""
    srcs = [rasterio.open(DEM_URL.format(t=t)) for t in tiles]
    arr, _ = merge(srcs, bounds=bounds, res=(DEM_STEP, DEM_STEP), resampling=rasterio.enums.Resampling.average)
    for s in srcs:
        s.close()
    z = arr[0].astype(np.float32)
    z[~np.isfinite(z) | (z < -100)] = 0.0
    return z


def terrarium(z):
    v = np.clip(z + 32768.0, 0, 65535.996)
    r = np.floor(v / 256)
    g = np.floor(v - r * 256)
    b = np.floor((v - np.floor(v)) * 256)
    return np.dstack([r, g, b]).astype(np.uint8)


def ground_image(z, bounds, geo, scale=3):
    """Hillshaded land, dark sea, coast and commune borders. Rows north->south."""
    from matplotlib.path import Path as MPath
    ny, nx = z.shape
    lon0, lat0, lon1, lat1 = bounds
    # hillshade from the NW
    dzdy, dzdx = np.gradient(z, 60.0, 60.0 * np.cos(np.radians((lat0 + lat1) / 2)))
    slope = np.arctan(np.hypot(dzdx, dzdy) * 2.0)
    aspect = np.arctan2(-dzdx, dzdy)
    az, alt = np.radians(315), np.radians(40)
    hs = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)
    # land mask from the commune polygons
    H, W = ny * scale, nx * scale
    yy, xx = np.mgrid[0:H, 0:W]
    lon = lon0 + (xx + 0.5) / W * (lon1 - lon0)
    lat = lat1 - (yy + 0.5) / H * (lat1 - lat0)
    pts = np.c_[lon.ravel(), lat.ravel()]
    land = np.zeros(H * W, bool)
    for ring in geo["land"]:
        land ^= MPath(np.array(ring)).contains_points(pts)       # even-odd: holes cancel
    land = land.reshape(H, W)
    hsu = np.kron(hs, np.ones((scale, scale)))
    zu = np.kron(z, np.ones((scale, scale)))
    # land: dusk slate tinted by height; sea: deep navy
    t = np.clip(zu / max(float(z.max()), 1.0), 0, 1)[..., None]
    low, high = np.array([46, 58, 66]), np.array([92, 98, 96])
    base = low * (1 - t) + high * t
    shade = (0.45 + 0.75 * hsu)[..., None]
    img = np.clip(base * shade, 0, 255)
    sea = np.array([12, 24, 38]) + np.zeros_like(img)
    img = np.where(land[..., None], img, sea)
    from PIL import ImageDraw
    im = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    px = lambda lo, la: ((lo - lon0) / (lon1 - lon0) * W, (lat1 - la) / (lat1 - lat0) * H)
    for line in geo["borders"]:
        d.line([px(*p) for p in line], fill=(170, 190, 205, 70), width=1)
    for line in geo["coast"]:
        d.line([px(*p) for p in line], fill=(190, 212, 228, 130), width=2)
    return png_data_url(np.asarray(im))


def city(key):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import export_phase4_explorer as E
    meta = CITIES[key]
    z4 = np.load(P4 / f"sim_{key}.npz", allow_pickle=True)
    step = float(z4["lat"][1] - z4["lat"][0])
    lat0, lat1 = float(z4["lat"][0]) - step / 2, float(z4["lat"][-1]) + step / 2
    lon0, lon1 = float(z4["lon"][0]) - step / 2, float(z4["lon"][-1]) + step / 2
    bounds = (lon0, lat0, lon1, lat1)
    geo = E.geography(key, [lat0, lat1], [lon0, lon1])
    dem = read_dem(meta["tiles"], bounds)
    print(key, "dem", dem.shape, "range", float(dem.min()), float(dem.max()), flush=True)

    def elev(lon, lat):
        r = np.clip(((lat1 - np.asarray(lat)) / (lat1 - lat0) * dem.shape[0]).astype(int), 0, dem.shape[0] - 1)
        c = np.clip(((np.asarray(lon) - lon0) / (lon1 - lon0) * dem.shape[1]).astype(int), 0, dem.shape[1] - 1)
        return dem[r, c]

    sites = pd.read_csv(ROOT / "data" / "processed" / "algeria_sites.csv")
    sites = sites[sites.lat.between(lat0, lat1) & sites.lon.between(lon0, lon1)].reset_index(drop=True)
    ops = {"Mobilis": 0, "Djezzy": 1, "Ooredoo": 2}
    links = pd.read_csv(RES / f"phase4_links_{key}.csv")
    # hourly rain per link: both directions averaged (sub-links are sorted by cml_id)
    ids = sorted([f"{l}_1" for l in links.link_id] + [f"{l}_2" for l in links.link_id])
    lt, la = z4["link_true_h"], z4["link_ai_h"]
    pos = {c: i for i, c in enumerate(ids)}
    lt_l = np.nanmean([[lt[pos[f"{l}_1"]], lt[pos[f"{l}_2"]]] for l in links.link_id], axis=1)
    la_l = np.nanmean([[la[pos[f"{l}_1"]], la[pos[f"{l}_2"]]] for l in links.link_id], axis=1)
    q = lambda a: np.clip(np.round(np.nan_to_num(a, nan=0.0) * 4), 0, 255)    # 0.25 mm/h steps
    truth, ai = z4["truth"], z4["ai"]
    H, ny, nx = truth.shape
    return {
        "key": key, "name": meta["name"], "event": meta["event"],
        "bounds": [lon0, lat0, lon1, lat1], "zmax": float(dem.max()), "zbase": float(np.floor(dem.min())),
        "terrain": png_data_url(terrarium(dem)), "ground": ground_image(dem, bounds, geo),
        "hours": [str(pd.Timestamp(h))[:16] for h in z4["hours"]], "ny": ny, "nx": nx,
        "truth": b64_u8(q(truth)), "ai": b64_u8(q(np.where(np.isfinite(ai), ai, 0))),
        "sites": [[round(float(r.lon), 5), round(float(r.lat), 5), round(float(e), 1), ops.get(r.operator, 0)]
                  for r, e in zip(sites.itertuples(), elev(sites.lon, sites.lat))],
        "links": [[round(float(a), 5), round(float(b), 5), round(float(ea), 1),
                   round(float(c), 5), round(float(d), 5), round(float(eb), 1)]
                  for a, b, c, d, ea, eb in zip(links.site_0_lon, links.site_0_lat, links.site_1_lon, links.site_1_lat,
                                                elev(links.site_0_lon, links.site_0_lat),
                                                elev(links.site_1_lon, links.site_1_lat))],
        "link_truth": b64_u8(q(lt_l)), "link_ai": b64_u8(q(la_l)),
        "labels": [[lo, la_, name, round(float(elev(lo, la_)), 1)] for lo, la_, name in geo["labels"]],
    }


def geojson(c):
    """Towers (points with ground height) and links (3D lines at mast top) for GeoLibre or any GIS."""
    names = ["Mobilis", "Djezzy", "Ooredoo"]
    towers = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lo, la, e]},
         "properties": {"operator": names[op], "ground_m": e}} for lo, la, e, op in c["sites"]]}
    lt = np.frombuffer(base64.b64decode(c["link_truth"]), np.uint8).reshape(len(c["hours"]), -1) / 4
    links = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[a, b, ea + 30], [cc, d, eb + 30]]},
         "properties": {"storm_total_mm": round(float(lt[:, i].sum()), 1), "peak_mm_h": round(float(lt[:, i].max()), 1)}}
        for i, (a, b, ea, cc, d, eb) in enumerate(c["links"])]}
    (OUT / f"towers_{c['key']}.geojson").write_text(json.dumps(towers))
    (OUT / f"links_{c['key']}.geojson").write_text(json.dumps(links))


def main():
    cities = [city(k) for k in CITIES if (P4 / f"sim_{k}.npz").exists()]
    for c in cities:
        geojson(c)
    html = (OUT / "template.html").read_text().replace("/*DATA*/null", json.dumps(cities, separators=(",", ":")))
    (OUT / "index.html").write_text(html)
    # live version (local site): satellite imagery, terrain and 3D buildings streamed as map tiles
    real = [{k: v for k, v in c.items() if k != "ground"} for c in cities]
    (OUT / "real.html").write_text((OUT / "real_template.html").read_text()
                                   .replace("/*DATA*/null", json.dumps(real, separators=(",", ":"))))
    print("wrote", OUT / "index.html", f"{len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
