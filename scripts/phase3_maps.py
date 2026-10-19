"""Phase 3a: rain maps from links (IDW), checked where the links cannot cheat.

    python scripts/phase3_maps.py

Both methods (classical and AI), after the Phase 2 neighbour check. Hourly maps.

Sweden, August 2015 (683 sub-links): map value at each of the 10 city gauges vs the
  gauge; the weather radar at the same gauges as a yardstick.
Italy, May-Sep 2021 and 2022 (reliable links): map value at every AWS rain gauge
  vs the gauge, grouped by distance to the nearest link.
Netherlands, 1 Jun - 22 Aug 2014 (reliable links): map vs KNMI gauge-adjusted radar
  on its 1-km grid (every 2nd pixel), for pixels within 5 km of a link.

Writes results/phase3_maps.json, runs/phase3/*.nc and figures for the explainer.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from targ import maps  # noqa: E402
from targ import neighbours as nb  # noqa: E402
from targ.classical import ClassicalParams, retrieve  # noqa: E402
from targ.data import netherlands, openmrg, openrainer  # noqa: E402
from targ.evaluation import OPENMRG_SPLITS, hourly, rain_scores, wet_dry_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS2, RUNS3 = ROOT / "runs" / "phase2", ROOT / "runs" / "phase3"
OUT = ROOT / "results"


def scores(ref, est):
    ref, est = np.ravel(ref), np.ravel(est)
    out = {"rain": rain_scores(ref, est)}
    ok = np.isfinite(ref) & np.isfinite(est)
    out["wet_dry"] = wet_dry_scores(ref[ok], (est[ok] >= 0.1).astype(float))
    hit = ((ref >= 1) & (est >= 1))[ok].sum()
    miss = ((ref >= 1) & (est < 1))[ok].sum()
    fa = ((ref < 1) & (est >= 1))[ok].sum()
    out["csi_1mm_h"] = float(hit / max(1, hit + miss + fa))
    return out


def map_to_points(R_hourly: xr.DataArray, ds_coords, lat, lon):
    """Hourly sub-link rain -> IDW values at points. Returns (values (n_pts, T), dist_km to nearest path)."""
    Rp, mlat, mlon = maps.merge_directions(R_hourly.values, ds_coords["site_0_lat"], ds_coords["site_0_lon"],
                                           ds_coords["site_1_lat"], ds_coords["site_1_lon"])
    lat0, lon0 = float(np.mean(mlat)), float(np.mean(mlon))
    src = np.column_stack(maps.to_km(mlat, mlon, lat0, lon0))
    dst = np.column_stack(maps.to_km(lat, lon, lat0, lon0))
    W = maps.idw_weights(src, dst)
    from scipy.spatial import cKDTree
    dist = cKDTree(src).query(dst)[0]
    return maps.apply_idw(W, Rp), dist


def coords_of(ds):
    return {c: ds[c].values for c in ("site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon")}


# ---------------------------------------------------------------- Sweden
def sweden():
    p1 = json.loads((OUT / "phase1_openmrg.json").read_text())
    sel = json.loads((OUT / "phase2_seeds.json").read_text())["selected"]
    pred = xr.open_dataset(RUNS2 / "selected_test_pred.nc").load()
    ids = pred.cml_id.values
    ds = openmrg.load_cml_1min().sel(cml_id=ids, time=OPENMRG_SPLITS["test"]).load()
    cl = retrieve(ds, ClassicalParams(**p1["params"]))
    miss = np.isnan(cl.tl.values)
    A = nb.neighbour_matrix(*(coords_of(ds)[c] for c in ("site_0_lat", "site_0_lon", "site_1_lat", "site_1_lon")))
    est = {}
    for m, R in (("classical", cl.R.values), ("ai", np.where(miss, np.nan, pred.R.values))):
        est[m] = xr.DataArray(nb.neighbour_check(R, A, step_min=1)[0], dims=("cml_id", "time"),
                              coords={"cml_id": ids, "time": ds.time.values})
    g = openmrg.load_city_gauges()
    gh = hourly(g.rainfall_rate).sel(time=OPENMRG_SPLITS["test"])
    out = {"n_gauges": int(g.sizes["id"])}
    for m, R in est.items():
        vals, dist = map_to_points(hourly(R), coords_of(ds), g.lat.values, g.lon.values)
        hours = hourly(R).time.values
        ref = gh.sel(time=hours).transpose("id", "time").values
        out[m] = scores(ref, vals)
        out["gauge_distance_to_nearest_link_km"] = [float(np.min(dist)), float(np.median(dist)), float(np.max(dist))]
        if m == "ai":
            sweden_vals = (hours, vals)
    # yardstick: the weather radar at the gauges (pixel containing the gauge)
    import phase1_baseline as P1
    rad = P1.radar_at_gauges(g).sel(time=OPENMRG_SPLITS["test"])
    out["radar_at_gauges"] = scores(gh.sel(time=rad.time).transpose("id", "time").values,
                                    rad.transpose("id", "time").values)
    # time series for the explainer: network-mean of gauges vs maps at gauges
    hours, vals = sweden_vals
    df = pd.DataFrame({"time": hours, "gauges": np.nanmean(gh.sel(time=hours).values, axis=0),
                       "ai_map": np.nanmean(vals, axis=0)})
    df.to_csv(RUNS3 / "sweden_gauge_mean.csv", index=False)
    return out


# ---------------------------------------------------------------- Italy
def hourly_end(x: xr.DataArray) -> xr.DataArray:
    m = x.resample(time="1h", label="right", closed="right").mean()
    n = x.notnull().resample(time="1h", label="right", closed="right").sum()
    return m.where(n == 4)


def italy():
    out = {}
    for season, months in (("2021", [f"2021{m:02d}" for m in range(5, 10)]),
                           ("2022", [f"2022{m:02d}" for m in range(5, 10)])):
        d = xr.open_dataset(RUNS2 / f"italy_{season}_15min.nc").load()
        d = d.sel(cml_id=d.cml_id.values[d.reliable.values])
        g = openrainer.gauges(months)
        gh = hourly_end(g.rainfall_rate)
        res = {"n_links": int(d.sizes["cml_id"]), "n_gauges": int(g.sizes["id"])}
        for m in ("classical_nb", "ai_nb"):
            Rh = hourly_end(d[m])
            vals, dist = map_to_points(Rh, coords_of(d), g.lat.values, g.lon.values)
            ref = gh.reindex(time=Rh.time.values).transpose("id", "time").values
            for lo, hi in ((0, 2), (2, 5), (5, 10)):
                sel = (dist >= lo) & (dist < hi)
                res.setdefault(m.replace("_nb", ""), {})[f"gauges_{lo}_{hi}_km"] = {
                    "n_gauges": int(sel.sum()), **scores(ref[sel], vals[sel])}
        res["gauges_beyond_10_km_no_map"] = int((dist >= 10).sum())
        out[season] = res
    return out


# ---------------------------------------------------------------- Netherlands
def nl():
    d = xr.open_dataset(RUNS2 / "nl_15min.nc").load()
    d = d.sel(cml_id=d.cml_id.values[d.reliable.values])
    # KNMI 1-km grid, every 2nd pixel
    proj = __import__("pyproj").Proj(netherlands.KNMI_PROJ)
    rows, cols = np.arange(0, 765, 2), np.arange(0, 700, 2)
    xx, yy = np.meshgrid(cols + 0.5, -3650.0 - (rows + 0.5))
    lon, lat = proj(xx.ravel(), yy.ravel(), inverse=True)
    hours = None
    res = {"n_links": int(d.sizes["cml_id"])}
    est = {}
    for m in ("classical_nb", "ai_nb"):
        Rh = hourly_end(d[m])
        hours = Rh.time.values
        vals, dist = map_to_points(Rh, coords_of(d), lat, lon)
        est[m] = vals
    inside = dist <= 5.0
    # radar at the same pixels
    ok_hours, frames = [], []
    with zipfile.ZipFile(netherlands.KNMI_ZIP) as z:
        names = set(z.namelist())
        import h5py
        for t in pd.DatetimeIndex(hours):
            n = f"RAD_NL25_RAC_MFBS_01H/{t:%Y}/{t:%m}/RAD_NL25_RAC_MFBS_01H_{t:%Y%m%d%H%M}_NL.h5"
            if n not in names:
                frames.append(None)
                continue
            with z.open(n) as f, h5py.File(f) as h:
                pv = h["image1/image_data"][()][::2, ::2].astype(np.float32)
            pv[pv == 65535] = np.nan
            frames.append((pv * 0.01).ravel())
    radar = np.stack([f if f is not None else np.full(len(lat), np.nan) for f in frames], axis=1)
    res["n_pixels_within_5km"] = int(inside.sum())
    for m, vals in est.items():
        res[m.replace("_nb", "")] = scores(radar[inside], vals[inside])
    # save one stormy hour for the explainer figure
    tot = np.nansum(np.where(inside[:, None], radar, 0), axis=0)
    k = int(np.nanargmax(tot))
    shape = xx.shape
    RUNS3.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"radar": (("y", "x"), radar[:, k].reshape(shape)),
                "ai_map": (("y", "x"), np.where(inside, est["ai_nb"][:, k], np.nan).reshape(shape)),
                "classical_map": (("y", "x"), np.where(inside, est["classical_nb"][:, k], np.nan).reshape(shape)),
                "inside": (("y", "x"), inside.reshape(shape))},
               coords={"lon": (("y", "x"), lon.reshape(shape)), "lat": (("y", "x"), lat.reshape(shape))},
               attrs={"time_end": str(pd.Timestamp(hours[k]))}).to_netcdf(RUNS3 / "nl_storm_hour.nc")
    res["example_hour_end"] = str(pd.Timestamp(hours[k]))
    return res


def main():
    RUNS3.mkdir(parents=True, exist_ok=True)
    res = {"method": {"interpolation": "inverse-distance weighting of path midpoints",
                      "nnear": maps.NNEAR, "power": maps.POWER, "max_km": maps.MAX_KM,
                      "estimates": "Phase 2 classical and AI, after the neighbour check, hourly"}}
    for name, fn in (("sweden", sweden), ("italy", italy), ("netherlands", nl)):
        print("==", name, flush=True)
        res[name] = fn()
        print(json.dumps(res[name], indent=1, default=float)[:2500], flush=True)
    (OUT / "phase3_maps.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
