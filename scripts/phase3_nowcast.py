"""Phase 3b: nowcasting (0-60 min) from link rain maps, Netherlands 2014.

    python scripts/phase3_nowcast.py

Every full hour with rain in the network: build 15-min AI link maps (Phase 2 AI
after the neighbour check, reliable links, IDW on a 2-km grid), estimate the storm
motion from the last three maps (t-30, t-15, t) with pysteps' Lucas-Kanade optical
flow, and move the map at t forward along that motion for 15, 30, 45 and 60 min
(semi-Lagrangian extrapolation). Baseline: persistence (rain stays where it is).

Checks:
  1. against the link maps observed 15-60 min later (same kind of data)
  2. against KNMI's gauge-adjusted radar for the next hour (independent): the
     forecast's mean rain over (t, t+60] vs the radar hourly accumulation.

Writes results/phase3_nowcast.json and runs/phase3/nowcast_example.nc.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from targ import maps
from targ.data import netherlands

ROOT = Path(__file__).resolve().parents[1]
RUNS2, RUNS3 = ROOT / "runs" / "phase2", ROOT / "runs" / "phase3"
OUT = ROOT / "results"
LEADS = (1, 2, 3, 4)          # x 15 min
MIN_RAIN_PIXELS = 200         # >= 1 mm/h somewhere: ~800 km2 at 2 km
THRESH = (0.5, 1.0, 5.0)      # mm/h, for CSI


def csi(obs, fc, thr):
    ok = np.isfinite(obs) & np.isfinite(fc)
    o, f = obs[ok] >= thr, fc[ok] >= thr
    hits, miss, fa = (o & f).sum(), (o & ~f).sum(), (~o & f).sum()
    return hits, miss, fa


def main():
    from pysteps import extrapolation, motion
    RUNS3.mkdir(parents=True, exist_ok=True)
    d = xr.open_dataset(RUNS2 / "nl_15min.nc").load()
    d = d.sel(cml_id=d.cml_id.values[d.reliable.values])
    Rp, mlat, mlon = maps.merge_directions(d["ai_nb"].values, d.site_0_lat.values, d.site_0_lon.values,
                                           d.site_1_lat.values, d.site_1_lon.values)
    times = pd.DatetimeIndex(d.time.values)

    proj = __import__("pyproj").Proj(netherlands.KNMI_PROJ)
    rows, cols = np.arange(0, 765, 2), np.arange(0, 700, 2)
    xx, yy = np.meshgrid(cols + 0.5, -3650.0 - (rows + 0.5))
    shape = xx.shape
    glon, glat = proj(xx.ravel(), yy.ravel(), inverse=True)
    lat0, lon0 = float(np.mean(mlat)), float(np.mean(mlon))
    src = np.column_stack(maps.to_km(mlat, mlon, lat0, lon0))
    dst = np.column_stack(maps.to_km(glat, glon, lat0, lon0))
    W = maps.idw_weights(src, dst)
    inside = cKDTree(src).query(dst)[0] <= 10.0

    def frame(i):
        m = maps.apply_idw(W, Rp[:, i:i + 1])[:, 0]
        return np.where(inside, m, np.nan).reshape(shape)

    oclock = np.where((times.minute == 0) & (np.arange(len(times)) >= 2) &
                      (np.arange(len(times)) + max(LEADS) < len(times)))[0]
    # quick rain screen on the hourly frames
    now = maps.apply_idw(W, Rp[:, oclock])
    rainy = oclock[np.nansum(now >= 1.0, axis=0) >= MIN_RAIN_PIXELS]
    print(f"{len(oclock)} full hours, {len(rainy)} with rain in the network", flush=True)

    oflow, extrap = motion.get_method("LK"), extrapolation.get_method("semilagrangian")
    acc = {m: {L: {t: [0, 0, 0] for t in THRESH} for L in LEADS} for m in ("nowcast", "persistence")}
    err = {m: {L: [] for L in LEADS} for m in ("nowcast", "persistence")}
    hour_pairs = {"nowcast": [], "persistence": [], "radar": []}
    knmi = zipfile.ZipFile(netherlands.KNMI_ZIP)
    names = set(knmi.namelist())
    import h5py
    example = None
    for n_done, i in enumerate(rainy):
        past = np.stack([frame(i - 2), frame(i - 1), frame(i)])
        if not np.isfinite(past).all(axis=0).any():
            continue
        past_f = np.where(np.isfinite(past), past, 0.0)
        log = np.log10(np.maximum(past_f, 0.01))   # optical flow on log-rain, like pysteps examples
        V = oflow(log)
        fc = extrap(np.where(inside.reshape(shape), past_f[-1], np.nan), V, len(LEADS),
                    allow_nonfinite_values=True)
        obs = [frame(i + L) for L in LEADS]
        for k, L in enumerate(LEADS):
            for name, f in (("nowcast", fc[k]), ("persistence", past[-1])):
                o = obs[k]
                for t in THRESH:
                    h, m_, fa = csi(o, f, t)
                    acc[name][L][t][0] += h
                    acc[name][L][t][1] += m_
                    acc[name][L][t][2] += fa
                ok = np.isfinite(o) & np.isfinite(f) & ((o >= 0.1) | (f >= 0.1))
                err[name][L].append(np.abs(o[ok] - f[ok]).mean() if ok.any() else np.nan)
        # independent check: next-hour mean vs KNMI hourly accumulation ending at t+60
        t_end = times[i] + pd.Timedelta("60min")
        fn = f"RAD_NL25_RAC_MFBS_01H/{t_end:%Y}/{t_end:%m}/RAD_NL25_RAC_MFBS_01H_{t_end:%Y%m%d%H%M}_NL.h5"
        if fn in names:
            with knmi.open(fn) as f, h5py.File(f) as h:
                pv = h["image1/image_data"][()][::2, ::2].astype(np.float32)
            pv[pv == 65535] = np.nan
            rad = pv * 0.01
            nc_hour = np.nanmean(np.stack(list(fc)), axis=0)
            ok = inside.reshape(shape) & np.isfinite(rad) & np.isfinite(nc_hour) & np.isfinite(past[-1])
            hour_pairs["nowcast"].append(nc_hour[ok])
            hour_pairs["persistence"].append(past[-1][ok])
            hour_pairs["radar"].append(rad[ok])
            if example is None or np.nansum(rad[ok] >= 5) > example[0]:
                example = (np.nansum(rad[ok] >= 5), i, past[-1], fc, obs, rad, V)
        if n_done % 100 == 0:
            print(f"  {n_done}/{len(rainy)}", flush=True)

    res = {"n_hours_scored": int(len(rainy)), "grid_km": 2, "leads_min": [15 * L for L in LEADS],
           "motion": "pysteps Lucas-Kanade on log10 rain of the last 3 maps",
           "extrapolation": "pysteps semi-Lagrangian", "vs_link_maps": {}, "vs_radar_next_hour": {}}
    for name in ("nowcast", "persistence"):
        res["vs_link_maps"][name] = {
            f"{15 * L}min": {**{f"csi_{t}": float(a[0] / max(1, sum(a))) for t, a in acc[name][L].items()},
                             "mae_mm_h": float(np.nanmean(err[name][L]))} for L in LEADS}
    radar = np.concatenate(hour_pairs["radar"])
    for name in ("nowcast", "persistence"):
        f = np.concatenate(hour_pairs[name])
        res["vs_radar_next_hour"][name] = {
            **{f"csi_{t}": float((lambda h, m, a: h / max(1, h + m + a))(*csi(radar, f, t))) for t in THRESH},
            "r": float(np.corrcoef(radar, f)[0, 1]),
            "mae_mm_h": float(np.mean(np.abs(radar - f)[(radar >= 0.1) | (f >= 0.1)]))}
    (OUT / "phase3_nowcast.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))

    _, i, now_map, fc, obs, rad, V = example
    xr.Dataset({"now": (("y", "x"), now_map), "fc60": (("y", "x"), fc[-1]), "obs60": (("y", "x"), obs[-1]),
                "radar_next_hour": (("y", "x"), rad), "u": (("y", "x"), V[0]), "v": (("y", "x"), V[1])},
               coords={"lon": (("y", "x"), glon.reshape(shape)), "lat": (("y", "x"), glat.reshape(shape))},
               attrs={"time": str(times[i])}).to_netcdf(RUNS3 / "nowcast_example.nc")


if __name__ == "__main__":
    main()
