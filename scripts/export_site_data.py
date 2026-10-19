"""Export the real Phase 1 numbers the project website needs → site/data.json.

    python scripts/export_site_data.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import xarray as xr

from targ.classical import ClassicalParams, quality_mask, retrieve, rolling_std, total_loss
from targ.data import openmrg
from targ.evaluation import OPENMRG_CALIB, OPENMRG_SPLITS, hourly, to_radar_times, wet_dry_scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data.json"


def r2(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def main():
    res = json.loads((ROOT / "results" / "phase1_openmrg.json").read_text())
    p = ClassicalParams(**res["params"])
    ds = openmrg.load_cml_1min().sel(time=slice("2015-06-01", "2015-08-31T23:59")).load()
    radar = openmrg.radar_along_cmls(ds).sel(time=slice("2015-06-01", "2015-08-31T23:59")).load()
    gauges = openmrg.load_city_gauges()
    keep = quality_mask(ds.sel(time=OPENMRG_CALIB), p)

    # per-link August scores (all links, so the map can show the faulty ones too)
    test = ds.sel(time=OPENMRG_SPLITS["test"])
    out = retrieve(test, p)
    rad_t = radar.sel(time=OPENMRG_SPLITS["test"])
    Rh, Dh = hourly(out.R), hourly(rad_t)
    links = []
    for c in ds.cml_id.values:
        x, y = Rh.sel(cml_id=c).values, Dh.sel(cml_id=c).values
        ok = ~(np.isnan(x) | np.isnan(y))
        r = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 10 and np.std(x[ok]) > 0 else np.nan
        d = ds.sel(cml_id=c)
        links.append({
            "id": int(c), "link": int(d.link_id), "dir": str(d.direction.values),
            "a": [r2(d.site_0_lon, 5), r2(d.site_0_lat, 5)],
            "b": [r2(d.site_1_lon, 5), r2(d.site_1_lat, 5)],
            "km": r2(d.length, 2), "ghz": r2(d.frequency, 2),
            "qc": bool(keep.sel(cml_id=c)), "r": r2(r),
            "cml_mm": r2(np.nansum(x), 1), "radar_mm": r2(np.nansum(y), 1),
        })

    # wet/dry tuning curves (June–July)
    tl = total_loss(ds.sel(time=OPENMRG_CALIB), p.max_gap_fill_min)
    ref = radar.sel(time=OPENMRG_CALIB)
    tuning = {}
    for w in (15, 30, 60):
        s = xr.DataArray(rolling_std(tl.values, w), coords=tl.coords, dims=tl.dims)
        pts = []
        for thr in np.round(np.arange(0.1, 1.55, 0.05), 2):
            wet = (s > thr).astype(float).where(s.notnull())
            sc = wet_dry_scores(ref.values, to_radar_times(wet, ref.time.values).values)
            pts.append([float(thr), r2(sc["MCC"])])
        tuning[str(w)] = pts

    # storm replay: sub-link closest to a gauge, wettest August day (same as phase-1 figure)
    pairs = openmrg.gauge_link_pairs(ds.sel(cml_id=keep.cml_id.values[keep.values]), gauges, 1000)
    best = pairs.sort_values("distance_m").iloc[0]
    g = gauges.rainfall_rate.sel(id=best.gauge, time=OPENMRG_SPLITS["test"])
    day = g.resample(time="1D").sum().idxmax().values
    sl = slice(day, day + np.timedelta64(1, "D") - np.timedelta64(1, "m"))
    o = out.sel(cml_id=best.cml_id, time=sl)
    g5 = g.sel(time=sl).rolling(time=5, center=True, min_periods=1).mean()
    rad_s = rad_t.sel(cml_id=best.cml_id, time=sl)
    storm = {
        "cml_id": int(best.cml_id), "gauge": str(best.gauge),
        "gauge_distance_m": r2(best.distance_m, 0),
        "km": r2(o.length, 2), "ghz": r2(o.frequency, 2), "date": str(day)[:10],
        "t0": str(o.time.values[0])[:16],
        "tl": [r2(v, 2) for v in o.tl.values],
        "baseline": [r2(v, 2) for v in o.baseline.values],
        "wet": [int(v) if np.isfinite(v) else 0 for v in o.wet.values],
        "R": [r2(v, 2) for v in o.R.rolling(time=5, center=True, min_periods=1).mean().values],
        "gauge_R": [r2(v, 2) for v in g5.values],
        "radar": [[int((t - o.time.values[0]) / np.timedelta64(1, "m")), r2(v, 2)]
                  for t, v in zip(rad_s.time.values, rad_s.values)],
    }

    data = {
        "results": res,
        "links": links,
        "gauges": [{"id": str(i), "lon": r2(lo, 5), "lat": r2(la, 5), "type": str(t)}
                   for i, lo, la, t in zip(gauges.id.values, gauges.lon.values,
                                           gauges.lat.values, gauges.gauge_type.values)],
        "tuning": tuning,
        "storm": storm,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB); storm: {storm['date']} "
          f"sub-link {storm['cml_id']} near {storm['gauge']}")


if __name__ == "__main__":
    main()
