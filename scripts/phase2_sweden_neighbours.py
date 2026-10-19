"""Neighbour check on Sweden (OpenMRG, August 2015), both methods, scored like Phase 1/2.

    python scripts/phase2_sweden_neighbours.py

Uses the validation-selected AI model's saved test predictions
(runs/phase2/selected_test_pred.nc) and the classical method with Phase 1 settings,
on the same 683 sub-links. Writes results/phase2_sweden_neighbours.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_openmrg as P2  # noqa: E402

from targ import neighbours as nb  # noqa: E402
from targ.classical import ClassicalParams, retrieve  # noqa: E402
from targ.data import openmrg  # noqa: E402
from targ.evaluation import OPENMRG_SPLITS, hourly  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def per_link_median_r(est, ref):
    rs = []
    for c in est.cml_id.values:
        x, y = est.sel(cml_id=c).values, ref.sel(cml_id=c).values
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 20 and x[ok].std() > 0 and y[ok].std() > 0:
            rs.append(np.corrcoef(x[ok], y[ok])[0, 1])
    return float(np.median(rs))


def main():
    p1 = json.loads((ROOT / "results" / "phase1_openmrg.json").read_text())
    sel = json.loads((ROOT / "results" / "phase2_seeds.json").read_text())["selected"]
    pred = xr.open_dataset(ROOT / "runs" / "phase2" / "selected_test_pred.nc").load()
    ids = pred.cml_id.values
    ds = openmrg.load_cml_1min().sel(cml_id=ids, time=OPENMRG_SPLITS["test"]).load()
    radar = openmrg.radar_along_cmls(ds).sel(cml_id=ids, time=OPENMRG_SPLITS["test"]).load()
    gauges = openmrg.load_city_gauges()
    pairs = openmrg.gauge_link_pairs(ds, gauges, 1000)
    cl = retrieve(ds, ClassicalParams(**p1["params"]))
    miss = np.isnan(cl.tl.values)
    ai_R = np.where(miss, np.nan, pred.R.values)
    ai_w = np.where(miss, np.nan, (pred.prob.values > sel["threshold"]).astype(float))
    est = {"classical": (cl.wet.values, cl.R.values), "ai": (ai_w, ai_R)}
    A = nb.neighbour_matrix(ds.site_0_lat.values, ds.site_0_lon.values,
                            ds.site_1_lat.values, ds.site_1_lon.values)
    info = {}
    for m in ("classical", "ai"):
        w, r = est[m]
        r2, info[m] = nb.neighbour_check(r, A, step_min=1)
        removed = (np.nan_to_num(r) >= 0.1) & (r2 == 0)
        est[m + "_nb"] = (np.where(removed, 0.0, w), r2)
    res = {"neighbour_check": info, "n_sublinks": int(len(ids))}
    rh = hourly(radar)
    for k, (w, r) in est.items():
        W, R = P2.as_da(w, ds, 0, len(ds.time)), P2.as_da(r, ds, 0, len(ds.time))
        res[k] = P2.scores(W, R, radar, gauges, pairs)
        Rh = hourly(R)
        res[k]["robust"] = {"median_per_link_hourly_r_vs_radar": per_link_median_r(Rh, rh),
                            "hours_over_100mm_h": int(np.nansum(Rh.values > 100))}
        t = res[k]
        print(k, "MCC", round(t["wet_dry_vs_radar_5min"]["MCC"], 3),
              "gauges r/RMSE/bias", round(t["rain_vs_gauges_hourly"]["r"], 3),
              round(t["rain_vs_gauges_hourly"]["RMSE"], 3), round(t["rain_vs_gauges_hourly"]["bias_pct"], 1),
              "radar h r", round(t["rain_vs_radar_hourly"]["r"], 3), t["robust"], flush=True)
    print(info)
    (ROOT / "results" / "phase2_sweden_neighbours.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
