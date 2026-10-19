"""How well do the methods measure rain that follows a mostly rainy day?

    python scripts/phase2_longrain.py [tag]

Rainy reference steps are grouped by the share of the previous 24 h that the
reference itself called rainy (>= 0.1 mm/h). For each group: share of all rainy
steps, share caught (method >= 0.1 mm/h), and rain measured as % of the reference
total. Sweden: 5-min radar, August. Italy / Netherlands: hourly radar, on the
sub-links of the earlier quality rule so before/after compare the same links,
without the neighbour check (this table is about the dry level only).

Writes results/phase2_longrain[_tag].json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from targ.classical import ClassicalParams, retrieve
from targ.data import openmrg
from targ.evaluation import OPENMRG_SPLITS, to_radar_times

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "phase2"
BINS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.01)]


def table(ref, ests, wetfrac):
    rainy_all = np.isfinite(ref) & (ref >= 0.1)
    rows = []
    for lo, hi in BINS:
        m = (wetfrac >= lo) & (wetfrac < hi) & rainy_all
        for e in ests.values():
            m &= np.isfinite(e)
        row = {"past24h_wet_from": lo, "past24h_wet_to": min(hi, 1.0), "n": int(m.sum()),
               "share_of_rainy_steps": float(m.sum() / max(1, rainy_all.sum()))}
        for k, e in ests.items():
            row[k] = {"caught": float(np.mean(e[m] >= 0.1)) if m.any() else None,
                      "total_pct": float(100 * e[m].sum() / ref[m].sum()) if m.any() else None}
        rows.append(row)
    return rows


def past_wet_fraction(ref: xr.DataArray, steps: int) -> xr.DataArray:
    wet = (ref >= 0.1).astype(float).where(ref.notnull())
    return wet.rolling(time=steps, min_periods=steps // 2).mean().shift(time=1)


def main(tag: str = ""):
    p1 = json.loads((ROOT / "results" / "phase1_openmrg.json").read_text())
    pred = xr.open_dataset(RUNS / "selected_test_pred.nc").load()   # the validation-selected model
    ids = pred.cml_id.values
    radar = openmrg.radar_along_cmls(openmrg.load_cml_1min()).sel(cml_id=ids).load()
    rt = radar.sel(time=OPENMRG_SPLITS["test"])
    ds = openmrg.load_cml_1min().sel(cml_id=ids, time=OPENMRG_SPLITS["test"]).load()
    cl = retrieve(ds, ClassicalParams(**p1["params"]))
    out = {"sweden_aug2015_5min": table(
        rt.values, {"classical": to_radar_times(cl.R, rt.time.values).values,
                    "ai": to_radar_times(pred.R, rt.time.values).values},
        past_wet_fraction(radar, 288).sel(time=rt.time).values)}
    for f, name in (("italy_2021_hourly.nc", "italy_2021_hourly"), ("italy_2022_hourly.nc", "italy_2022_hourly"),
                    ("nl_hourly.nc", "netherlands_2014_hourly")):
        h = xr.open_dataset(RUNS / f).load()
        if "old_rule" in h.coords:   # same sub-links as the "before" table
            h = h.sel(cml_id=h.old_rule.values)
        out[name] = table(h.radar.values, {"classical": h.classical.values, "ai": h.ai.values},
                          past_wet_fraction(h.radar, 24).values)
    path = ROOT / "results" / f"phase2_longrain{'_' + tag if tag else ''}.json"
    path.write_text(json.dumps(out, indent=2))
    for k, rows in out.items():
        print("==", k)
        for r in rows:
            print(f"  {r['past24h_wet_from']:.0%}-{r['past24h_wet_to']:.0%} share {r['share_of_rainy_steps']:.1%} "
                  + " | ".join(f"{m}: caught {r[m]['caught'] or 0:.0%} total {r[m]['total_pct'] or 0:.0f}%"
                               for m in ("classical", "ai")))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
