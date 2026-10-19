"""Sanity check of the link reliability score on Sweden (calibration months only).

    python scripts/phase2_reliability_sweden.py

Computes the score for all 728 sub-links on June-July 2015 and compares it with
the Phase 1 quality rule (which used the classical detector). No test data used.
Writes results/phase2_reliability_sweden.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from targ import reliability as rel
from targ.classical import total_loss
from targ.data import openmrg
from targ.evaluation import OPENMRG_CALIB

ROOT = Path(__file__).resolve().parents[1]


def main():
    p1 = json.loads((ROOT / "results" / "phase1_openmrg.json").read_text())
    ds = openmrg.load_cml_1min().sel(time=OPENMRG_CALIB).load()
    tl = total_loss(ds, 5).values
    lev = rel.levels_15min(tl)
    r = rel.score_links(ds, level_15=lev)
    avail = np.isfinite(tl).mean(axis=1)
    keep = r["keep"] & (avail >= 0.5)
    old_drop = np.isin(ds.cml_id.values, p1["qc_dropped"])
    out = {
        "n_sublinks": int(len(keep)), "kept": int(keep.sum()),
        "phase1_dropped": int(old_drop.sum()),
        "both_drop": int((~keep & old_drop).sum()),
        "only_new_drops": int((~keep & ~old_drop).sum()),
        "only_phase1_drops": int((keep & old_drop).sum()),
        "judged": int(r["judged"].sum()),
        "median_unconfirmed_share_judged": float(np.median(r["unconfirmed_share"][r["judged"]])),
        "rule": r["rule"],
    }
    (ROOT / "results" / "phase2_reliability_sweden.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    # raw-signal skill of each group: 15-min excess vs radar (calibration months)
    from targ.features import low_reference
    rad = openmrg.radar_along_cmls(openmrg.load_cml_1min()).sel(cml_id=ds.cml_id, time=OPENMRG_CALIB).load()
    r15 = rad.resample(time="15min", label="left").mean().values
    ex = lev - low_reference(lev, block=1, window=288)
    n = min(ex.shape[1], r15.shape[1])
    rs = []
    for i in range(len(ex)):
        a, b = ex[i, :n], r15[i, :n]
        ok = np.isfinite(a) & np.isfinite(b)
        rs.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 50 and a[ok].std() > 0 and b[ok].std() > 0
                  else np.nan)
    rs = np.array(rs)
    out["raw_signal_r_vs_radar_median"] = {
        name: {"n": int(m.sum()), "median_r": float(np.nanmedian(rs[m])) if m.any() else None}
        for name, m in (("kept_by_both", keep & ~old_drop), ("dropped_only_by_new_score", ~keep & ~old_drop),
                        ("dropped_by_both", ~keep & old_drop), ("dropped_only_by_phase1", keep & old_drop))}
    (ROOT / "results" / "phase2_reliability_sweden.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["raw_signal_r_vs_radar_median"], indent=1))


if __name__ == "__main__":
    main()
