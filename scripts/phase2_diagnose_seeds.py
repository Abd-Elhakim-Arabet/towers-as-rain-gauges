"""Why do training runs agree on rain/no-rain but not on amounts? Diagnosis only.

    python scripts/phase2_diagnose_seeds.py [glob]   (default runs/phase2/physics_seed*.pt)

For every saved run, on validation (16-31 July) and test (August), against the radar
along each path (5-min), the rain total is split into:

  missed     radar rain at steps where the run says "dry" (p <= threshold)
  amplitude  run total / radar total on steps where both say rain
  false      run rain at steps where the radar is dry
so that total = (radar_on_hits * amplitude + false) / radar_total. The same split is
done on the rate head with the threshold fixed at 0.25 for every run, to separate
the threshold choice from the rate itself. If the validation bias predicts the test
bias run by run, a validation-based calibration can remove the spread.

Writes results/phase2_diagnose_seeds.json.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_openmrg as P2  # noqa: E402

from targ import features as feat  # noqa: E402
from targ import train_torch as T  # noqa: E402
from targ.data import openmrg  # noqa: E402
from targ.evaluation import OPENMRG_SPLITS, to_radar_times  # noqa: E402
from targ.models.tcn_torch import TCN  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def split(P, R, thr, ds, rng, radar):
    wet = P2.as_da((P > thr).astype(float), ds, *rng)
    est = P2.as_da(np.where(P > thr, R, 0.0), ds, *rng)
    w5 = to_radar_times(wet, radar.time.values).values >= 0.5
    e5 = to_radar_times(est, radar.time.values).values
    r = radar.values
    ok = np.isfinite(r) & np.isfinite(e5)
    rw = ok & (r >= 0.1)
    hit = rw & w5
    tot = r[ok].sum()
    return {"bias_pct": float(100 * (e5[ok].sum() / tot - 1)),
            "missed_share_of_radar": float(r[rw & ~w5].sum() / tot),
            "amplitude_on_hits": float(e5[hit].sum() / r[hit].sum()),
            "false_share_of_radar": float(e5[ok & (r < 0.1)].sum() / tot),
            "hits_share_of_radar": float(r[hit].sum() / tot)}


def main(pattern: str | None = None):
    p1 = json.loads((ROOT / "results" / "phase1_openmrg.json").read_text())
    kw = json.loads((ROOT / "results" / "phase2_openmrg.json").read_text())["variants"]["physics"]["model_kw"]
    seeds = json.loads((ROOT / "results" / "phase2_seeds.json").read_text())["runs"]
    thr_of = {r["seed"]: r["threshold"] for r in seeds}
    ds = openmrg.load_cml_1min().sel(time=slice("2015-06-01", "2015-08-31T23:59")).load()
    ds = ds.sel(cml_id=np.setdiff1d(ds.cml_id.values, p1["qc_dropped"]))
    times = ds.time.values
    radar = openmrg.radar_along_cmls(ds).sel(cml_id=ds.cml_id.values, time=slice(times[0], times[-1])).load()
    dyn = P2.build_inputs(ds)
    stat = feat.static_features(ds.length.values, ds.frequency.values, ds.polarization.values)
    phys = feat.physics_params(ds.length.values, ds.frequency.values, ds.polarization.values)
    va, te = (P2.idx(times, OPENMRG_SPLITS[k]) for k in ("val", "test"))
    rv, rt = radar.sel(time=OPENMRG_SPLITS["val"]), radar.sel(time=OPENMRG_SPLITS["test"])
    out = []
    for f in sorted(glob.glob(pattern or str(ROOT / "runs" / "phase2" / "physics_seed*.pt"))):
        seed = int(Path(f).stem.split("seed")[-1])
        m = TCN(dyn.shape[1], stat.shape[1], **kw).to(T.device())
        m.load_state_dict(torch.load(f))
        row = {"run": Path(f).name, "threshold": thr_of.get(seed)}
        for name, rng, rad in (("val", va, rv), ("test", te, rt)):
            P, R = T.predict(m, dyn, stat, phys, rng)
            row[name] = split(P, R, row["threshold"], ds, rng, rad)
            row[name + "_thr025"] = split(P, R, 0.25, ds, rng, rad)
        out.append(row)
        print(row["run"], {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in row.items()
                           if isinstance(v, dict)}, flush=True)
    vb = np.array([r["val"]["bias_pct"] for r in out])
    tb = np.array([r["test"]["bias_pct"] for r in out])
    res = {"runs": out, "val_vs_test_bias_r": float(np.corrcoef(vb, tb)[0, 1])}
    (ROOT / "results" / "phase2_diagnose_seeds.json").write_text(json.dumps(res, indent=2))
    print("val vs test bias r:", res["val_vs_test_bias_r"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
