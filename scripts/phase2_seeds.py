"""How much do the Phase 2 Sweden results move between training runs?

    python scripts/phase2_seeds.py

Retrains the main (physics) model with 5 seeds and scores each on August exactly
like phase2_openmrg.py. The run with the lowest VALIDATION loss (16-31 July; no test
data involved) becomes the model used everywhere else:
runs/phase2/physics_selected.pt (+ its threshold in results/phase2_seeds.json).
Writes results/phase2_seeds.json.
"""

from __future__ import annotations

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
from targ.evaluation import OPENMRG_SPLITS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    p1 = json.loads((ROOT / "results" / "phase1_openmrg.json").read_text())
    p2 = json.loads((ROOT / "results" / "phase2_openmrg.json").read_text())
    kw = p2["variants"]["physics"]["model_kw"]
    ds = openmrg.load_cml_1min().sel(time=slice("2015-06-01", "2015-08-31T23:59")).load()
    ids = np.setdiff1d(ds.cml_id.values, p1["qc_dropped"])
    ds = ds.sel(cml_id=ids)
    times = ds.time.values
    radar = openmrg.radar_along_cmls(ds).sel(cml_id=ids, time=slice(times[0], times[-1])).load()
    gauges = openmrg.load_city_gauges()
    pairs = openmrg.gauge_link_pairs(ds, gauges, 1000)
    dyn, dyn_q = P2.build_inputs(ds), P2.build_inputs(ds, quant_step=1.0)
    stat = feat.static_features(ds.length.values, ds.frequency.values, ds.polarization.values)
    phys = feat.physics_params(ds.length.values, ds.frequency.values, ds.polarization.values)
    target = P2.minute_target(radar, times)
    tr, va, te = (P2.idx(times, OPENMRG_SPLITS[k]) for k in ("train", "val", "test"))
    radar_val, radar_te = radar.sel(time=OPENMRG_SPLITS["val"]), radar.sel(time=OPENMRG_SPLITS["test"])
    tgt_v = torch.from_numpy(target[:, va[0]:va[1]])

    def val_fn(model):
        P, R = T.predict(model, dyn, stat, phys, va)
        logit = torch.logit(torch.from_numpy(P).clamp(1e-6, 1 - 1e-6))
        return float(T.loss_fn(logit, torch.from_numpy(R), tgt_v, 0.05))

    runs = []
    for seed in range(5):
        model, hist = T.train([dyn, dyn_q], stat, phys, target, tr, val_fn=val_fn, steps=P2.STEPS,
                              seed=seed, model_kw=kw, log_every=500)
        torch.save(model.state_dict(), ROOT / "runs" / "phase2" / f"physics_seed{seed}.pt")
        P, R = T.predict(model, dyn, stat, phys, va)
        thr, _ = P2.best_threshold(P, R, radar_val, ds, *va)
        out = {"seed": seed, "threshold": thr, "val_loss": float(min(h["val"] for h in hist))}
        for tag, d in (("test", dyn), ("test_1dB", dyn_q)):
            P, R = T.predict(model, d, stat, phys, te)
            out[tag] = P2.scores(P2.as_da((P > thr).astype(float), ds, *te),
                                 P2.as_da(np.where(P > thr, R, 0.0), ds, *te), radar_te, gauges, pairs)
        runs.append(out)
        t = out["test"]
        print(seed, "MCC", round(t["wet_dry_vs_radar_5min"]["MCC"], 3), "gauges r/RMSE/bias",
              round(t["rain_vs_gauges_hourly"]["r"], 3), round(t["rain_vs_gauges_hourly"]["RMSE"], 3),
              round(t["rain_vs_gauges_hourly"]["bias_pct"], 1), flush=True)

    def stats(tag, block, key):
        v = np.array([r[tag][block][key] for r in runs])
        return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}

    summary = {tag: {f"{block}.{key}": stats(tag, block, key)
                     for block, key in [("wet_dry_vs_radar_5min", "MCC"), ("rain_vs_radar_5min", "r"),
                                        ("rain_vs_radar_hourly", "r"), ("rain_vs_radar_hourly", "bias_pct"),
                                        ("rain_vs_gauges_hourly", "r"), ("rain_vs_gauges_hourly", "RMSE"),
                                        ("rain_vs_gauges_hourly", "bias_pct")]}
               for tag in ("test", "test_1dB")}
    sel = min(runs, key=lambda r: r["val_loss"])
    import shutil
    shutil.copy(ROOT / "runs" / "phase2" / f"physics_seed{sel['seed']}.pt",
                ROOT / "runs" / "phase2" / "physics_selected.pt")
    print("selected by validation loss: seed", sel["seed"], flush=True)
    from targ.models.tcn_torch import TCN
    m = TCN(dyn.shape[1], stat.shape[1], **kw).to(T.device())
    m.load_state_dict(torch.load(ROOT / "runs" / "phase2" / "physics_selected.pt"))
    P, R = T.predict(m, dyn, stat, phys, te)
    import xarray as xr
    xr.Dataset({"prob": P2.as_da(P, ds, *te), "rate": P2.as_da(R, ds, *te),
                "R": P2.as_da(np.where(P > sel["threshold"], R, 0.0), ds, *te)}).to_netcdf(
        ROOT / "runs" / "phase2" / "selected_test_pred.nc")
    selected = {"seed": sel["seed"], "threshold": sel["threshold"], "val_loss": sel["val_loss"],
                "rule": "lowest validation loss (16-31 July 2015) of 5 runs", "test": sel["test"],
                "test_1dB": sel["test_1dB"]}
    (ROOT / "results" / "phase2_seeds.json").write_text(json.dumps({"runs": runs, "summary": summary,
                                                                   "selected": selected},
                                                                  indent=2, default=float))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
