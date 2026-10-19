"""Phase 2: causal neural network (TCN) on OpenMRG, scored exactly like Phase 1.

    python scripts/phase2_openmrg.py

Split (same as Phase 1): train 1 Jun–15 Jul, validation 16–31 Jul (checkpoint and
rain/no-rain threshold), test August (scored once). Target: radar rain along each
link path (5-min). Same QC'd sub-links as Phase 1.

Variants trained (same seed, same steps):
  physics           main model. The network decides rain/no-rain and a dB correction
                    (wet antenna, baseline drift); ITU-R P.838 converts the rest to
                    mm/h. Half of the batches see rsl/tsl rounded to 1 dB (many
                    operators log 1-dB steps; Italy does).
  free              same network, but it outputs mm/h directly (ablation)
  physics_no_quant  main model without the 1-dB rounding (ablation)
Also scores everything again with August rounded to 1 dB, for classical and AI.

Writes results/phase2_openmrg.json/.md, runs/phase2/*.pt, figures.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from targ import features as feat
from targ import train_torch as T
from targ.classical import ClassicalParams, retrieve, total_loss
from targ.data import openmrg
from targ.evaluation import OPENMRG_SPLITS, hourly, rain_scores, to_radar_times, wet_dry_scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
RUNS = ROOT / "runs" / "phase2"
STEPS = int(__import__("os").environ.get("PHASE2_STEPS", 4000))


def build_inputs(ds, quant_step=None):
    if quant_step:
        ds = ds.assign(rsl=feat.quantise(ds.rsl, quant_step), tsl=feat.quantise(ds.tsl, quant_step))
    tl = total_loss(ds, 5).values
    return feat.dynamic_features(tl, ds.length.values, ds.frequency.values, ds.polarization.values)


def minute_target(radar: xr.DataArray, times: np.ndarray) -> np.ndarray:
    """Radar values placed on the 1-min grid at scan minutes, NaN elsewhere."""
    tgt = np.full((radar.sizes["cml_id"], len(times)), np.nan, np.float32)
    pos = np.searchsorted(times, radar.time.values)
    ok = (pos < len(times)) & (times[np.minimum(pos, len(times) - 1)] == radar.time.values)
    tgt[:, pos[ok]] = radar.values[:, ok]
    return tgt


def idx(times, sl):
    return int(np.searchsorted(times, np.datetime64(sl.start))), \
        int(np.searchsorted(times, np.datetime64(sl.stop) + np.timedelta64(1, "D")))


def as_da(x, ds, a, b):
    return xr.DataArray(x, dims=("cml_id", "time"),
                        coords={"cml_id": ds.cml_id.values, "time": ds.time.values[a:b]})


def scores(wet_1min, R_1min, radar, gauges, pairs):
    wet5 = to_radar_times(wet_1min, radar.time.values)
    R5 = to_radar_times(R_1min, radar.time.values)
    ref, est = gauge_series(R_1min, gauges, pairs)
    return {
        "wet_dry_vs_radar_5min": wet_dry_scores(radar.values, wet5.values),
        "rain_vs_radar_5min": rain_scores(radar.values, R5.values),
        "rain_vs_radar_hourly": rain_scores(hourly(radar).values, hourly(R_1min).values),
        "rain_vs_gauges_hourly": rain_scores(ref, est),
    }


def gauge_series(R, gauges, pairs):
    cml_h = hourly(R.sel(cml_id=pairs.cml_id.values))
    g_h = hourly(gauges.rainfall_rate).sel(id=pairs.gauge.values).sel(time=cml_h.time)
    return g_h.values.ravel(), cml_h.values.ravel()


def best_threshold(prob, rate, radar_val, ds, a, b):
    """Rain/no-rain threshold with the best MCC on validation."""
    rows = []
    for thr in np.round(np.arange(0.05, 0.81, 0.05), 2):
        wet = as_da((prob > thr).astype(float), ds, a, b)
        sc = wet_dry_scores(radar_val.values, to_radar_times(wet, radar_val.time.values).values)
        rows.append((sc["MCC"], float(thr)))
    return max(rows)[1], rows


def main():
    OUT.mkdir(exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    p1 = json.loads((OUT / "phase1_openmrg.json").read_text())
    ds = openmrg.load_cml_1min().sel(time=slice("2015-06-01", "2015-08-31T23:59")).load()
    qc_ids = np.setdiff1d(ds.cml_id.values, p1["qc_dropped"])
    ds = ds.sel(cml_id=qc_ids)
    times = ds.time.values
    radar = openmrg.radar_along_cmls(ds).sel(cml_id=qc_ids, time=slice(times[0], times[-1])).load()
    gauges = openmrg.load_city_gauges()
    pairs = openmrg.gauge_link_pairs(ds, gauges, 1000)
    print(f"{len(qc_ids)} sub-links, {len(pairs)} near gauges")

    print("features ...", flush=True)
    dyn = build_inputs(ds)
    dyn_q = build_inputs(ds, quant_step=1.0)
    stat = feat.static_features(ds.length.values, ds.frequency.values, ds.polarization.values)
    phys = feat.physics_params(ds.length.values, ds.frequency.values, ds.polarization.values)
    target = minute_target(radar, times)

    tr, va, te = (idx(times, OPENMRG_SPLITS[k]) for k in ("train", "val", "test"))
    radar_val = radar.sel(time=OPENMRG_SPLITS["val"])
    radar_te = radar.sel(time=OPENMRG_SPLITS["test"])

    def make_val(dyn_eval, cols):
        tgt_v = target[:, va[0]:va[1]]

        def val_fn(model):
            P, R = T.predict(model, dyn_eval[:, cols], stat, phys, va)
            P_t, R_t = torch.from_numpy(P), torch.from_numpy(R)
            logit = torch.logit(P_t.clamp(1e-6, 1 - 1e-6))
            return float(T.loss_fn(logit, R_t, torch.from_numpy(tgt_v), 0.05))
        return val_fn

    ALL = [0, 1, 2, 3]
    variants = {
        "physics": dict(cols=ALL, dyn=[dyn, dyn_q], head="physics"),
        "free": dict(cols=ALL, dyn=[dyn, dyn_q], head="free"),
        "physics_no_quant": dict(cols=ALL, dyn=[dyn], head="physics"),
    }
    res = {"split": {k: [v.start, v.stop] for k, v in OPENMRG_SPLITS.items()},
           "n_sublinks": int(len(qc_ids)), "steps": STEPS, "variants": {}}
    models = {}
    for name, v in variants.items():
        print(f"== training {name}", flush=True)
        cols = v["cols"]
        model_kw = dict(head=v["head"], dropout=0.2)
        model, hist = T.train([d[:, cols] for d in v["dyn"]], stat, phys, target, tr,
                              val_fn=make_val(dyn, cols), steps=STEPS, seed=0, model_kw=model_kw)
        torch.save(model.state_dict(), RUNS / f"{name}.pt")
        models[name] = (model, cols)
        P, R = T.predict(model, dyn[:, cols], stat, phys, va)
        thr, thr_rows = best_threshold(P, R, radar_val, ds, *va)
        res["variants"][name] = {"channels": [feat.CHANNELS[c] for c in cols], "model_kw": model_kw,
                                 "n_params": int(sum(p.numel() for p in model.parameters())),
                                 "receptive_field_min": model.receptive_field,
                                 "history": hist, "threshold": thr}
        for tag, d in (("test", dyn), ("test_1dB", dyn_q)):
            P, R = T.predict(model, d[:, cols], stat, phys, te)
            wet = as_da((P > thr).astype(float), ds, *te)
            Rm = as_da(np.where(P > thr, R, 0.0), ds, *te)
            res["variants"][name][tag] = scores(wet, Rm, radar_te, gauges, pairs)
        t = res["variants"][name]["test"]
        print(name, "MCC", round(t["wet_dry_vs_radar_5min"]["MCC"], 3),
              "r5", round(t["rain_vs_radar_5min"]["r"], 3),
              "gauges", {k: round(t["rain_vs_gauges_hourly"][k], 3) for k in ("r", "RMSE", "bias_pct")},
              flush=True)

    # classical, same sub-links, original and 1-dB data
    cp = ClassicalParams(**{k: v for k, v in p1["params"].items()})
    res["classical"] = {}
    for tag, q in (("test", None), ("test_1dB", 1.0)):
        d = ds.sel(time=OPENMRG_SPLITS["test"])
        if q:
            d = d.assign(rsl=feat.quantise(d.rsl, q), tsl=feat.quantise(d.tsl, q))
        out = retrieve(d, cp)
        res["classical"][tag] = scores(out.wet, out.R, radar_te, gauges, pairs)

    # save main model predictions for figures / site
    model, cols = models["physics"]
    P, R = T.predict(model, dyn[:, cols], stat, phys, te)
    thr = res["variants"]["physics"]["threshold"]
    pred = xr.Dataset({"prob": as_da(P, ds, *te), "rate": as_da(R, ds, *te),
                       "R": as_da(np.where(P > thr, R, 0.0), ds, *te)})
    pred.to_netcdf(RUNS / "physics_test_pred.nc")
    np.save(RUNS / "stat.npy", stat)
    np.save(RUNS / "phys.npy", phys)

    (OUT / "phase2_openmrg.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v["test"] for k, v in res["variants"].items()}, indent=1, default=float)[:3000])
    print(json.dumps(res["classical"], indent=1, default=float)[:2000])


if __name__ == "__main__":
    main()
