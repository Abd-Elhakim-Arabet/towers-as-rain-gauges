"""Phase 2 follow-up: does more varied training data (and a better rate loss) fix
(1) the under-measured rain after rainy days in Italy and (2) the spread of rain
totals between training runs?

    python scripts/phase2_mixed.py CONFIG [CONFIG ...]

CONFIG = <data>_<loss>:
  data  se       Sweden only (1 Jun-15 Jul; as before)
        mix21    Sweden + Italy 2021 (May-Aug train, Sep validation); Italy scored on 2022
        mix22    Sweden + Italy 2022 (May-Aug train, Sep validation); Italy scored on 2021
        <data>nb same, plus two inputs from the neighbours (targ.features.neighbour_features)
        mix21rel Italian training links limited to those kept by the reliability score
  loss  mse      rate error in mm/h (as before)
        log      rate error of log(1 + R)
se_mse reuses the five existing runs (runs/phase2/physics_seed*.pt).

Every config trains 5 runs (seeds 0-4), half of the batches from each country.
Nothing scored is ever trained on: Sweden is scored on August, Italy on the other
season. The same Italian links appear in both seasons (it is one network), so the
Italian test is a test on new weather, not new links; zero-shot results on new
countries stay in phase2_italy/netherlands.

Scored per run, for the 5-run average (mean probability and rate), and for each
run after a validation calibration (its rain scaled so validation totals match):
  Sweden  gauges hourly r / RMSE / bias, radar 5-min MCC
  Italy   long-rain table (phase2_diagnose_longrain: same links, groups, numbers)

Writes results/phase2_mixed_<config>.json, models in runs/phase2/mixed/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_diagnose_longrain as DL  # noqa: E402
import phase2_openmrg as P2  # noqa: E402

from targ import features as feat  # noqa: E402
from targ import neighbours as nbr  # noqa: E402
from targ import train_torch as T  # noqa: E402
from targ.classical import ClassicalParams, total_loss  # noqa: E402
from targ.data import openmrg, openrainer  # noqa: E402
from targ.evaluation import OPENMRG_SPLITS, to_radar_times, wet_dry_scores  # noqa: E402
from targ.models.tcn_torch import TCN  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "phase2" / "mixed"
OUT = ROOT / "results"
SEEDS = range(5)
STEPS = int(__import__("os").environ.get("PHASE2_STEPS", 4000))
THRESHOLDS = np.round(np.arange(0.05, 0.81, 0.05), 2)


# ------------------------------------------------------------------ data
def sweden():
    p1 = json.loads((OUT / "phase1_openmrg.json").read_text())
    ds = openmrg.load_cml_1min().sel(time=slice("2015-06-01", "2015-08-31T23:59")).load()
    ds = ds.sel(cml_id=np.setdiff1d(ds.cml_id.values, p1["qc_dropped"]))
    times = ds.time.values
    radar = openmrg.radar_along_cmls(ds).sel(cml_id=ds.cml_id.values, time=slice(times[0], times[-1])).load()
    d = {"ds": ds, "radar": radar, "dyn": P2.build_inputs(ds), "dyn_q": P2.build_inputs(ds, quant_step=1.0),
         "stat": feat.static_features(ds.length.values, ds.frequency.values, ds.polarization.values),
         "phys": feat.physics_params(ds.length.values, ds.frequency.values, ds.polarization.values),
         "target": P2.minute_target(radar, times),
         "gauges": openmrg.load_city_gauges()}
    d["pairs"] = openmrg.gauge_link_pairs(ds, d["gauges"], 1000)
    d["tr"], d["va"], d["te"] = (P2.idx(times, OPENMRG_SPLITS[k]) for k in ("train", "val", "test"))
    d["radar_val"], d["radar_te"] = radar.sel(time=OPENMRG_SPLITS["val"]), radar.sel(time=OPENMRG_SPLITS["test"])
    return d


def italy_train(year: str, reliable_only: bool = False):
    """One Italian season for training: sub-links with >= 50% data; radar 15-min mean
    rate placed on its 15 minutes ([T-15, T) for the label T, as found in phase2_italy)."""
    months = [f"{year}{m:02d}" for m in range(5, 10)]
    ds = openrainer.cml(months)
    tl = total_loss(ds, 5).values
    keep = np.isfinite(tl).mean(axis=1) >= 0.5
    ds, tl = ds.isel(cml_id=np.where(keep)[0]), tl[keep]
    if reliable_only:
        from targ import reliability as rel
        k = rel.score_links(ds, tl_1min=tl)["keep"]
        ds, tl = ds.isel(cml_id=np.where(k)[0]), tl[k]
        print(f"reliability score keeps {int(k.sum())} training sub-links", flush=True)
    times = ds.time.values
    radar = openrainer.radadj_along(ds, months)
    target = np.full(tl.shape, np.nan, np.float32)
    pos = np.searchsorted(times, radar.time.values)
    rv = radar.values.astype(np.float32)
    for k in range(1, 16):
        p = pos - k
        ok = (p >= 0) & (p < len(times))
        target[:, p[ok]] = rv[:, ok]
    target[~np.isfinite(tl)] = np.nan
    d = {"ds": ds, "target": target, "radar": radar,
         "dyn": feat.dynamic_features(tl, ds.length.values, ds.frequency.values, ds.polarization.values),
         "stat": feat.static_features(ds.length.values, ds.frequency.values, ds.polarization.values),
         "phys": feat.physics_params(ds.length.values, ds.frequency.values, ds.polarization.values)}
    sep = int(np.searchsorted(times, np.datetime64(f"{year}-09-01")))
    d["tr"], d["va"] = (0, sep), (sep, len(times))
    return d


def with_neighbours(d: dict, keys=("dyn", "dyn_q")) -> dict:
    """Copy of a data dict whose input arrays get the two neighbour channels appended."""
    ds = d["ds"]
    A = nbr.neighbour_matrix(ds.site_0_lat.values, ds.site_0_lon.values, ds.site_1_lat.values, ds.site_1_lon.values)
    out = dict(d)
    for k in keys:
        if k in d:
            out[k] = np.concatenate([d[k], feat.neighbour_features(d[k], A)], axis=1)
    return out


# ------------------------------------------------------------------ helpers
def mcc_15(P, R, thr, target):
    """Rain/no-rain MCC on 15-min blocks vs the (block-constant) minute target."""
    est = DL.block15(np.where(P > thr, R, 0.0))
    ref = DL.block15(target)
    return wet_dry_scores(ref, (est >= 0.1).astype(float))["MCC"]


def val_predictions(model, se, it):
    out = {"se": T.predict(model, se["dyn"], se["stat"], se["phys"], se["va"])}
    if it is not None:
        out["it"] = T.predict(model, it["dyn"], it["stat"], it["phys"], it["va"])
    return out


def choose_threshold(vp, se, it):
    best = None
    for thr in THRESHOLDS:
        P, R = vp["se"]
        wet = P2.as_da((P > thr).astype(float), se["ds"], *se["va"])
        m = [wet_dry_scores(se["radar_val"].values, to_radar_times(wet, se["radar_val"].time.values).values)["MCC"]]
        if "it" in vp:
            m.append(mcc_15(*vp["it"], thr, it["target"][:, it["va"][0]:it["va"][1]]))
        if best is None or np.mean(m) > best[0]:
            best = (float(np.mean(m)), float(thr))
    return best[1]


def val_scale(vp, thr, se, it):
    """Factor that makes validation rain totals match the reference (on steps both call rainy)."""
    num = den = 0.0
    P, R = vp["se"]
    est = P2.as_da(np.where(P > thr, R, 0.0), se["ds"], *se["va"])
    e5 = to_radar_times(est, se["radar_val"].time.values).values
    r5 = se["radar_val"].values
    ok = np.isfinite(e5) & np.isfinite(r5)
    num += r5[ok].sum()
    den += e5[ok].sum()
    if "it" in vp:
        P, R = vp["it"]
        e = DL.block15(np.where(P > thr, R, 0.0))
        r = DL.block15(it["target"][:, it["va"][0]:it["va"][1]])
        ok = np.isfinite(e) & np.isfinite(r)
        num += r[ok].sum()
        den += e[ok].sum()
    return float(num / den)


def sweden_scores(P, R, thr, se):
    a, b = se["te"]
    wet = P2.as_da((P > thr).astype(float), se["ds"], a, b)
    Rm = P2.as_da(np.where(P > thr, R, 0.0), se["ds"], a, b)
    s = P2.scores(wet, Rm, se["radar_te"], se["gauges"], se["pairs"])
    return {"MCC_5min": s["wet_dry_vs_radar_5min"]["MCC"],
            "radar_hourly_bias_pct": s["rain_vs_radar_hourly"]["bias_pct"],
            **{f"gauges_{k}": s["rain_vs_gauges_hourly"][k] for k in ("r", "RMSE", "bias_pct")}}


def italy_scores(c, P, R, thr):
    e = DL.evaluate(c, P, R, thr)
    rows = {f"{r['past24h_wet'][0]:.2f}-{r['past24h_wet'][1]:.2f}": r for r in e["rows"]}
    out = {**e["overall"]}
    for k, r in rows.items():
        out[f"pct_of_radar_{k}"] = r["ai_pct_of_radar"]
        out[f"pct_of_gauges_{k}"] = r.get("ai_pct_of_gauges")
        out[f"correction_db_{k}"] = r["ai_correction_db"]
        out[f"needed_db_{k}"] = r["needed_correction_db"]
        out[f"missed_share_{k}"] = r["ai_missed_share_of_radar"]
        out[f"classical_pct_of_radar_{k}"] = r["classical_pct_of_radar"]
    return out


def spread(rows, key):
    v = np.array([r[key] for r in rows if r.get(key) is not None], float)
    if v.size == 0:
        return None
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "range": float(v.max() - v.min())}


# ------------------------------------------------------------------ main
def run(config: str, se, cache):
    data, loss = config.split("_")
    use_nb = data.endswith("nb")
    data = data[:-2] if use_nb else data
    kw = json.loads((OUT / "phase2_openmrg.json").read_text())["variants"]["physics"]["model_kw"]
    it = None
    rel_only = data.endswith("rel")
    data = data[:-3] if rel_only else data
    if data.startswith("mix"):
        year = "20" + data[3:]
        if ("train", year, rel_only) not in cache:
            cache[("train", year, rel_only)] = italy_train(year, rel_only)
        it = cache[("train", year, rel_only)]
    if use_nb:
        if "se_nb" not in cache:
            cache["se_nb"] = with_neighbours(se)
        se = cache["se_nb"]
        if it is not None:
            if ("train_nb", year) not in cache:
                cache[("train_nb", year)] = with_neighbours(it)
            it = cache[("train_nb", year)]
    test_years = {"se": ["2021", "2022"], "mix21": ["2022"], "mix22": ["2021"]}[data]
    RUNS.mkdir(parents=True, exist_ok=True)

    models = []
    for seed in SEEDS:
        f = (ROOT / "runs" / "phase2" / f"physics_seed{seed}.pt") if config == "se_mse" \
            else RUNS / f"{config}_seed{seed}.pt"
        m = TCN(se["dyn"].shape[1], se["stat"].shape[1], **kw).to(T.device())
        if f.exists():
            m.load_state_dict(torch.load(f))
            m.eval()
        else:
            tv_se = torch.from_numpy(se["target"][:, se["va"][0]:se["va"][1]])
            tv_it = None if it is None else torch.from_numpy(it["target"][:, it["va"][0]:it["va"][1]])

            def val_fn(model):
                tot = 0.0
                for (P, R), tv in zip(val_predictions(model, se, it).values(), (tv_se, tv_it)):
                    logit = torch.logit(torch.from_numpy(P).clamp(1e-6, 1 - 1e-6))
                    tot += float(T.loss_fn(logit, torch.from_numpy(R), tv, 0.05, loss))
                return tot
            domains = [{"dyn": [se["dyn"], se["dyn_q"]], "stat": se["stat"], "phys": se["phys"],
                        "target": se["target"], "t_range": se["tr"], "weight": 1.0}]
            if it is not None:
                domains.append({"dyn": [it["dyn"]], "stat": it["stat"], "phys": it["phys"],
                                "target": it["target"], "t_range": it["tr"], "weight": 1.0})
            print(f"== {config} seed {seed}", flush=True)
            m, hist = T.train_multi(domains, val_fn=val_fn, steps=STEPS, seed=seed, model_kw=kw,
                                    log_every=500, rate_loss=loss)
            torch.save(m.state_dict(), f)
        vp = val_predictions(m, se, it)
        thr = choose_threshold(vp, se, it)
        models.append({"seed": seed, "model": m, "thr": thr, "scale": val_scale(vp, thr, se, it), "vp": vp})
        print(config, "seed", seed, "thr", thr, "val scale", round(models[-1]["scale"], 3), flush=True)

    res = {"config": config, "steps": STEPS, "italy_test_seasons": test_years, "runs": [], "ensemble": {},
           "notes": "ensemble = mean probability and mean rate of the 5 runs; calibrated = each run's rate "
                    "times its validation scale"}
    # ensemble threshold chosen on validation like a single run
    ens_vp = {k: (np.mean([mm["vp"][k][0] for mm in models], axis=0), np.mean([mm["vp"][k][1] for mm in models], axis=0))
              for k in models[0]["vp"]}
    ens_thr = choose_threshold(ens_vp, se, it)
    ens_scale = val_scale(ens_vp, ens_thr, se, it)
    for mm in models:
        P, R = T.predict(mm["model"], se["dyn"], se["stat"], se["phys"], se["te"])
        mm["se_test"] = (P, R)
        row = {"seed": mm["seed"], "threshold": mm["thr"], "val_scale": mm["scale"],
               "sweden": sweden_scores(P, R, mm["thr"], se),
               "sweden_calibrated": sweden_scores(P, R * mm["scale"], mm["thr"], se)}
        res["runs"].append(row)
    Pe = np.mean([mm["se_test"][0] for mm in models], axis=0)
    Re = np.mean([mm["se_test"][1] for mm in models], axis=0)
    res["ensemble"] = {"threshold": ens_thr, "val_scale": ens_scale, "sweden": sweden_scores(Pe, Re, ens_thr, se),
                       "sweden_calibrated": sweden_scores(Pe, Re * ens_scale, ens_thr, se)}
    cp = ClassicalParams(**json.loads((OUT / "phase1_openmrg.json").read_text())["params"])
    for y in test_years:
        if ("test", y) not in cache:
            cache[("test", y)] = DL.prepare([f"{y}{m:02d}" for m in range(5, 10)], cp)
        c = cache[("test", y)]
        if use_nb:
            if ("test_nb", y) not in cache:
                cache[("test_nb", y)] = with_neighbours(c, keys=("dyn",))
            c = cache[("test_nb", y)]
        preds = [DL.predict(c, mm["model"], mm["thr"]) for mm in models]
        for row, mm, (P, R) in zip(res["runs"], models, preds):
            row[f"italy_{y}"] = italy_scores(c, P, R, mm["thr"])
            row[f"italy_{y}_calibrated"] = italy_scores(c, P, R * mm["scale"], mm["thr"])
        Pe, Re = np.mean([p[0] for p in preds], axis=0), np.mean([p[1] for p in preds], axis=0)
        res["ensemble"][f"italy_{y}"] = italy_scores(c, Pe, Re, ens_thr)
        res["ensemble"][f"italy_{y}_calibrated"] = italy_scores(c, Pe, Re * ens_scale, ens_thr)
    keys = {"sweden": ["MCC_5min", "gauges_r", "gauges_RMSE", "gauges_bias_pct", "radar_hourly_bias_pct"]}
    res["spread"] = {}
    for block in ["sweden", "sweden_calibrated"] + [f"italy_{y}{s}" for y in test_years for s in ("", "_calibrated")]:
        ks = keys["sweden"] if block.startswith("sweden") else \
            ["mcc_15min", "bias_pct_all_steps", "pct_of_radar_0.00-0.10", "pct_of_radar_0.50-1.00",
             "pct_of_gauges_0.50-1.00", "correction_db_0.50-1.00"]
        res["spread"][block] = {k: spread([r[block] for r in res["runs"]], k) for k in ks}
    (OUT / f"phase2_mixed_{config}.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({"spread": res["spread"], "ensemble": res["ensemble"]}, indent=1, default=float)[:6000],
          flush=True)


def main(configs):
    se = sweden()
    cache = {}
    for cfg in configs:
        run(cfg, se, cache)


if __name__ == "__main__":
    main(sys.argv[1:] or ["se_mse"])
