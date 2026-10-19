"""Phase 2 cross-domain test 2: the Netherlands, 15-min min/max data. REAL data, zero-shot.

    python scripts/phase2_netherlands.py

Many operators do not keep 1-min averages; they keep only the lowest and highest
received level in each 15-min period (as in the Dutch dataset). So:

1. We rebuild Sweden's data in that format (15-min min/max of the 10-s signal;
   half the time after 1-dB rounding, like Nokia equipment), and train a min/max
   version of the Phase 2 network on it (same split: train 1 Jun–15 Jul, validation
   16–31 Jul, Sweden test August).
2. A classical min/max method in the style of RAINLINK (the Dutch reference
   algorithm, Overeem et al. 2016): wet if the 15-min maximum loss rises above the
   24-h reference by a threshold; rain from both min and max via ITU-R P.838 with
   a fixed wet-antenna allowance; R = 0.33 R(max) + 0.67 R(min). Threshold and
   allowance are tuned on Sweden June–July only.
   (Real RAINLINK also uses neighbouring links for wet/dry; we do not.)
3. Both are applied unchanged to the Netherlands, 1 Jun – 22 Aug 2014
   (~2,200 sub-links, 7–58 GHz), against KNMI's gauge-adjusted hourly radar.

Writes results/phase2_netherlands.json and runs/phase2/minmax_physics.pt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from targ import features as feat
from targ import neighbours as nb
from targ import reliability as rel
from targ import train_torch as T
from targ.data import netherlands, openmrg
from targ.evaluation import OPENMRG_SPLITS, rain_scores, wet_dry_scores
from targ.models.tcn_torch import TCN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
RUNS = ROOT / "runs" / "phase2"
MODEL_KW = dict(head="physics", dropout=0.2, n_levels=6)   # receptive field 253 steps = 63 h
STEPS = 4000
ALPHA = 0.33


def hourly15(x: xr.DataArray) -> xr.DataArray:
    """15-min values labelled at the interval end → hourly mean labelled at the hour end."""
    m = x.resample(time="1h", label="right", closed="right").mean()
    n = x.notnull().resample(time="1h", label="right", closed="right").sum()
    return m.where(n == 4)


def classical_minmax(loss_max, loss_min, length, freq, pol, thr_db, waa_db):
    """RAINLINK-style min/max retrieval (no neighbour check). Returns wet, R (n_links, T)."""
    import bottleneck as bn

    ref = bn.move_median(loss_min, window=96, min_count=24, axis=-1)
    ex_max, ex_min = loss_max - ref, loss_min - ref
    wet = (ex_max > thr_db).astype(float)
    a, b = feat.itu_a_b(freq, pol)
    L = np.asarray(length, float)[:, None]

    def r(ex):
        return (np.clip(ex - waa_db, 0, None) / L / a[:, None]) ** (1 / b[:, None])

    R = (ALPHA * r(ex_max) + (1 - ALPHA) * r(ex_min)) * wet
    R[R < 0.1] = 0.0
    bad = ~(np.isfinite(ex_max) & np.isfinite(ex_min))
    wet[bad], R[bad] = np.nan, np.nan
    return wet, R


def da(x, ids, times):
    return xr.DataArray(x, dims=("cml_id", "time"), coords={"cml_id": ids, "time": times})


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    p1 = json.loads((OUT / "phase1_openmrg.json").read_text())

    # ---------------- Sweden in min/max format ----------------
    mm = openmrg.load_minmax_15min().sel(time=slice("2015-06-01T00:15", "2015-09-01T00:00")).load()
    ids = np.setdiff1d(mm.cml_id.values, p1["qc_dropped"])
    mm = mm.sel(cml_id=ids).transpose("cml_id", "time")
    times = mm.time.values
    rad5 = openmrg.radar_along_cmls(openmrg.load_cml_1min()).sel(cml_id=ids).load()
    rad15 = openmrg.radar_15min(rad5).reindex(time=times)
    L, f, pol = mm.length.values, mm.frequency.values, mm.polarization.values
    dyn = feat.minmax_features(mm.loss_max.values, mm.loss_min.values, L, f, pol)
    dyn_q = feat.minmax_features(mm.loss_max_q.values, mm.loss_min_q.values, L, f, pol)
    stat = feat.static_features(L, f, pol)
    phys = feat.physics_params(L, f, pol)
    target = rad15.values.astype(np.float32)

    def rng(sl):
        a = int(np.searchsorted(times, np.datetime64(sl.start) + np.timedelta64(15, "m")))
        b = int(np.searchsorted(times, np.datetime64(sl.stop) + np.timedelta64(1, "D"), side="right"))
        return a, b

    tr, va, te = (rng(OPENMRG_SPLITS[k]) for k in ("train", "val", "test"))

    def val_fn(model):
        P, R = T.predict(model, dyn, stat, phys, va, context=256)
        logit = torch.logit(torch.from_numpy(P).clamp(1e-6, 1 - 1e-6))
        return float(T.loss_fn(logit, torch.from_numpy(R), torch.from_numpy(target[:, va[0]:va[1]]), 0.05))

    prev = OUT / "phase2_netherlands.json"
    if (RUNS / "minmax_physics.pt").exists() and prev.exists() and not os.environ.get("RETRAIN"):
        # reuse the trained min/max model (GPU training is not bit-reproducible, so
        # retraining would move every number); set RETRAIN=1 to train again
        print("== reusing runs/phase2/minmax_physics.pt", flush=True)
        model = TCN(dyn.shape[1], stat.shape[1], **MODEL_KW).to(T.device())
        model.load_state_dict(torch.load(RUNS / "minmax_physics.pt"))
        model.eval()
        hist = json.loads(prev.read_text()).get("history", [])
    else:
        print("== training min/max model on Sweden", flush=True)
        model, hist = T.train([dyn, dyn_q], stat, phys, target, tr, val_fn=val_fn, steps=STEPS,
                              crop=768, warm=256, batch=128, model_kw=MODEL_KW, seed=0, log_every=500)
        torch.save(model.state_dict(), RUNS / "minmax_physics.pt")

    def ai_out(model, d, s, ph, t_range=None):
        P, R = T.predict(model, d, s, ph, t_range, context=256)
        return P, R

    P, R = ai_out(model, dyn, stat, phys, va)
    rv = target[:, va[0]:va[1]]
    best_thr = max(((wet_dry_scores(rv, (P > t).astype(float))["MCC"], float(t))
                    for t in np.round(np.arange(0.05, 0.81, 0.05), 2)))[1]

    # classical min/max tuning on June–July (MCC vs radar 15-min; then allowance by
    # lowest hourly RMSE vs radar, ties within 2% → smallest |bias|, like Phase 1)
    cal = slice(0, va[1])
    lmax, lmin = mm.loss_max.values, mm.loss_min.values
    best = max(((wet_dry_scores(target[:, cal], classical_minmax(lmax, lmin, L, f, pol, t, 0)[0][:, cal])["MCC"], float(t))
                for t in np.round(np.arange(0.5, 4.01, 0.25), 2)))
    thr_db = best[1]
    rows = []
    for w in (0.0, 0.5, 1.0, 1.5, 2.3):
        _, Rc = classical_minmax(lmax, lmin, L, f, pol, thr_db, w)
        sc = rain_scores(hourly15(da(target[:, cal], ids, times[cal])).values,
                         hourly15(da(Rc[:, cal], ids, times[cal])).values)
        rows.append({"waa_db": w, **sc})
    rmin = min(r["RMSE"] for r in rows)
    waa_db = min((r for r in rows if r["RMSE"] <= 1.02 * rmin), key=lambda r: abs(r["bias_pct"]))["waa_db"]
    print(f"classical min/max: threshold {thr_db} dB, wet-antenna allowance {waa_db} dB; AI threshold {best_thr}")

    res = {"model_kw": MODEL_KW, "steps": STEPS, "history": hist, "ai_threshold": best_thr,
           "classical": {"threshold_db": thr_db, "waa_db": waa_db, "alpha": ALPHA,
                         "waa_tuning": rows, "wet_dry_best_mcc_calib": best[0]}}

    def evaluate(wet, R, ref15=None, ref_h=None, ids_=None, times_=None):
        out = {}
        if ref15 is not None:
            out["wet_dry_vs_radar_15min"] = wet_dry_scores(ref15, wet)
            out["rain_vs_radar_15min"] = rain_scores(ref15, R)
        Rh = hourly15(da(R, ids_, times_))
        ref_h = ref_h.reindex(time=Rh.time.values)
        out["wet_dry_vs_radar_hourly"] = wet_dry_scores(ref_h.values, (Rh.values >= 0.1).astype(float)
                                                        + np.where(np.isnan(Rh.values), np.nan, 0))
        out["rain_vs_radar_hourly"] = rain_scores(ref_h.values, Rh.values)
        return out

    # Sweden test (August) in min/max format
    sl = slice(*te)
    P, R = ai_out(model, dyn, stat, phys, te)
    ai_w, ai_R = (P > best_thr).astype(float), np.where(P > best_thr, R, 0.0)
    cw, cR = classical_minmax(lmax, lmin, L, f, pol, thr_db, waa_db)
    ref_h = hourly15(da(target, ids, times)).sel(time=slice(times[sl][0], times[sl][-1]))
    res["sweden_test"] = {
        "ai": evaluate(ai_w, ai_R, target[:, sl], ref_h, ids, times[sl]),
        "classical": evaluate(cw[:, sl], cR[:, sl], target[:, sl], ref_h, ids, times[sl]),
    }

    # ---------------- Netherlands ----------------
    nl = netherlands.load_cml().load()
    nl = nl.sel(time=slice("2014-06-01T00:15", "2014-08-22T22:00"))
    loss_max, loss_min = -nl.pmin.values.astype(float), -nl.pmax.values.astype(float)
    Ln, fn, pn = nl.length.values, nl.frequency.values, nl.polarization.values
    cw, cR = classical_minmax(loss_max, loss_min, Ln, fn, pn, thr_db, waa_db)
    avail = np.isfinite(loss_max).mean(axis=1)
    # QC: >= 50% data and >= 10 GHz. Earlier runs also dropped sub-links the classical
    # detector calls wet > 30% of the time; that set is scored again under "old_links".
    keep = (avail >= 0.5) & (fn >= 10)
    old_rule = keep & (np.nanmean(cw, axis=1) <= 0.3)
    print(f"Netherlands: {keep.sum()} / {keep.size} sub-links pass QC "
          f"(>=50% data, >=10 GHz; old rule kept {old_rule.sum()})", flush=True)
    old_ids = nl.cml_id.values[old_rule]
    nl = nl.isel(cml_id=np.where(keep)[0])
    loss_max, loss_min, cw, cR = loss_max[keep], loss_min[keep], cw[keep], cR[keep]
    Ln, fn, pn = Ln[keep], fn[keep], pn[keep]
    dn = feat.minmax_features(loss_max, loss_min, Ln, fn, pn)
    P, R = ai_out(model, dn, feat.static_features(Ln, fn, pn), feat.physics_params(Ln, fn, pn))
    ai_w, ai_R = (P > best_thr).astype(float), np.where(P > best_thr, R, 0.0)
    bad = ~np.isfinite(loss_max)
    ai_w[bad], ai_R[bad] = np.nan, np.nan

    print("KNMI radar along links ...", flush=True)
    ref = netherlands.radar_along(nl, "2014-06-01T01:00", "2014-08-22T22:00").sel(cml_id=nl.cml_id.values)
    ids_n, t_n = nl.cml_id.values, nl.time.values

    # neighbour check, same rule for both methods
    A = nb.neighbour_matrix(nl.site_0_lat.values, nl.site_0_lon.values,
                            nl.site_1_lat.values, nl.site_1_lon.values)
    est = {"classical": (cw, cR), "ai": (ai_w, ai_R)}
    nb_info = {}
    for m in ("classical", "ai"):
        w, r = est[m]
        r2, nb_info[m] = nb.neighbour_check(r, A, step_min=15)
        removed = (np.nan_to_num(r) >= 0.1) & (r2 == 0)
        est[m + "_nb"] = (np.where(removed, 0.0, w), r2)
    old = np.isin(ids_n, old_ids)
    # link reliability from the raw signal (15-min mid-point loss) and the 5 nearest neighbours
    rl = rel.score_links(nl, level_15=(loss_max + loss_min) / 2)
    good = rl["keep"]
    rel_info = {"kept": int(good.sum()), "lonely": int(rl["lonely"].sum()), "always": int(rl["always"].sum()),
                "judged": int(rl["judged"].sum()), "overlap_with_old_rule": int((good & old).sum()),
                "rule": rl["rule"]}
    print(f"reliability score keeps {good.sum()} / {len(good)} ({(good & old).sum()} also kept by the old rule)",
          flush=True)
    res["netherlands"] = {
        "period": ["2014-06-01", "2014-08-22"], "n_sublinks_total": int(keep.size),
        "n_sublinks_qc": int(keep.sum()),
        "vendor_counts": {v: int((nl.vendor.values == v).sum()) for v in ("NEC", "NOKIA")},
        "frequency_ghz_pct": np.percentile(fn, [0, 50, 100]).round(1).tolist(),
        "length_km_pct": np.percentile(Ln, [0, 50, 100]).round(2).tolist(),
        "assumptions": ["polarisation unknown, vertical assumed",
                        "links below 10 GHz dropped (rain barely affects them)"],
        "qc_rule": ">= 50% data and >= 10 GHz",
        "n_sublinks_old_rule": int(old.sum()),
        "neighbour_check": nb_info,
        **{k: evaluate(w, r, None, ref, ids_n, t_n) for k, (w, r) in est.items()},
        "old_links": {k: evaluate(w[old], r[old], None, ref.sel(cml_id=ids_n[old]), ids_n[old], t_n)
                      for k, (w, r) in est.items()},
        "n_sublinks_reliable": int(good.sum()), "reliability": rel_info,
        "reliable_links": {k: evaluate(w[good], r[good], None, ref.sel(cml_id=ids_n[good]), ids_n[good], t_n)
                           for k, (w, r) in est.items()},
        "by_vendor": {},
    }
    for v in ("NEC", "NOKIA"):
        m = nl.vendor.values == v
        res["netherlands"]["by_vendor"][v] = {
            k: evaluate(w[m], r[m], None, ref.sel(cml_id=ids_n[m]), ids_n[m], t_n) for k, (w, r) in est.items()}

    # daily rain over the whole country network: one number per day per method
    def daily_mean(R):
        h = hourly15(da(R, ids_n, t_n))
        return h
    hz = {k: daily_mean(r) for k, (w, r) in est.items()}
    ai_h, cl_h = hz["ai_nb"], hz["classical_nb"]
    ref_h = ref.reindex(time=ai_h.time.values)
    xr.Dataset({**hz, "radar": ref_h}).assign_coords(
        length=("cml_id", Ln), frequency=("cml_id", fn), vendor=("cml_id", nl.vendor.values),
        old_rule=("cml_id", old), reliable=("cml_id", good)).to_netcdf(RUNS / "nl_hourly.nc")
    ok = np.isfinite(ai_h.values) & np.isfinite(cl_h.values) & np.isfinite(ref_h.values)
    net = {k: np.where(ok, x, np.nan) for k, x in (("ai", ai_h.values), ("classical", cl_h.values),
                                                   ("radar", ref_h.values))}
    days = ai_h.time.values.astype("datetime64[D]")
    ud = np.unique(days)
    res["netherlands"]["network_daily_mean_mm"] = {
        "day": [str(d) for d in ud],
        **{k: [float(np.nanmean(v[:, days == d]) * 24) for d in ud] for k, v in net.items()}}
    (OUT / "phase2_netherlands.json").write_text(json.dumps(res, indent=2, default=float))
    for k, ms in (("sweden_test", ("classical", "ai")),
                  ("netherlands", ("classical", "ai", "classical_nb", "ai_nb"))):
        for m in ms:
            s = res[k][m]
            print(k, m, "hourly wet MCC", round(s["wet_dry_vs_radar_hourly"]["MCC"], 3),
                  "hourly r/RMSE/bias", round(s["rain_vs_radar_hourly"]["r"], 3),
                  round(s["rain_vs_radar_hourly"]["RMSE"], 3), round(s["rain_vs_radar_hourly"]["bias_pct"], 1))


if __name__ == "__main__":
    main()
