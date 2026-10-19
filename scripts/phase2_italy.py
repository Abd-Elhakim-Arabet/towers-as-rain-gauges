"""Phase 2 cross-domain test 1: OpenRainER (Emilia-Romagna, Italy). REAL data, zero-shot.

    python scripts/phase2_italy.py

The AI model trained on Sweden (runs/phase2/physics_selected.pt, chosen by validation loss) and the classical method
with its Sweden-tuned settings are both applied unchanged: no Italian data is used
for training or tuning. Italy differs in almost everything: ~25 GHz links only,
up to 25 km long, hills and mountains, 1-dB rounding, different equipment.

Period: May–September 2021 and 2022 (warm season; winter has snow, which links
see differently). References, both converted to mm/h:
  RADadj  gauge-adjusted radar, 15-min, ~1 km, averaged along each link path
  AWS     rain gauges (15-min), for links passing within 1 km of a gauge

Quality control: keep sub-links with >= 50% data. (Earlier runs also dropped
sub-links the classical detector flags wet > 30% of the time; scores on that old
link set are kept under "old_links" for comparison.) Both methods are also scored
after the neighbour check (targ.neighbours), as "<method>_nb".
The 15-min time-label convention (start or end) is not documented; we test both
on the classical estimates and record which fits (this choice is applied to both
methods equally).

Writes results/phase2_italy.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from targ import features as feat
from targ import neighbours as nb
from targ import train_torch as T
from targ.classical import ClassicalParams, retrieve, total_loss
from targ.data import openmrg, openrainer
from targ.evaluation import rain_scores, wet_dry_scores
from targ.models.tcn_torch import TCN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEASONS = {"2021": [f"2021{m:02d}" for m in range(5, 10)],
           "2022": [f"2022{m:02d}" for m in range(5, 10)]}


def to_15(x: xr.DataArray, label: str) -> xr.DataArray:
    """1-min → 15-min mean; label 'end' = interval (T-15, T] … i.e. [T-15, T) labelled T."""
    kw = dict(label="right", closed="left") if label == "end" else dict(label="left", closed="left")
    m = x.resample(time="15min", **kw).mean()
    cov = x.notnull().resample(time="15min", **kw).mean()
    return m.where(cov >= 0.8)


def hourly_from_15(x: xr.DataArray, label: str) -> xr.DataArray:
    kw = dict(label="right", closed="right") if label == "end" else dict(label="left", closed="left")
    m = x.resample(time="1h", **kw).mean()
    n = x.notnull().resample(time="1h", **kw).sum()
    return m.where(n == 4)


def pooled_r(a, b):
    a, b = np.ravel(a), np.ravel(b)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def run_season(months, model, thr, cp):
    ds_all = openrainer.cml(months)
    radar_all = openrainer.radadj_along(ds_all, months)   # cached for every sub-link
    tl = total_loss(ds_all, cp.max_gap_fill_min)
    # QC: data availability only. (The Sweden rule also dropped sub-links the classical
    # detector calls wet > 30% of the time; with 1-dB data that removes most links, and it
    # judges links by one of the two methods being compared. We record which links it
    # would keep, to compare with earlier results on the same links.)
    from targ.classical import wet_dry
    avail = np.isfinite(tl.values).mean(axis=1)
    old_rule = (np.nanmean(wet_dry(tl.values, cp), axis=1) <= 0.3) & (avail >= 0.5)
    keep = avail >= 0.5
    ds = ds_all.isel(cml_id=np.where(keep)[0])
    old_ids = ds_all.cml_id.values[old_rule]
    print(f"{months[0]}..{months[-1]}: {keep.sum()} / {keep.size} sub-links have >= 50% data "
          f"(old rule kept {old_rule.sum()})", flush=True)

    cl = retrieve(ds, cp)
    dyn = feat.dynamic_features(total_loss(ds, 5).values, ds.length.values, ds.frequency.values,
                                ds.polarization.values)
    stat = feat.static_features(ds.length.values, ds.frequency.values, ds.polarization.values)
    phys = feat.physics_params(ds.length.values, ds.frequency.values, ds.polarization.values)
    P, R = T.predict(model, dyn, stat, phys)
    coords = {"cml_id": ds.cml_id.values, "time": ds.time.values}
    dims = ("cml_id", "time")
    miss = np.isnan(cl.tl.values)
    ai_R = np.where(P > thr, R, 0.0)
    ai_R[miss] = np.nan
    est = {"classical": (cl.wet.values, cl.R.values),
           "ai": (np.where(miss, np.nan, (P > thr).astype(float)), ai_R)}

    # neighbour check, same rule for both methods
    A = nb.neighbour_matrix(ds.site_0_lat.values, ds.site_0_lon.values,
                            ds.site_1_lat.values, ds.site_1_lon.values)
    nb_info = {}
    for m in ("classical", "ai"):
        w, r = est[m]
        r2, nb_info[m] = nb.neighbour_check(r, A, step_min=1)
        removed = (np.nan_to_num(r) >= 0.1) & (r2 == 0)
        w2 = np.where(removed, 0.0, w)
        est[m + "_nb"] = (w2, r2)
    est = {k: (xr.DataArray(w, coords=coords, dims=dims), xr.DataArray(r, coords=coords, dims=dims))
           for k, (w, r) in est.items()}

    radar = radar_all.sel(cml_id=ds.cml_id.values)
    g = openrainer.gauges(months)
    pairs = openmrg.gauge_link_pairs(ds, g, 1000)
    return ds, cl, est, nb_info, old_ids, radar, g, pairs


def score_all(est_wet, est_R, radar, g, pairs, label):
    w15, r15 = to_15(est_wet, label), to_15(est_R, label)
    rad = radar.reindex(time=r15.time.values)
    out = {
        "wet_dry_vs_radar_15min": wet_dry_scores(rad.values, w15.values),
        "rain_vs_radar_15min": rain_scores(rad.values, r15.values),
        "rain_vs_radar_hourly": rain_scores(hourly_from_15(rad, label).values,
                                            hourly_from_15(r15, label).values),
    }
    if len(pairs):
        gc = g.rainfall_rate.sel(id=pairs.gauge.values).reindex(time=r15.time.values)
        e = r15.sel(cml_id=pairs.cml_id.values)
        out["rain_vs_gauges_hourly"] = rain_scores(hourly_from_15(gc, label).values,
                                                   hourly_from_15(e, label).values)
        out["n_gauge_pairs"] = int(len(pairs))
    return out


def main():
    p1 = json.loads((OUT / "phase1_openmrg.json").read_text())
    p2 = json.loads((OUT / "phase2_openmrg.json").read_text())
    cp = ClassicalParams(**p1["params"])
    v = p2["variants"]["physics"]
    model = TCN(len(feat.CHANNELS), 3, **v["model_kw"]).to(T.device())
    sel = json.loads((OUT / "phase2_seeds.json").read_text())["selected"]
    model.load_state_dict(torch.load(ROOT / "runs" / "phase2" / "physics_selected.pt"))
    model.eval()
    thr = sel["threshold"]

    res = {"model": f"runs/phase2/physics_selected.pt (seed {sel['seed']}, trained on Sweden only)", "threshold": thr,
           "classical_params": p1["params"], "seasons": {}}
    for name, months in SEASONS.items():
        ds, cl, est, nb_info, old_ids, radar, g, pairs = run_season(months, model, thr, cp)
        # time-label convention: test both on the classical estimate only
        lab = {}
        for label in ("start", "end"):
            lab[label] = pooled_r(radar.reindex(time=to_15(cl.R, label).time.values).values,
                                  to_15(cl.R, label).values)
        label = max(lab, key=lab.get)
        print(name, "label check (r classical vs radar):", lab, "->", label, flush=True)
        old_pairs = pairs[pairs.cml_id.isin(old_ids)]
        season = {
            "months": months, "n_sublinks_qc": int(ds.sizes["cml_id"]),
            "n_links_qc": int(len({c.split("_")[0] for c in ds.cml_id.values})),
            "n_sublinks_old_rule": int(len(old_ids)),
            "qc_rule": "sub-link has >= 50% of its 1-min data",
            "frequency_ghz": [float(ds.frequency.min()), float(ds.frequency.max())],
            "length_km": [float(ds.length.min()), float(np.median(ds.length)), float(ds.length.max())],
            "label_check": lab, "label": label, "neighbour_check": nb_info,
            "old_links": {},
        }
        for m, (w, r) in est.items():
            season[m] = score_all(w, r, radar, g, pairs, label)
            season["old_links"][m] = score_all(w.sel(cml_id=old_ids), r.sel(cml_id=old_ids),
                                               radar.sel(cml_id=old_ids), g, old_pairs, label)
            s_ = season[m]
            print(name, m, "MCC", round(s_["wet_dry_vs_radar_15min"]["MCC"], 3),
                  "radar h r/RMSE/bias", round(s_["rain_vs_radar_hourly"]["r"], 2),
                  round(s_["rain_vs_radar_hourly"]["RMSE"], 2),
                  round(s_["rain_vs_radar_hourly"]["bias_pct"]), "gauges",
                  {k: round(s_.get("rain_vs_gauges_hourly", {}).get(k, np.nan), 2)
                   for k in ("r", "RMSE", "bias_pct")}, flush=True)
        res["seasons"][name] = season
        hz = {k: hourly_from_15(to_15(r, label), label) for k, (w, r) in est.items()}
        hz["radar"] = hourly_from_15(radar.reindex(time=to_15(cl.R, label).time.values), label)
        xr.Dataset(hz).assign_coords(length=("cml_id", ds.length.values),
                                     old_rule=("cml_id", np.isin(ds.cml_id.values, old_ids))).to_netcdf(
            ROOT / "runs" / "phase2" / f"italy_{name}_hourly.nc")
        if name == "2021":
            # keep one storm for the explainer: the closest link-gauge pair that saw > 20 mm/h
            save_example(ds.sel(cml_id=old_ids), xr.Dataset({"R": est["classical_nb"][1].sel(cml_id=old_ids)}),
                         est["ai_nb"][1].sel(cml_id=old_ids),
                         radar.sel(cml_id=old_ids), g, old_pairs, label)
    (OUT / "phase2_italy.json").write_text(json.dumps(res, indent=2, default=float))


def save_example(ds, cl, ai_R, radar, g, pairs, label):
    near = pairs.sort_values("distance_m")
    best = None
    for row in near.itertuples():
        gr = g.rainfall_rate.sel(id=row.gauge)
        if float(gr.max()) > 20 and row.distance_m < 500:
            best = row
            break
    if best is None:
        best = near.iloc[0]
    gr = g.rainfall_rate.sel(id=best.gauge)
    t_peak = gr.idxmax().values
    sl = slice(t_peak - np.timedelta64(6, "h"), t_peak + np.timedelta64(6, "h"))
    c = best.cml_id
    ex = {"cml_id": c, "gauge": best.gauge, "distance_m": float(best.distance_m),
          "length_km": float(ds.length.sel(cml_id=c)), "frequency_ghz": float(ds.frequency.sel(cml_id=c)),
          "label": label,
          "t15": [str(t)[:16] for t in to_15(cl.R.sel(cml_id=c, time=sl), label).time.values],
          "classical": to_15(cl.R.sel(cml_id=c, time=sl), label).values.round(2).tolist(),
          "ai": to_15(ai_R.sel(cml_id=c, time=sl), label).values.round(2).tolist()}
    t15 = to_15(cl.R.sel(cml_id=c, time=sl), label).time.values
    ex["radar"] = radar.sel(cml_id=c).reindex(time=t15).values.round(2).tolist()
    ex["gauge_mm_h"] = gr.reindex(time=t15).values.round(2).tolist()
    (OUT / "phase2_italy_example.json").write_text(json.dumps(ex, default=float))


if __name__ == "__main__":
    main()
