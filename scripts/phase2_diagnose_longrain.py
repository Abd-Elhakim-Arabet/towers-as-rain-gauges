"""Why is rain after a mostly rainy day under-measured in Italy? Diagnosis only.

    python scripts/phase2_diagnose_longrain.py [model.pt] [tag]

For Italy (2021, 2022) and the same sub-links as the long-rain table (old QC set),
every rainy 15-min step (radar >= 0.1 mm/h) is grouped by how much of the previous
24 h the radar called rainy. Per group we compare:

  dry level   the causal dry level the AI uses (low_reference) against an "oracle"
              dry level: the median loss of radar-dry 15-min blocks in a centred
              2-day window (uses the future: for diagnosis only, never as input)
  excess      measured extra loss above each dry level, vs the loss the radar rain
              should cause by ITU-R P.838 (a R^b L)
  correction  the dB the AI subtracts before converting (wet antenna etc.)
  totals      AI and classical rain as % of radar, and as % of the rain gauges
              (links within 1 km of a gauge), to see whether the radar is the issue
  ablation    the AI again with its 24-h channel replaced by the 3-day channel

Writes results/phase2_diagnose_longrain[_tag].json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from targ import features as feat
from targ import train_torch as T
from targ.classical import ClassicalParams, retrieve, total_loss, wet_dry
from targ.data import openmrg, openrainer
from targ.models.tcn_torch import TCN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
BINS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.01)]
SEASONS = {"2021": [f"2021{m:02d}" for m in range(5, 10)],
           "2022": [f"2022{m:02d}" for m in range(5, 10)]}


def block15(x: np.ndarray) -> np.ndarray:
    """(n, T) 1-min -> (n, T//15) block means (NaN-aware, needs 12 of 15 minutes)."""
    n, T_ = x.shape
    nb = T_ // 15
    b = x[:, :nb * 15].reshape(n, nb, 15)
    cnt = np.isfinite(b).sum(axis=2)
    with np.errstate(all="ignore"):
        m = np.nanmean(b, axis=2)
    return np.where(cnt >= 12, m, np.nan)


def oracle_dry(tl15: np.ndarray, rad15: np.ndarray, window: int = 192) -> np.ndarray:
    """Median TL of radar-dry blocks (and dry neighbours in time) in a centred window."""
    dry = (rad15 == 0)
    dry = dry & np.roll(dry, 1, axis=1) & np.roll(dry, -1, axis=1)
    x = np.where(dry, tl15, np.nan)
    return pd.DataFrame(x.T).rolling(window, center=True, min_periods=8).median().to_numpy().T


def model_and_thr(path: Path):
    p2 = json.loads((OUT / "phase2_openmrg.json").read_text())
    sel = json.loads((OUT / "phase2_seeds.json").read_text())["selected"]
    m = TCN(len(feat.CHANNELS), 3, **p2["variants"]["physics"]["model_kw"]).to(T.device())
    m.load_state_dict(torch.load(path, map_location=T.device()))
    m.eval()
    thr_file = path.with_suffix(".thr.json")
    thr = json.loads(thr_file.read_text())["threshold"] if thr_file.exists() else sel["threshold"]
    return m, thr


def prepare(months, cp, label="end"):
    """Load one Italian season (old QC links) once: inputs, references, classical."""
    ds_all = openrainer.cml(months)
    tl_all = total_loss(ds_all, cp.max_gap_fill_min)
    avail = np.isfinite(tl_all.values).mean(axis=1)
    old_rule = (np.nanmean(wet_dry(tl_all.values, cp), axis=1) <= 0.3) & (avail >= 0.5)
    ds = ds_all.isel(cml_id=np.where(old_rule)[0])
    t0 = pd.Timestamp(ds.time.values[0])
    tl = total_loss(ds, 5).values
    c = {"ds": ds, "tl": tl,
         "dyn": feat.dynamic_features(tl, ds.length.values, ds.frequency.values, ds.polarization.values),
         "stat": feat.static_features(ds.length.values, ds.frequency.values, ds.polarization.values),
         "phys": feat.physics_params(ds.length.values, ds.frequency.values, ds.polarization.values)}
    a, b = feat.itu_a_b(ds.frequency.values, ds.polarization.values)
    c["a"], c["b"], c["L"] = a, b, ds.length.values
    low = feat.low_reference(tl)
    n15 = tl.shape[1] // 15
    # 15-min blocks: [T-15, T) labelled T ("end", as found in phase2_italy)
    t15 = t0 + pd.to_timedelta(15 * (np.arange(n15) + (1 if label == "end" else 0)), "min")
    radar = openrainer.radadj_along(ds, months).reindex(time=t15.values).values
    c["radar"] = radar
    c["blk"] = {k: block15(v) for k, v in dict(tl=tl, low=low, cl=retrieve(ds, cp).R.values,
                                                ex=tl - low).items()}
    c["dry_or"] = oracle_dry(c["blk"]["tl"], radar)
    c["wetfrac"] = pd.DataFrame((radar >= 0.1).astype(float).T).where(np.isfinite(radar.T)) \
        .rolling(96, min_periods=48).mean().shift(1).to_numpy().T
    c["A_exp"] = a[:, None] * c["L"][:, None] * np.maximum(radar, 0) ** b[:, None]
    g = openrainer.gauges(months)
    pairs = openmrg.gauge_link_pairs(ds, g, 1000)
    gauge15 = np.full_like(radar, np.nan)
    gv = g.rainfall_rate.reindex(time=t15.values)
    pos = {cid: i for i, cid in enumerate(ds.cml_id.values)}
    for row in pairs.sort_values("distance_m").itertuples():
        i = pos[row.cml_id]
        if np.isnan(gauge15[i]).all():
            gauge15[i] = gv.sel(id=row.gauge).values
    c["gauge15"] = gauge15
    return c


def predict(c, model, thr, no24h=False):
    dyn = c["dyn"]
    if no24h:
        dyn = dyn.copy()
        dyn[:, 4] = dyn[:, 0]                   # 24-h channel := 3-day channel
    P, R = T.predict(model, dyn, c["stat"], c["phys"])
    return P, R


def evaluate(c, P, R, thr, P_ab=None, R_ab=None):
    """Per-group diagnosis for one model's (or ensemble's) probabilities and rates."""
    tl, radar, blk = c["tl"], c["radar"], dict(c["blk"])
    a, b, L = c["a"], c["b"], c["L"]

    def gate(P_, R_):
        x = np.where(P_ > thr, R_, 0.0)
        x[~np.isfinite(tl)] = np.nan
        return block15(x)
    blk["ai"] = gate(P, R)
    if P_ab is not None:
        blk["ai_ab"] = gate(P_ab, R_ab)
    excess = c["dyn"][:, 0].astype(float)
    corr = np.where((R > 0.05) & (P > thr),
                    excess - a[:, None] * L[:, None] * np.maximum(R, 0) ** b[:, None], np.nan)
    blk["corr"] = block15(corr)
    dry_or, wetfrac, A_exp, gauge15 = c["dry_or"], c["wetfrac"], c["A_exp"], c["gauge15"]
    rows = []
    rainy = np.isfinite(radar) & (radar >= 0.1)
    for lo, hi in BINS + [(0.0, 1.01)]:
        m = rainy & (wetfrac >= lo) & (wetfrac < hi) & np.isfinite(blk["ai"]) & np.isfinite(blk["cl"]) \
            & np.isfinite(dry_or) & np.isfinite(blk["low"])
        mg = m & np.isfinite(gauge15) & (gauge15 >= 0.1)
        r = {"past24h_wet": [lo, min(hi, 1.0)], "n": int(m.sum()), "n_with_gauge": int(mg.sum()),
             "radar_mean_mm_h": float(radar[m].mean()),
             "dry_level_error_db": float(np.mean(blk["low"][m] - dry_or[m])),
             "dry_level_error_db_median": float(np.median(blk["low"][m] - dry_or[m])),
             "excess_over_low_db": float(np.mean(blk["ex"][m])),
             "excess_over_oracle_db": float(np.mean(blk["tl"][m] - dry_or[m])),
             "expected_from_radar_db": float(np.mean(A_exp[m])),
             "needed_correction_db": float(np.mean(blk["ex"][m] - A_exp[m])),
             "ai_correction_db": float(np.nanmean(blk["corr"][m])),
             "ai_pct_of_radar": float(100 * blk["ai"][m].sum() / radar[m].sum()),
             "classical_pct_of_radar": float(100 * blk["cl"][m].sum() / radar[m].sum()),
             # split of the AI total: rain at steps the AI calls dry, and size where it calls rain
             "ai_missed_share_of_radar": float(radar[m & (blk["ai"] < 0.1)].sum() / radar[m].sum()),
             "ai_amplitude_on_hits": float(blk["ai"][m & (blk["ai"] >= 0.1)].sum()
                                           / max(1e-9, radar[m & (blk["ai"] >= 0.1)].sum()))}
        if "ai_ab" in blk:
            r["ai_no24h_pct_of_radar"] = float(100 * blk["ai_ab"][m].sum() / radar[m].sum())
        if mg.sum() > 20:
            r.update({"radar_pct_of_gauges": float(100 * radar[mg].sum() / gauge15[mg].sum()),
                      "ai_pct_of_gauges": float(100 * blk["ai"][mg].sum() / gauge15[mg].sum()),
                      "classical_pct_of_gauges": float(100 * blk["cl"][mg].sum() / gauge15[mg].sum())})
        rows.append(r)
    # overall rain/no-rain skill (15-min, vs radar) and false rain, all steps
    ok = np.isfinite(radar) & np.isfinite(blk["ai"])
    o, e = radar[ok] >= 0.1, blk["ai"][ok] >= 0.1
    tp, tn, fp, fn = (float(x) for x in ((o & e).sum(), (~o & ~e).sum(), (~o & e).sum(), (o & ~e).sum()))
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    overall = {"mcc_15min": float(mcc), "bias_pct_all_steps": float(100 * (blk["ai"][ok].sum() / radar[ok].sum() - 1))}
    return {"n_sublinks": int(c["ds"].sizes["cml_id"]), "overall": overall, "rows": rows}


def main(path: str | None = None, tag: str = ""):
    path = Path(path) if path else ROOT / "runs" / "phase2" / "physics_selected.pt"
    model, thr = model_and_thr(path)
    cp = ClassicalParams(**json.loads((OUT / "phase1_openmrg.json").read_text())["params"])
    res = {"model": str(path.relative_to(ROOT)), "threshold": thr}
    for name, months in SEASONS.items():
        c = prepare(months, cp)
        P, R = predict(c, model, thr)
        P_ab, R_ab = predict(c, model, thr, no24h=True)
        res[name] = evaluate(c, P, R, thr, P_ab, R_ab)
        print("==", name, res[name]["n_sublinks"], "sub-links", flush=True)
        for r in res[name]["rows"]:
            print(" ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    (OUT / f"phase2_diagnose_longrain{'_' + tag if tag else ''}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
