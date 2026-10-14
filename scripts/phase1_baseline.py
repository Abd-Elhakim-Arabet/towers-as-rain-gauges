"""Phase 1: classical CML rain retrieval on OpenMRG, scored against radar and gauges.

    python scripts/phase1_baseline.py

1. Tune the wet/dry window + threshold on June–July (max MCC vs radar).
2. Pick the wet-antenna model on June–July (smallest |bias| vs nearby gauges, hourly).
3. Score once on August (held out).
Writes results/phase1_openmrg.json, results/phase1_openmrg.md and figures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import xarray as xr

from targ.classical import ClassicalParams, quality_mask, retrieve, rolling_std, total_loss
from targ.data import openmrg
from targ.evaluation import (OPENMRG_CALIB, OPENMRG_SPLITS, hourly, rain_scores,
                             to_radar_times, wet_dry_scores)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
GAUGE_MAX_DIST_M = 1000


def tune_wet_dry(ds, radar, base: ClassicalParams):
    tl = total_loss(ds, base.max_gap_fill_min)
    ref = radar.sel(time=slice(tl.time.values[0], tl.time.values[-1]))
    rows = []
    for window in (15, 30, 60):
        s = xr.DataArray(rolling_std(tl.values, window), coords=tl.coords, dims=tl.dims)
        for thr in np.round(np.arange(0.1, 1.55, 0.05), 2):
            wet = (s > thr).astype(float).where(s.notnull())
            est = to_radar_times(wet, ref.time.values)
            sc = wet_dry_scores(ref.values, est.values)
            rows.append({"window_min": window, "threshold_db": float(thr), **sc})
    best = max(rows, key=lambda r: r["MCC"])
    return best, rows


def gauge_series(R, gauges, pairs):
    """Hourly CML R and gauge R for each (sub-link, gauge) pair, stacked to 1-D."""
    cml_h = hourly(R.sel(cml_id=pairs.cml_id.values))
    g_h = hourly(gauges.rainfall_rate).sel(id=pairs.gauge.values)
    g_h = g_h.sel(time=cml_h.time)
    return g_h.values.ravel(), cml_h.values.ravel()


def per_link_r(a: xr.DataArray, b: xr.DataArray) -> float:
    """Median over sub-links of the Pearson r between two (cml_id, time) series."""
    rs = []
    for c in a.cml_id.values:
        x, y = a.sel(cml_id=c).values, b.sel(cml_id=c).values
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() > 10 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
            rs.append(np.corrcoef(x[ok], y[ok])[0, 1])
    return float(np.median(rs))


def radar_at_gauges(gauges):
    """Radar pixel containing each gauge, hourly — tells us how good the radar reference is."""
    import pyproj

    rad = xr.open_dataset(openmrg.RAW / "radar" / "radar.nc")
    tr = pyproj.Transformer.from_crs("EPSG:4326", rad.attrs["proj_string"], always_xy=True)
    gx, gy = tr.transform(gauges.lon.values, gauges.lat.values)
    R = openmrg.load_radar_rainrate()
    px = R.sel(x=xr.DataArray(gx, dims="id"), y=xr.DataArray(gy, dims="id"), method="nearest")
    px = px.assign_coords(id=gauges.id.values)
    # radar scans are instantaneous every 5 min; hourly mean of 12 scans
    rad_h = px.resample(time="1h", label="left", closed="left").mean()
    return rad_h


def main():
    OUT.mkdir(exist_ok=True)
    ds = openmrg.load_cml_1min().sel(time=slice("2015-06-01", "2015-08-31T23:59")).load()
    radar = openmrg.radar_along_cmls(ds).sel(time=slice("2015-06-01", "2015-08-31T23:59")).load()
    gauges = openmrg.load_city_gauges()
    pairs = openmrg.gauge_link_pairs(ds, gauges, GAUGE_MAX_DIST_M)
    print(f"{ds.sizes['cml_id']} sub-links; {len(pairs)} sub-links within "
          f"{GAUGE_MAX_DIST_M} m of {pairs.gauge.nunique()} gauges")

    base = ClassicalParams()
    calib = ds.sel(time=OPENMRG_CALIB)

    # 1. wet/dry tuning on calibration months
    best, grid = tune_wet_dry(calib, radar, base)
    print("best wet/dry on calibration:", best)
    params = replace(base, window_min=best["window_min"], threshold_db=best["threshold_db"])

    # 2. quality control (calibration months only), then wet-antenna choice
    keep = quality_mask(calib, params)
    qc_ids = keep.cml_id.values[keep.values]
    print(f"QC keeps {keep.values.sum()} / {keep.size} sub-links")
    qc_pairs = pairs[pairs.cml_id.isin(qc_ids)].reset_index(drop=True)

    waa_grid = ([{"waa": "none"}]
                + [{"waa": "schleiss", "waa_max_db": w} for w in (1.0, 1.5, 2.3)]
                + [{"waa": "pastorek", "pastorek_A_max": a} for a in (4.0, 6.0, 8.0, 10.0, 14.0)])
    waa_rows = []
    calib_qc = calib.sel(cml_id=qc_ids)
    for cfg in waa_grid:
        out = retrieve(calib_qc, replace(params, **cfg))
        ref, est = gauge_series(out.R, gauges, qc_pairs)
        sc = rain_scores(ref, est)
        waa_rows.append({**cfg, **sc})
        print(f"calib {cfg} vs gauges hourly: r={sc['r']:.3f} RMSE={sc['RMSE']:.3f} "
              f"bias={sc['bias_pct']:+.1f}%")
    # Rule (calibration data only; August never seen): lowest hourly RMSE vs gauges;
    # candidates within 2% of that RMSE are tied, and the tie goes to the smallest |bias|.
    # (RMSE alone rewards under-estimation: A_max 14 vs 10 differ by 0.4% RMSE but 38% vs 12% bias.)
    rmse_min = min(r["RMSE"] for r in waa_rows)
    tied = [r for r in waa_rows if r["RMSE"] <= 1.02 * rmse_min]
    best_waa = min(tied, key=lambda r: abs(r["bias_pct"]))
    params = replace(params, **{k: best_waa[k] for k in best_waa if k in
                                ("waa", "waa_max_db", "pastorek_A_max")})
    print("chosen params:", params)

    # 3. score on held-out August
    test_all = ds.sel(time=OPENMRG_SPLITS["test"])
    res = {
        "params": asdict(params),
        "split": {k: [v.start, v.stop] for k, v in OPENMRG_SPLITS.items()},
        "n_sublinks": int(ds.sizes["cml_id"]),
        "n_sublinks_qc": int(len(qc_ids)),
        "qc_dropped": [int(c) for c in keep.cml_id.values[~keep.values]],
        "n_gauge_pairs": int(len(pairs)),
        "n_gauge_pairs_qc": int(len(qc_pairs)),
        "calibration": {"wet_dry_best": best, "waa_choice": waa_rows},
    }
    for tag, ids, prs in (("test", qc_ids, qc_pairs), ("test_no_qc", ds.cml_id.values, pairs)):
        test = test_all.sel(cml_id=ids)
        out = retrieve(test, params)
        rad_t = radar.sel(time=OPENMRG_SPLITS["test"], cml_id=ids)
        wet5 = to_radar_times(out.wet, rad_t.time.values)
        R5 = to_radar_times(out.R, rad_t.time.values)
        ref, est = gauge_series(out.R, gauges, prs)
        res[tag] = {
            "wet_dry_vs_radar_5min": wet_dry_scores(rad_t.values, wet5.values),
            "rain_vs_radar_5min": rain_scores(rad_t.values, R5.values),
            "rain_vs_radar_hourly": rain_scores(hourly(rad_t).values, hourly(out.R).values),
            "rain_vs_gauges_hourly": rain_scores(ref, est),
            "per_link_hourly_r_vs_radar_median": per_link_r(hourly(out.R), hourly(rad_t)),
        }
    out = retrieve(test_all.sel(cml_id=qc_ids), params)
    rad_t = radar.sel(time=OPENMRG_SPLITS["test"], cml_id=qc_ids)
    pairs = qc_pairs

    # Reference quality: radar pixel vs gauge (same hours), so readers can judge the radar
    rad_g = radar_at_gauges(gauges).sel(time=OPENMRG_SPLITS["test"])
    g_h = hourly(gauges.rainfall_rate).sel(time=rad_g.time)
    res["test"]["radar_vs_gauges_hourly"] = rain_scores(g_h.values, rad_g.transpose(*g_h.dims).values)

    (OUT / "phase1_openmrg.json").write_text(json.dumps(res, indent=2, default=float))
    write_markdown(res)
    make_figures(out, rad_t, gauges, pairs, grid)
    print(json.dumps(res["test"], indent=2, default=float))


def write_markdown(res):
    p = res["params"]
    waa_desc = {"none": "none", "schleiss": f"Schleiss 2013 (max {p['waa_max_db']} dB)",
                "pastorek": f"Pastorek 2021 (A_max {p['pastorek_A_max']} dB)"}[p["waa"]]

    def row(name, t):
        wd, rr, rh, rg = (t["wet_dry_vs_radar_5min"], t["rain_vs_radar_5min"],
                          t["rain_vs_radar_hourly"], t["rain_vs_gauges_hourly"])
        return (f"| {name} | {wd['MCC']:.3f} | {wd['TPR']:.3f} / {wd['FPR']:.3f} "
                f"| {rr['r']:.2f} / {rr['RMSE']:.2f} / {rr['bias_pct']:+.0f}% "
                f"| {rh['r']:.2f} / {rh['RMSE']:.2f} / {rh['bias_pct']:+.0f}% "
                f"| {t['per_link_hourly_r_vs_radar_median']:.2f} "
                f"| {rg['r']:.2f} / {rg['RMSE']:.2f} / {rg['bias_pct']:+.0f}% |")

    qg = res["test"]["radar_vs_gauges_hourly"]
    lines = [
        "# Phase 1 — classical baseline, OpenMRG (REAL data)",
        "",
        "Test period: **August 2015, held out** (tuning used June–July only).",
        "",
        f"Pipeline: rolling-std wet/dry (window {p['window_min']} min, centred; threshold "
        f"{p['threshold_db']} dB) → constant baseline (mean of last {p['n_average_last_dry']} dry min) "
        f"→ wet-antenna {waa_desc} → ITU-R P.838-3 k-R power law.",
        "",
        f"Quality control (from June–July only): drop sub-links flagged wet > 30% of the time or with "
        f"< 50% data. Keeps {res['n_sublinks_qc']} / {res['n_sublinks']} sub-links; "
        f"{res['n_gauge_pairs_qc']} sub-links lie within 1 km of one of the 10 city gauges.",
        "",
        "| Sub-links | Wet/dry MCC | TPR / FPR | vs radar 5-min: r / RMSE [mm/h] / bias "
        "| vs radar hourly | median per-link r (hourly, radar) | vs gauges hourly |",
        "|---|---|---|---|---|---|---|",
        row(f"QC ({res['n_sublinks_qc']})", res["test"]),
        row(f"all ({res['n_sublinks']})", res["test_no_qc"]),
        "",
        f"Reference check (how good is the radar?): radar pixel vs gauge, hourly, August: "
        f"r = {qg['r']:.2f}, RMSE = {qg['RMSE']:.2f} mm/h, bias = {qg['bias_pct']:+.0f}%.",
        "",
        "Wet-antenna tuning on June–July (QC sub-links, vs gauges hourly; chosen = lowest RMSE, ties within 2% → smallest |bias|):",
        "",
        "| WAA model | r | RMSE [mm/h] | bias |",
        "|---|---|---|---|",
    ]
    for r in res["calibration"]["waa_choice"]:
        name = r["waa"] + (f" max {r['waa_max_db']}" if "waa_max_db" in r else "") + \
            (f" A_max {r['pastorek_A_max']}" if "pastorek_A_max" in r else "")
        lines.append(f"| {name} | {r['r']:.3f} | {r['RMSE']:.3f} | {r['bias_pct']:+.1f}% |")
    lines += [
        "",
        "Notes:",
        "- Metrics use pairs where reference or estimate ≥ 0.1 mm/h (poligrain convention); "
        "wet = radar path average ≥ 0.1 mm/h.",
        "- Radar = SMHI reflectivity, Z = 200 R^1.5, **not gauge-adjusted**, 2 km pixels, "
        "instantaneous 5-min scans. It is a reference, not ground truth.",
        "- Pooled Pearson r across links is sensitive to a few bad links; the median per-link r "
        "is the robust view.",
        "- The centred 60-min window looks 30 min ahead, so this baseline is not strictly real-time; "
        "Phase 2 models are causal.",
    ]
    (OUT / "phase1_openmrg.md").write_text("\n".join(lines) + "\n")


def make_figures(out, rad_t, gauges, pairs, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)

    # threshold tuning curve
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for w in sorted({r["window_min"] for r in grid}):
        rs = [r for r in grid if r["window_min"] == w]
        ax.plot([r["threshold_db"] for r in rs], [r["MCC"] for r in rs], label=f"{w} min")
    ax.set(xlabel="rolling-std threshold [dB]", ylabel="MCC (June–July)",
           title="Wet/dry tuning, OpenMRG")
    ax.legend(title="window")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase1_wetdry_tuning.png", dpi=150)

    # hexbin vs radar on test
    R5 = to_radar_times(out.R, rad_t.time.values)
    ref, est = rad_t.values.ravel(), R5.values.ravel()
    ok = ~(np.isnan(ref) | np.isnan(est)) & ((ref >= 0.1) | (est >= 0.1))
    fig, ax = plt.subplots(figsize=(4.2, 4))
    hb = ax.hexbin(ref[ok], est[ok], gridsize=60, bins="log", extent=(0, 30, 0, 30), mincnt=1)
    ax.plot([0, 30], [0, 30], "k--", lw=0.8)
    ax.set(xlabel="radar path-average [mm/h]", ylabel="CML classical [mm/h]",
           title="August 2015, 5-min")
    fig.colorbar(hb, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase1_hexbin_radar.png", dpi=150)

    # time series for the sub-link closest to a gauge, wettest test day
    best = pairs.sort_values("distance_m").iloc[0]
    g = gauges.rainfall_rate.sel(id=best.gauge).sel(time=slice("2015-08-01", "2015-08-31"))
    day = g.resample(time="1D").sum().idxmax().values
    sl = slice(day, day + np.timedelta64(1, "D"))
    o = out.sel(cml_id=best.cml_id, time=sl)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axs[0].plot(o.time, o.tl, lw=0.7, label="TL")
    axs[0].plot(o.time, o.baseline, lw=1, label="baseline")
    axs[0].fill_between(o.time.values, o.tl.min().item(), o.tl.max().item(),
                        where=o.wet.values == 1, alpha=0.15, label="wet")
    axs[0].set(ylabel="dB")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[1].plot(o.time, o.R.rolling(time=5, center=True).mean(), label="CML (5-min mean)")
    axs[1].plot(g.sel(time=sl).time, g.sel(time=sl).rolling(time=5, center=True).mean(),
                label=f"gauge {best.gauge} (5-min mean)")
    axs[1].plot(rad_t.sel(cml_id=best.cml_id, time=sl).time,
                rad_t.sel(cml_id=best.cml_id, time=sl), label="radar path-avg")
    axs[1].set(ylabel="mm/h")
    axs[1].legend(fontsize=8)
    fig.suptitle(f"sub-link {best.cml_id} ({float(o.length):.1f} km, {float(o.frequency):.1f} GHz), "
                 f"{best.distance_m:.0f} m from gauge {best.gauge}")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase1_timeseries.png", dpi=150)


if __name__ == "__main__":
    main()
