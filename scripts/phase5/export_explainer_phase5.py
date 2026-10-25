"""Numbers, curves and figures for explainer Part 5 (docs/explainer/part5.tex).

    python scripts/phase5/export_explainer_phase5.py
Writes docs/explainer/p5_numbers.tex, p5_q_<city>_<scenario>.dat, p5_thr_<city>.dat, fig_p5_*.png and
results/phase5/phase5.md (the same numbers as a table).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from floodrisk import config, hazard  # noqa: E402
import run_floods as RF  # noqa: E402
from export_flood_site import DEPTH_EDGES, DEPTH_RGB, HAZ_RGB  # noqa: E402

DOC = ROOT / "docs" / "explainer"
TAG = {"algiers": "Alg", "ghardaia": "Gha"}
SRC = {"truth": "Truth", "ai": "AI", "classical": "Cl", "ai_sparse": "Sparse", "imerg": "Sat"}
SCEN = {"gauge": "G", "satellite": "S"}
SENS = {"wet_soil_amc3": "Wet", "dry_soil_amc1": "Dry", "terrain_unfilled": "Unfilled", "built_n_0.02": "Smooth",
        "built_n_0.10": "Rough", "drains_20mmh": "Drains"}
ZOOM = {"algiers": (36.755, 2.995, 36.805, 3.075), "ghardaia": (32.44, 3.55, 32.53, 3.80)}


def when(s):
    return "--" if s is None else pd.Timestamp(s).strftime("%-d %b %H:%M")


def pct(x, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{100 * x:+.0f}\\%" if signed else f"{100 * x:.0f}\\%"


def num(x, d=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:,.{d}f}".replace(",", "{,}")


def lead(x):
    if x is None:
        return "no alarm"
    h, m = divmod(abs(round(x)), 60)
    t = f"{h}\\,h\\,{m:02d}" if h else f"{m}\\,min"
    return t + (" early" if x >= 0 else " late")


def macros(key, s):
    t = TAG[key]
    b = next(iter(s["basins"].values()))
    bk = next(iter(s["basins"]))
    M = {
        "BasinArea": num(b["area_km2"], 1 if b["area_km2"] < 100 else 0),
        "BasinRef": num(b["area_km2_ref"], 1) if b["area_km2_ref"] else "--",
        "CN": num(b["cn_mean"]), "Built": pct(b["built_share"], False),
        "TowerShare": pct(b["tower_box_share"], False),
        "GridY": num(s["grid"][0]), "GridX": num(s["grid"][1]), "Cells": num(s["grid"][0] * s["grid"][1] / 1e6, 2),
        "CellX": num(s["cell_m"][0]), "CellY": num(s["cell_m"][1]),
        "Pits": num(s["filled_pits_Mm3"], 1),
        "StormSat": num(s["basin_true_total_mm"]), "GaugeTotal": num(s["gauge_total_mm"]),
        "Factor": num(s["gauge_factor"], 2), "AlarmHa": num(s["alarm_ha"]),
    }
    for i, (T, th) in enumerate(s["thresholds"].items()):
        L = "ABCDE"[i]
        M[f"Dur{L}"] = T
        M[f"Thr{L}"] = num(th["threshold_mm"]) if th["threshold_mm"] else "--"
    err = 0.0
    for sc, S in s["scenarios"].items():
        for src, r in S.items():
            p = f"{SCEN[sc]}{SRC[src]}"
            q = r[bk]
            M[p + "Peak"] = num(q["peak_m3s"])
            M[p + "PeakTime"] = when(q["peak_time"])
            M[p + "Vol"] = num(q["volume_Mm3"], 2)
            M[p + "Wet"] = num(r["built_ha_wet"])
            M[p + "Sig"] = num(r["built_ha_significant"])
            M[p + "Flood"] = when(r["zone_flood_time"])
            M[p + "Alarm"] = when(r["alarm_time"])
            M[p + "Lead"] = lead(r["alarm_lead_min"])
            M[p + "ModelLead"] = lead(r["model_alarm_lead_min"])
            err = max(err, abs(r["budget"]["error_frac"]))
            if src != "truth":
                M[p + "PeakErr"] = pct(q.get("peak_err"))
                M[p + "VolErr"] = pct(q.get("volume_err"))
                M[p + "QR"] = num(q.get("q_r"), 2)
                M[p + "CSI"] = num(r.get("csi_wet_built"), 2)
                M[p + "Spear"] = num(r.get("commune_spearman"), 2)
                M[p + "Top"] = str(r.get("commune_top5_overlap", "--"))
    for k, r in s.get("sensitivity", {}).items():
        q = r[bk]
        M[f"Sens{SENS[k]}"] = num(q["peak_m3s"])
        M[f"Sens{SENS[k]}Err"] = pct(q.get("peak_err"))
        M[f"Sens{SENS[k]}Vol"] = num(q["volume_Mm3"], 2)
        M[f"Sens{SENS[k]}Sig"] = num(r["built_ha_significant"])
    M["BudgetErr"] = f"{err:.0e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"
    obs = s["observed"].get(bk, {})
    for k in ("peak_m3s", "peak_model_m3s", "volume_Mm3"):
        if k in obs:
            M[{"peak_m3s": "ObsPeak", "peak_model_m3s": "ObsModel", "volume_Mm3": "ObsVol"}[k]] = num(obs[k], 1 if k == "volume_Mm3" else 0)
    if "peak_range_m3s" in obs:
        M["ObsLo"], M["ObsHi"] = num(obs["peak_range_m3s"][0]), num(obs["peak_range_m3s"][1])
    top = s["scenarios"]["gauge"]["truth"]["communes_top"]
    for i, r in enumerate(top[:3]):
        M[f"Commune{'ABC'[i]}"] = r["commune"].replace("_", " ")
        M[f"Commune{'ABC'[i]}Ha"] = num(r["built_ha_at_risk"])
    return {f"\\newcommand{{\\Pfive{t}{k}}}{{{v}}}%" for k, v in M.items()}


def dat_files(key, s):
    d = config.PROC / key
    bk = next(iter(s["basins"]))
    for sc in ("gauge", "satellite"):
        cols = {}
        for src in RF.SOURCES:
            q = np.load(d / f"{sc}_{src}.npz", allow_pickle=True)[f"q_{bk}"]
            cols[src] = q[: q.size // 10 * 10].reshape(-1, 10).mean(1)
        n = min(len(v) for v in cols.values())
        df = pd.DataFrame({"hour": np.arange(n) / 6.0, **{k: np.round(v[:n], 3) for k, v in cols.items()}})
        df.to_csv(DOC / f"p5_q_{key}_{sc}.dat", sep=" ", index=False)
    rows = []
    for T, th in s["thresholds"].items():
        for D, ha in zip(th["depths_mm"], th["town_wet_ha"]):
            rows.append({"hours": int(T), "mm": D, "ha": round(ha, 2)})
    pd.DataFrame(rows).to_csv(DOC / f"p5_thr_{key}.dat", sep=" ", index=False)


def _extent(g):
    r = g.lon[1] - g.lon[0]
    return [g.lon[0] - r / 2, g.lon[-1] + r / 2, g.lat[-1] - r / 2, g.lat[0] + r / 2]


def fig_ground(setups):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), gridspec_kw={"width_ratios": [1, 1]})
    for row, (key, S) in enumerate(setups.items()):
        g = S["g"]
        ext = _extent(g)
        ls = LightSource(azdeg=315, altdeg=40)
        z = np.where(S["sea"], np.nan, g.z)
        shade = ls.hillshade(np.nan_to_num(z, nan=0), vert_exag=3, dx=g.dx, dy=g.dy)
        ax = axes[row, 0]
        ax.imshow(np.where(S["sea"], np.nan, shade), cmap="Greys_r", extent=ext, vmin=0, vmax=1)
        ax.imshow(z, cmap="terrain", alpha=0.45, extent=ext)
        from floodrisk import terrain as T
        zf = T.fill(g.z, S["sea"])
        acc = T.accumulate(zf, T.d8(zf, g.dx, g.dy, S["sea"])) * g.dx * g.dy / 1e6
        thr = 0.5 if key == "algiers" else 20
        ax.imshow(np.ma.masked_less(acc, thr) * 0 + 1, cmap=ListedColormap(["#2a78d6"]), extent=ext, interpolation="nearest")
        for k, b in S["basins"].items():
            ax.contour(b["mask"].astype(float), levels=[0.5], colors="k", linewidths=1.2, extent=ext, origin="upper")
            ax.plot(b["lon"], b["lat"], "o", ms=6, mfc="#eb6834", mec="k")
        tb = S["tbox"]
        ax.contour(tb.astype(float), levels=[0.5], colors="#eb6834", linewidths=1.2, linestyles="--", extent=ext, origin="upper")
        ax.set_title(f"{config.CITIES[key].name}: terrain, streams (blue), checked basin (black), tower rain (orange dashes)",
                     fontsize=8.5)
        ax = axes[row, 1]
        cn = np.where(S["sea"], np.nan, S["cn"])
        im = ax.imshow(cn, cmap="YlOrBr", vmin=55, vmax=100, extent=ext, interpolation="nearest")
        ax.contour(next(iter(S["basins"].values()))["mask"].astype(float), levels=[0.5], colors="k", linewidths=1, extent=ext, origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="curve number")
        ax.set_title(f"{config.CITIES[key].name}: how much rain runs off (curve number; 100 = all of it)", fontsize=8.5)
        for a in axes[row]:
            a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3])
            a.tick_params(labelsize=7)
            a.set_aspect(1 / np.cos(np.deg2rad(g.lat.mean())))
    fig.tight_layout()
    fig.savefig(DOC / "fig_p5_ground.png", dpi=170)
    plt.close(fig)


def fig_flood(key, S):
    g = S["g"]
    d = config.PROC / key
    ext = _extent(g)
    lat0, lon0, lat1, lon1 = ZOOM[key]
    panels = [("gauge", "truth", "depth", "true rain: deepest water"), ("gauge", "truth", "hazard", "true rain: worst danger"),
              ("gauge", "ai", "depth", "towers + AI: deepest water"), ("gauge", "imerg", "depth", "satellite only: deepest water")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6 if key == "algiers" else 5.8))
    ls = LightSource(azdeg=315, altdeg=40)
    shade = ls.hillshade(g.z, vert_exag=2, dx=g.dx, dy=g.dy)
    dcm = ListedColormap(DEPTH_RGB)
    dn = BoundaryNorm(DEPTH_EDGES + [100], dcm.N)
    hcm = ListedColormap(HAZ_RGB)
    hn = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], hcm.N)
    for ax, (sc, src, kind, title) in zip(axes.ravel(), panels):
        R = np.load(d / f"{sc}_{src}.npz", allow_pickle=True)
        ax.imshow(np.where(S["sea"], np.nan, shade), cmap="Greys_r", extent=ext, vmin=0, vmax=1)
        ax.imshow(np.where(S["sea"], 1.0, np.nan), cmap=ListedColormap(["#dfe9f2"]), extent=ext)
        hm = R["hmax"].astype(np.float32)
        if kind == "depth":
            im = ax.imshow(np.ma.masked_less(hm, DEPTH_EDGES[0]), cmap=dcm, norm=dn, extent=ext, interpolation="nearest")
        else:
            hc = hazard.hazard_class(R["hrmax"].astype(np.float32), hm).astype(float)
            im = ax.imshow(np.ma.masked_less(hc, 1), cmap=hcm, norm=hn, extent=ext, interpolation="nearest")
        ax.contour(next(iter(S["basins"].values()))["mask"].astype(float), levels=[0.5], colors="k", linewidths=0.9,
                   extent=ext, origin="upper")
        ax.contour(S["zone"].astype(float), levels=[0.5], colors="#eb6834", linewidths=0.9, linestyles=":", extent=ext, origin="upper")
        ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
        ax.set_aspect(1 / np.cos(np.deg2rad(0.5 * (lat0 + lat1))))
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        if kind == "depth":
            cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=DEPTH_EDGES)
            cb.set_label("m", fontsize=7)
        else:
            cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[1, 2, 3, 4])
            cb.ax.set_yticklabels(["low", "moderate", "significant", "extreme"], fontsize=7)
        cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(DOC / f"fig_p5_flood_{key}.png", dpi=170)
    plt.close(fig)


def md_table(summ):
    lines = ["# Phase 5: flood risk (SIMULATION)", "",
             "Flood runs on the Phase 4 storms. Peaks and volumes are for the checked basin "
             "(Koriche at Bab El Oued; M'zab at Ghardaïa). 'gauge' = rain maps scaled to the gauge storm total.", ""]
    for key, s in summ.items():
        bk = next(iter(s["basins"]))
        lines += [f"## {s['city']}", "",
                  f"- basin {s['basins'][bk]['area_km2']:.1f} km², curve number {s['basins'][bk]['cn_mean']:.0f}, "
                  f"tower rain over {100 * s['basins'][bk]['tower_box_share']:.0f}% of it",
                  f"- true storm over the basin {s['basin_true_total_mm']:.0f} mm; gauge {s['gauge_total_mm']:.0f} mm "
                  f"(factor {s['gauge_factor']:.2f})",
                  f"- observed: {s['observed'].get(bk, {}).get('src', '')}", "",
                  "| storm | rain | peak m³/s | peak time | volume Mm³ | peak vs truth | flooded built-up CSI | "
                  "communes Spearman | alarm | before the town floods |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for sc, S in s["scenarios"].items():
            for src, r in S.items():
                q = r[bk]
                lead_m = r["alarm_lead_min"]
                lines.append(f"| {sc} | {src} | {q['peak_m3s']:.0f} | {q['peak_time']} | {q['volume_Mm3']:.2f} | "
                             f"{'' if src == 'truth' else f'{100 * q['peak_err']:+.0f}%'} | "
                             f"{'' if src == 'truth' else f'{r['csi_wet_built']:.2f}'} | "
                             f"{'' if src == 'truth' else f'{r['commune_spearman']:.2f}'} | {r['alarm_time'] or '-'} | "
                             f"{'-' if lead_m is None else f'{lead_m:.0f} min'} |")
        lines += ["", "| sensitivity (truth, gauge storm) | peak m³/s | vs base |", "|---|---|---|"]
        for k, r in s.get("sensitivity", {}).items():
            lines.append(f"| {k} | {r[bk]['peak_m3s']:.0f} | {100 * r[bk]['peak_err']:+.0f}% |")
        lines += ["", "| rain threshold | mm |", "|---|---|"]
        for T, th in s["thresholds"].items():
            lines.append(f"| {T} h | {th['threshold_mm'] if th['threshold_mm'] is None else round(th['threshold_mm'])} |")
        lines.append("")
    (config.RESULTS / "phase5.md").write_text("\n".join(lines))


def main():
    summ, setups, out = {}, {}, set()
    for key in config.CITIES:
        f = config.RESULTS / f"{key}.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        summ[key] = s
        out |= macros(key, s)
        dat_files(key, s)
        S = RF.setup(config.CITIES[key])
        from floodrisk import rain
        S["tbox"] = RF._tower_box(rain.Phase4Rain(key), S["g"])
        setups[key] = S
        fig_flood(key, S)
    fig_ground(setups)
    (DOC / "p5_numbers.tex").write_text("% generated by scripts/phase5/export_explainer_phase5.py\n" + "\n".join(sorted(out)) + "\n")
    md_table(summ)
    print("wrote p5_numbers.tex,", len(out), "numbers; figures and .dat files")


if __name__ == "__main__":
    main()
