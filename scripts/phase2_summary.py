"""Collect all Phase 2 numbers into results/phase2_summary.json and results/phase2.md.

    python scripts/phase2_summary.py

Adds one robust view the per-run scripts do not compute: the median over links of
each link's own hourly correlation with the radar (Phase 1 reported the same for
Sweden), and a count of impossible hours (> 100 mm/h estimated for a whole hour).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from targ.data import openmrg
from targ.evaluation import OPENMRG_SPLITS, hourly

ROOT = Path(__file__).resolve().parents[1]
RES, RUNS = ROOT / "results", ROOT / "runs" / "phase2"


def per_link_median_r(est: xr.DataArray, ref: xr.DataArray) -> float:
    rs = []
    for c in est.cml_id.values:
        x, y = est.sel(cml_id=c).values, ref.sel(cml_id=c).values
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 20 and x[ok].std() > 0 and y[ok].std() > 0:
            rs.append(np.corrcoef(x[ok], y[ok])[0, 1])
    return float(np.median(rs))


def robust(h: xr.Dataset, m: str, subset=None) -> dict:
    if subset is not None:
        h = h.sel(cml_id=subset)
    v = h[m].values
    return {"median_per_link_hourly_r_vs_radar": per_link_median_r(h[m], h.radar),
            "hours_over_100mm_h": int(np.nansum(v > 100)),
            "hours_valid": int(np.isfinite(v).sum())}


METHODS = ("classical", "ai", "classical_nb", "ai_nb")


def robust_all(path) -> dict:
    h = xr.open_dataset(path).load()
    out = {m: robust(h, m) for m in METHODS}
    old = h.cml_id.values[h.old_rule.values]
    out["old_links"] = {m: robust(h, m, old) for m in METHODS}
    if "reliable" in h.coords:
        good = h.cml_id.values[h.reliable.values]
        out["reliable_links"] = {m: robust(h, m, good) for m in METHODS}
    return out


def main():
    p1 = json.loads((RES / "phase1_openmrg.json").read_text())
    p2 = json.loads((RES / "phase2_openmrg.json").read_text())
    seeds = json.loads((RES / "phase2_seeds.json").read_text())
    swnb = json.loads((RES / "phase2_sweden_neighbours.json").read_text())
    it = json.loads((RES / "phase2_italy.json").read_text())
    nl = json.loads((RES / "phase2_netherlands.json").read_text())
    ms = json.loads((RES / "phase2_mindspore.json").read_text())

    # Sweden robust view for the AI (Phase 1 already has it for classical)
    pred = xr.open_dataset(RUNS / "selected_test_pred.nc")
    rad = openmrg.radar_along_cmls(openmrg.load_cml_1min()).sel(
        cml_id=pred.cml_id, time=OPENMRG_SPLITS["test"]).load()
    sweden_ai_median_r = per_link_median_r(hourly(pred.R.load()), hourly(rad))

    out = {
        "sweden": {
            "classical": p2["classical"], "ai": seeds["selected"],
            "ai_first_run": p2["variants"]["physics"],
            "ablations": {k: p2["variants"][k] for k in ("free", "physics_no_quant")},
            "seeds": seeds["summary"],
            "neighbour_check": swnb,
            "median_per_link_hourly_r_vs_radar": {
                "classical": p1["test"]["per_link_hourly_r_vs_radar_median"], "ai": sweden_ai_median_r},
        },
        "italy": {name: {**s, "robust": robust_all(RUNS / f"italy_{name}_hourly.nc")}
                  for name, s in it["seasons"].items()},
        "netherlands": {**nl["netherlands"], "robust": robust_all(RUNS / "nl_hourly.nc"),
                        "sweden_minmax_test": nl["sweden_test"], "classical_params": nl["classical"]},
        "mindspore": ms,
        "long_rain": {"after": json.loads((RES / "phase2_longrain.json").read_text()),
                      "before": json.loads((RES / "phase2_longrain_before.json").read_text())
                      if (RES / "phase2_longrain_before.json").exists() else None},
    }
    (RES / "phase2_summary.json").write_text(json.dumps(out, indent=2, default=float))

    S = out["sweden"]
    lines = ["# Phase 2 — AI model (causal TCN, physics head) — REAL data", "",
             "## Sweden (OpenMRG), August 2015 held out", "",
             "| | MCC | TPR / FPR | radar 5-min r / bias | radar hourly r / bias | gauges hourly r / RMSE / bias | median per-link r |",
             "|---|---|---|---|---|---|---|"]

    def srow(n, t, med):
        w, a, h, g = (t["wet_dry_vs_radar_5min"], t["rain_vs_radar_5min"], t["rain_vs_radar_hourly"],
                      t["rain_vs_gauges_hourly"])
        return (f"| {n} | {w['MCC']:.3f} | {w['TPR']:.2f} / {w['FPR']:.3f} | {a['r']:.2f} / {a['bias_pct']:+.0f}% "
                f"| {h['r']:.2f} / {h['bias_pct']:+.0f}% | {g['r']:.2f} / {g['RMSE']:.2f} / {g['bias_pct']:+.0f}% | {med} |")
    m = S["median_per_link_hourly_r_vs_radar"]
    lines += [srow("classical (Phase 1)", S["classical"]["test"], f"{m['classical']:.2f}"),
              srow(f"AI physics (selected by validation: seed {S['ai']['seed']})", S["ai"]["test"], f"{m['ai']:.2f}"),
              srow("AI free head (ablation)", S["ablations"]["free"]["test"], "–"),
              srow("AI no 1-dB augmentation (ablation)", S["ablations"]["physics_no_quant"]["test"], "–"),
              srow("classical, data rounded to 1 dB", S["classical"]["test_1dB"], "–"),
              srow("AI physics, data rounded to 1 dB", S["ai"]["test_1dB"], "–"),
              srow("classical + neighbour check", swnb["classical_nb"],
                   f"{swnb['classical_nb']['robust']['median_per_link_hourly_r_vs_radar']:.2f}"),
              srow("AI + neighbour check", swnb["ai_nb"],
                   f"{swnb['ai_nb']['robust']['median_per_link_hourly_r_vs_radar']:.2f}"), "",
              f"Neighbour check removed {swnb['neighbour_check']['classical']['removed_share']:.1%} (classical) and "
              f"{swnb['neighbour_check']['ai']['removed_share']:.1%} (AI) of rainy minutes in Sweden.", "",
              "Ablation rows come from single runs (phase2_openmrg.py). 5 retrainings of the main design (seeds 0–4):", ""]
    for k, v in S["seeds"]["test"].items():
        lines.append(f"- {k}: mean {v['mean']:.3f}, range {v['min']:.3f} – {v['max']:.3f}")
    lines += ["", "## Italy (OpenRainER), May–Sep 2021 and 2022, zero-shot", "",
              "QC: sub-links with >= 50% data. `old links` = the earlier QC set (also dropping sub-links the classical "
              "detector calls wet > 30% of the time). `reliable` = kept by the raw-signal reliability score "
              "(targ.reliability). `_nb` = after the neighbour check (10 km, ±15 min).", "",
              "| season | links | method | MCC 15-min | radar hourly r / RMSE / bias | gauges hourly r / bias | median per-link r | hours > 100 mm/h |",
              "|---|---|---|---|---|---|---|---|"]
    for name, s_ in out["italy"].items():
        for tag, blk, rob, n in (("all", s_, s_["robust"], s_["n_sublinks_qc"]),
                                 ("old links", s_["old_links"], s_["robust"]["old_links"], s_["n_sublinks_old_rule"]),
                                 ("reliable", s_["reliable_links"], s_["robust"]["reliable_links"],
                                  s_["n_sublinks_reliable"])):
            for mth in METHODS:
                t = blk[mth]
                g = t.get("rain_vs_gauges_hourly", {})
                lines.append(f"| {name} | {tag} ({n}) | {mth} | {t['wet_dry_vs_radar_15min']['MCC']:.3f} "
                             f"| {t['rain_vs_radar_hourly']['r']:.2f} / {t['rain_vs_radar_hourly']['RMSE']:.2f} / {t['rain_vs_radar_hourly']['bias_pct']:+.0f}% "
                             f"| {g.get('r', float('nan')):.2f} / {g.get('bias_pct', float('nan')):+.0f}% "
                             f"| {rob[mth]['median_per_link_hourly_r_vs_radar']:.2f} | {rob[mth]['hours_over_100mm_h']} |")
        nbi = s_["neighbour_check"]
        lines.append(f"\nNeighbour check {name}: removed {nbi['classical']['removed_share']:.1%} (classical) / "
                     f"{nbi['ai']['removed_share']:.1%} (AI) of rainy minutes; median {nbi['ai']['median_neighbours']:.0f} neighbours per sub-link.\n")
    N = out["netherlands"]
    lines += ["", f"## Netherlands, 1 Jun – 22 Aug 2014, 15-min min/max, zero-shot ({N['n_sublinks_qc']} sub-links; old QC set {N['n_sublinks_old_rule']})", "",
              "| links | method | MCC hourly | radar hourly r / RMSE / bias | median per-link r | hours > 100 mm/h |",
              "|---|---|---|---|---|---|"]
    for tag, blk, rob in (("all", N, N["robust"]), ("old links", N["old_links"], N["robust"]["old_links"]),
                          (f"reliable ({N['n_sublinks_reliable']})", N["reliable_links"], N["robust"]["reliable_links"])):
        for mth in METHODS:
            t = blk[mth]
            lines.append(f"| {tag} | {mth} | {t['wet_dry_vs_radar_hourly']['MCC']:.3f} | {t['rain_vs_radar_hourly']['r']:.2f} / "
                         f"{t['rain_vs_radar_hourly']['RMSE']:.2f} / {t['rain_vs_radar_hourly']['bias_pct']:+.0f}% "
                         f"| {rob[mth]['median_per_link_hourly_r_vs_radar']:.2f} | {rob[mth]['hours_over_100mm_h']} |")
    nbi = N["neighbour_check"]
    lines.append(f"\nNeighbour check: removed {nbi['classical']['removed_share']:.1%} (classical) / "
                 f"{nbi['ai']['removed_share']:.1%} (AI) of rainy 15-min steps; median {nbi['ai']['median_neighbours']:.0f} neighbours per sub-link.")
    LR = out["long_rain"]
    lines += ["", "## Rain after a mostly rainy day (past 24 h > 50% rainy)", "",
              "Rain measured as % of the reference total in those steps (before = 24-h median dry level; "
              "after = 3-day 20th-percentile dry level).", "",
              "| test | share of rainy steps | classical | AI before | AI after |", "|---|---|---|---|---|"]
    for k, rows in LR["after"].items():
        r = rows[-1]
        rb = LR["before"][k][-1] if LR["before"] else None
        fmt = lambda v: "–" if v is None else f"{v:.0f}%"
        lines.append(f"| {k} | {r['share_of_rainy_steps']:.1%} | {fmt(r['classical']['total_pct'])} "
                     f"| {fmt(rb['ai']['total_pct']) if rb else '–'} | {fmt(r['ai']['total_pct'])} |")
    lines += varied_section()
    lines += ["", "## MindSpore", "", f"- {ms['framework']}: PyTorch weights ported, max difference "
              f"{ms['port_check']['max_abs_diff_rain_mm_h']:.1e} mm/h over {ms['port_check']['n_values']} values; "
              f"rain/no-rain agreement {ms['port_check']['rain_no_rain_agreement']:.0%}.",
              f"- Training from scratch on CPU: loss {ms['train_smoke']['loss_first50']:.2f} → "
              f"{ms['train_smoke']['loss_last50']:.2f} in {ms['train_smoke']['steps']} steps.", ""]
    (RES / "phase2.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def varied_section():
    """Follow-up: why runs disagree, long rain in Italy, training on more varied data."""
    import json as _j
    f = RES / "phase2_italy_cv_mse-rel.json"
    if not f.exists():
        return []
    ds = _j.loads((RES / "phase2_diagnose_seeds.json").read_text())
    L = ["", "## Follow-up: why runs disagree, long rain in Italy, more varied training", "",
         "Runs differ in the strength of their rain (AI/radar where both say rain), not in rain/no-rain. "
         f"July bias vs August bias across runs: r = {ds['val_vs_test_bias_r']:.2f}.", "",
         "| run | July bias | July strength | Aug bias | Aug strength | Aug missed / false (share of radar) |",
         "|---|---|---|---|---|---|"]
    for r in ds["runs"]:
        L.append(f"| {r['run']} | {r['val']['bias_pct']:+.0f}% | {r['val']['amplitude_on_hits']:.2f} | {r['test']['bias_pct']:+.0f}% "
                 f"| {r['test']['amplitude_on_hits']:.2f} | {r['test']['missed_share_of_radar']:.0%} / {r['test']['false_share_of_radar']:.0%} |")
    L += ["", "One-season experiments (5 runs each; Sweden August, Italy 2022 held out; long-rain groups in 15-min steps):", "",
          "| config | Sweden MCC (mean) | Sweden gauge bias (runs) | Italy 2022 bias (runs) | IT22 after dry day | IT22 after mostly rainy day |",
          "|---|---|---|---|---|---|"]
    for c in ("se_mse", "se_log", "mix21_mse", "mix21_log", "mix21nb_mse", "mix21rel_mse"):
        p_ = RES / f"phase2_mixed_{c}.json"
        if not p_.exists():
            continue
        d = _j.loads(p_.read_text())
        sp, e = d["spread"], d["ensemble"]["italy_2022"]
        g, it = sp["sweden"]["gauges_bias_pct"], sp["italy_2022"]["bias_pct_all_steps"]
        L.append(f"| {c} | {sp['sweden']['MCC_5min']['mean']:.3f} | {g['min']:+.0f}% .. {g['max']:+.0f}% "
                 f"| {it['min']:+.0f}% .. {it['max']:+.0f}% | {e['pct_of_radar_0.00-0.10']:.0f}% | {e['pct_of_radar_0.50-1.00']:.0f}% |")
    d = _j.loads(f.read_text())
    sw = d["sweden_august"]
    L += ["", "Month-by-month cross-validation (Sweden + both Italian summers minus the judged month, reliable links only, "
          "3 runs per month). Hourly long-rain table, AI rain as % of radar, groups by share of past 24 h rainy "
          "(<10%, 10-25%, 25-50%, >50%):", "",
          "| season | Sweden-only kept model | varied training | Italy total bias (runs) |", "|---|---|---|---|"]
    for y in ("2021", "2022"):
        a_ = " / ".join(f"{r['ai']['total_pct']:.0f}%" for r in d["longrain_hourly_table"][y + "_sweden_only_kept_model"])
        b_ = " / ".join(f"{r['ai']['total_pct']:.0f}%" for r in d["longrain_hourly_table"][y])
        bb = d["spread_" + y]["bias_pct_all_steps"]
        L.append(f"| {y} | {a_} | {b_} | {bb['min']:+.0f}% .. {bb['max']:+.0f}% |")
    L += ["", f"Sweden August, same networks: MCC {sw['MCC_5min']['mean']:.3f}, gauges r {sw['gauges_r']['mean']:.2f}, "
          f"RMSE {sw['gauges_RMSE']['mean']:.2f}, bias {sw['gauges_bias_pct']['mean']:+.0f}% "
          f"(runs {sw['gauges_bias_pct']['min']:+.0f}% .. {sw['gauges_bias_pct']['max']:+.0f}%).",
          "Italy and Netherlands zero-shot results above (and Phase 3) keep the Sweden-only model on purpose."]
    return L


if __name__ == "__main__":
    main()
