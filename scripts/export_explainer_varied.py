"""Data for the explainer figure on the 6 May 2022 storm (Italy), hourly.

    python scripts/export_explainer_varied.py           figure data (needs the May models)
    python scripts/export_explainer_varied.py numbers   docs/explainer/varied_numbers.tex

Link 182_1 (5.3 km, 24.6 GHz), 508 m from the Castelnovo ne' Monti gauge.
Columns: hours since 5 May 12:00 UTC, extra loss above the AI's dry level [dB],
loss the radar rain should cause (ITU-R P.838) [dB], radar along the path, the gauge,
AI trained on Sweden only (kept model), AI from the cross-validation (models that
never saw May, trained on reliable Italian links; average of 3 runs) [mm/h].
Writes docs/explainer/may6.dat.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_diagnose_longrain as DL  # noqa: E402

from targ import train_torch as T  # noqa: E402
from targ.classical import ClassicalParams  # noqa: E402
from targ.data import openrainer  # noqa: E402
from targ.models.tcn_torch import TCN  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LINK, GAUGE = "182_1", "Castelnovo ne' Monti_103947_4443487"
START, END = "2022-05-05T12:00", "2022-05-08T00:00"


def main():
    cp = ClassicalParams(**json.loads((ROOT / "results" / "phase1_openmrg.json").read_text())["params"])
    c = DL.prepare(["202205"], cp)
    i = int(np.where(c["ds"].cml_id.values == LINK)[0][0])
    one = {k: c[k][i:i + 1] for k in ("dyn", "stat", "phys")}
    kw = json.loads((ROOT / "results" / "phase2_openmrg.json").read_text())["variants"]["physics"]["model_kw"]

    def run(path, thr):
        m = TCN(5, 3, **kw).to(T.device())
        m.load_state_dict(torch.load(path, map_location=T.device()))
        P, R = T.predict(m, one["dyn"], one["stat"], one["phys"])
        x = np.where(P > thr, R, 0.0)
        x[~np.isfinite(c["tl"][i:i + 1])] = np.nan
        return DL.block15(x)[0]
    m_se, thr_se = DL.model_and_thr(ROOT / "runs" / "phase2" / "physics_selected.pt")
    ai_se = run(ROOT / "runs" / "phase2" / "physics_selected.pt", thr_se)
    cv = []
    for f in sorted(glob.glob(str(ROOT / "runs" / "phase2" / "cv" / "mse-rel_m5_s*.pt"))):
        cv.append(run(f, json.loads(Path(f).with_suffix(".json").read_text())["threshold"]))
    ai_cv = np.mean(cv, axis=0)
    t0 = pd.Timestamp(c["ds"].time.values[0])
    t15 = t0 + pd.to_timedelta(15 * (np.arange(c["radar"].shape[1]) + 1), "min")
    g = openrainer.gauges(["202205"]).rainfall_rate.sel(id=GAUGE).reindex(time=t15.values).values
    df = pd.DataFrame({"excess": c["blk"]["ex"][i], "expected": c["A_exp"][i], "radar": c["radar"][i],
                       "gauge": g, "ai_se": ai_se, "ai_cv": ai_cv}, index=t15)
    h = df.resample("1h", label="right", closed="right").mean().loc[START:END]
    h.insert(0, "hour", ((h.index - pd.Timestamp(START)) / pd.Timedelta("1h")).astype(int))
    out = ROOT / "docs" / "explainer" / "may6.dat"
    h.round(3).to_csv(out, sep=" ", index=False, na_rep="nan")
    tot = h[["radar", "gauge", "ai_se", "ai_cv"]].sum()
    print(tot.round(1).to_dict(), "n models cv", len(cv))
    (ROOT / "results" / "phase2_may6_example.json").write_text(json.dumps(
        {"link": LINK, "gauge": GAUGE, "distance_m": 508, "window": [START, END],
         "totals_mm": {k: float(v) for k, v in tot.items()}}, indent=1))


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


# ------------------------------------------------------------------ numbers for the text
def fmt_pct(x, sign=False):
    s = f"{x:+.0f}" if sign else f"{x:.0f}"
    return s.replace("-", "$-$").replace("+", "$+$") + r"\%"


def numbers():
    """Write docs/explainer/varied_numbers.tex: every number of the explainer section,
    read from the result files (so text and results cannot drift apart)."""
    R = ROOT / "results"
    L = []

    def cmd(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # why runs disagree (phase2_diagnose_seeds.json)
    ds = json.loads((R / "phase2_diagnose_seeds.json").read_text())
    rows = []
    for r in ds["runs"]:
        s = r["run"].split("seed")[-1].split(".")[0]
        rows.append(f"{s} & {fmt_pct(r['val']['bias_pct'], True)} & {r['val']['amplitude_on_hits']:.2f} & "
                    f"{fmt_pct(r['test']['bias_pct'], True)} & {r['test']['amplitude_on_hits']:.2f} & "
                    f"{100 * r['test']['missed_share_of_radar']:.0f}\\% / {100 * r['test']['false_share_of_radar']:.0f}\\%\\\\")
    cmd("SeedRows", "\n".join(rows))
    cmd("SeedValTestR", f"{ds['val_vs_test_bias_r']:.2f}")

    # one-season experiments (phase2_mixed_*.json)
    names = {"se_mse": "Sweden only (as before)", "se_log": "Sweden only, rate error in log scale",
             "mix21_mse": "Sweden + Italy 2021", "mix21_log": "Sweden + Italy 2021, log scale",
             "mix21nb_mse": "Sweden + Italy 2021 + neighbour inputs",
             "mix21rel_mse": "\\textbf{Sweden + Italy 2021, reliable links only}"}
    rows = []
    for c, label in names.items():
        d = json.loads((R / f"phase2_mixed_{c}.json").read_text())
        sp, e = d["spread"], d["ensemble"]
        g = sp["sweden"]["gauges_bias_pct"]
        it = sp["italy_2022"]["bias_pct_all_steps"]
        i = e["italy_2022"]
        rows.append(f"{label} & {sp['sweden']['MCC_5min']['mean']:.2f} & {fmt_pct(g['min'], True)} to {fmt_pct(g['max'], True)}"
                    f" & {fmt_pct(it['min'], True)} to {fmt_pct(it['max'], True)} & "
                    f"{fmt_pct(i['pct_of_radar_0.00-0.10'])} & {fmt_pct(i['pct_of_radar_0.50-1.00'])}\\\\")
    cmd("ExpRows", "\n".join(rows))

    # month-by-month cross-validation (phase2_italy_cv_*.json)
    for tag, key in (("mse", "CVall"), ("mse-rel", "CVrel")):
        f = R / f"phase2_italy_cv_{tag}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        sw = d["sweden_august"]
        cmd(key + "SeMCC", f"{sw['MCC_5min']['mean']:.3f}")
        cmd(key + "SeR", f"{sw['gauges_r']['mean']:.2f}")
        cmd(key + "SeRMSE", f"{sw['gauges_RMSE']['mean']:.2f}")
        cmd(key + "SeBias", fmt_pct(sw["gauges_bias_pct"]["mean"], True))
        cmd(key + "SeBiasRange", f"{fmt_pct(sw['gauges_bias_pct']['min'], True)} to {fmt_pct(sw['gauges_bias_pct']['max'], True)}")
        for y in ("2021", "2022"):
            b = d["spread_" + y]["bias_pct_all_steps"]
            cmd(key + "Bias" + ("A" if y == "2021" else "B"),
                f"{fmt_pct(b['min'], True)} to {fmt_pct(b['max'], True)}")
            for k, row in enumerate(d["longrain_hourly_table"][y]):
                cmd(key + "Long" + ("A" if y == "2021" else "B") + "abcd"[k], fmt_pct(row["ai"]["total_pct"]))
            for k, row in enumerate(d["longrain_hourly_table"][y + "_sweden_only_kept_model"]):
                if key == "CVall":       # same model in both files: write once
                    continue
                cmd("SeOnlyLong" + ("A" if y == "2021" else "B") + "abcd"[k], fmt_pct(row["ai"]["total_pct"]))
            e = d["ensemble"][y]
            cmd(key + "Gauges" + ("A" if y == "2021" else "B"), fmt_pct(e["pct_of_gauges_0.00-1.00"]))

    # the 6 May storm
    ex = json.loads((R / "phase2_may6_example.json").read_text())["totals_mm"]
    for k, v in ex.items():
        cmd("May" + k.replace("_", ""), f"{v:.0f}")
    out = ROOT / "docs" / "explainer" / "varied_numbers.tex"
    out.write_text("% generated by scripts/export_explainer_varied.py -- do not edit\n" + "\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "numbers":
    numbers()
