"""Phase 7: correct the AI's low reading of heavy rain, for the live watch only.

The kept Phase 2 model (mix21rel+mfb+mask+gt+west_mse) reads heavy rain too low (Sweden gauges -7 to -11%), and in
the Phase 5 replay its flood alarm came late. The classical method cannot be used live (it needs 30 minutes of future
signal), so the live watch corrects the AI instead:

  quantile mapping of hourly AI rain onto the hourly rain of the city gauges next to the links (within 1 km),
  fitted on Sweden 1 June - 31 July 2015 ONLY (the model's own training and validation months).

A first try mapped onto the gauge-adjusted radar instead; it made heavy rain worse (August heavy hours -10% -> -20%),
because the radar itself reads the heaviest rain lower than the AI does. The gauges are the honest reference.

Result (August, never used for fitting): all rain -7.3% -> -1.8%, RMSE 0.905 -> 0.877, r 0.853 -> 0.861, but hours
with >= 5 mm at the gauge -10.4% -> -11.8%. A gain that grows with intensity, fitted on June-July heavy hours, was
also tried: June-July heavy hours are -35% while August's are -10%, so it overshot August (+40% heavy, +22% overall).
The heavy-rain shortfall is not stable from month to month, so no fixed correction removes it, and the one that
helps overall makes the hours that decide a flood alarm slightly worse. Decision: NOT used by the live watch
("use": false). The real fix is in training (more weight on heavy rain), a separate piece of work.

August 2015, the Phase 2 test month, is never used for the fit; it is only scored, before and after, against the
city rain gauges. The Phase 2 results themselves are not changed.

    python scripts/phase7_heavy_rain_correction.py
Writes results/phase7_heavy_rain_correction.json (read by src/watch/engine.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_mixed as M  # noqa: E402
import phase2_openmrg as P2  # noqa: E402

from targ import train_torch as T  # noqa: E402
from targ.evaluation import OPENMRG_CALIB  # noqa: E402
from targ.models.tcn_torch import TCN  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase7_heavy_rain_correction.json"
CFG = "mix21rel+mfb+mask+gt+west_mse"
PROBS = np.r_[np.linspace(0.0, 0.9, 19), 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]


def main() -> None:
    se = M.mfb_adjust(M.sweden())                          # radar rescaled to the gauges in June-July only
    kw = json.loads((ROOT / "results" / "phase2_openmrg.json").read_text())["variants"]["physics"]["model_kw"]
    thr = json.loads((ROOT / "results" / f"phase2_mixed_{CFG}.json").read_text())["ensemble"]["threshold"]
    Ps, Rs = [], []
    for seed in range(5):
        m = TCN(se["dyn"].shape[1], se["stat"].shape[1], **kw).to(T.device())
        m.load_state_dict(torch.load(M.RUNS / f"{CFG}_seed{seed}.pt", map_location=T.device()))
        P, R = T.predict(m, se["dyn"], se["stat"], se["phys"])
        Ps.append(P)
        Rs.append(R)
    P, R = np.mean(Ps, 0), np.mean(Rs, 0)
    est = np.where(P > thr, R, 0.0)

    # fit: hourly AI rain vs the nearby gauges, June-July only, hours either side calls wet
    fit_end = int(np.searchsorted(se["ds"].time.values, np.datetime64(OPENMRG_CALIB.stop) + np.timedelta64(1, "D")))
    g_fit, e_fit = P2.gauge_series(P2.as_da(est[:, :fit_end], se["ds"], 0, fit_end), se["gauges"], se["pairs"])
    g_fit, e_fit = np.asarray(g_fit, float), np.asarray(e_fit, float)
    ok = np.isfinite(g_fit) & np.isfinite(e_fit) & ((g_fit >= 0.1) | (e_fit >= 0.1))
    read_q = np.quantile(e_fit[ok], PROBS)
    true_q = np.quantile(g_fit[ok], PROBS)
    read_q, true_q = np.maximum.accumulate(read_q), np.maximum.accumulate(true_q)
    keep = np.r_[True, np.diff(read_q) > 1e-3]
    read_q, true_q = read_q[keep], true_q[keep]
    top = float(true_q[-1] / max(read_q[-1], 1e-6))
    read_q = np.r_[0.0, read_q[1:], 300.0]                 # start at 0, and keep the top ratio beyond the data
    true_q = np.r_[0.0, true_q[1:], 300.0 * top]
    true_q = np.maximum(true_q, 0.0)

    # score on the untouched test month against the city gauges, before and after
    a, b = se["te"]
    wet = P2.as_da((P[:, a:b] > thr).astype(float), se["ds"], a, b)
    before = P2.scores(wet, P2.as_da(est[:, a:b], se["ds"], a, b), se["radar_te"], se["gauges"], se["pairs"])
    corr = np.interp(est[:, a:b], read_q, true_q)
    after = P2.scores(wet, P2.as_da(corr, se["ds"], a, b), se["radar_te"], se["gauges"], se["pairs"])
    g = lambda s: {k: round(float(s["rain_vs_gauges_hourly"][k]), 3) for k in ("r", "RMSE", "bias_pct")}   # noqa: E731
    # heavy hours: gauge hourly rain >= 5 mm
    ref, e_b = P2.gauge_series(P2.as_da(est[:, a:b], se["ds"], a, b), se["gauges"], se["pairs"])
    _, e_a = P2.gauge_series(P2.as_da(corr, se["ds"], a, b), se["gauges"], se["pairs"])
    rv, bv, av = (np.asarray(x, float).ravel() for x in (ref, e_b, e_a))
    h = np.isfinite(rv) & np.isfinite(bv) & np.isfinite(av) & (rv >= 5)
    heavy = {"hours": int(h.sum()), "bias_pct_before": round(float(100 * (bv[h].sum() / rv[h].sum() - 1)), 1),
             "bias_pct_after": round(float(100 * (av[h].sum() / rv[h].sum() - 1)), 1)}
    res = {"model": CFG, "fit": "quantile mapping, hourly AI rain vs city gauges within 1 km of the links, "
                                "Sweden 2015-06-01..07-31 (training + validation months only)",
           "use": False,
           "why_not_used": "improves all rain on the test month but heavy hours get slightly worse (-10.4% -> -11.8%); "
                           "a heavy-rain gain fitted on June-July overshoots August. See the docstring.",
           "fit_steps": int(ok.sum()), "read_mmh": [round(float(v), 3) for v in read_q],
           "true_mmh": [round(float(v), 3) for v in true_q],
           "test_august_vs_gauges": {"before": g(before), "after": g(after), "heavy_hours_ge_5mm": heavy}}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["test_august_vs_gauges"], indent=1))
    print("mapping:", list(zip(res["read_mmh"], res["true_mmh"]))[-8:])


if __name__ == "__main__":
    main()
