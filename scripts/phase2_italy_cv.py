"""Phase 2 follow-up: month-by-month cross-validation in Italy (more varied training).

    python scripts/phase2_italy_cv.py train MONTHS SEEDS [LOSS]  e.g. train 5,6 0,1,2 (data loaded once)
    python scripts/phase2_italy_cv.py eval [LOSS]                score all saved models
LOSS is "mse" or "log"; "<loss>-rel" trains only on Italian links kept by the
reliability score (targ.reliability). Further "+flags" as in phase2_mixed: +mfb (Swedish
training radar rescaled to the gauges, June-July only), +mask (ambiguous "dry" targets
left out), +gt (Italian links near a gauge learn from the gauge), +rf6 (4-hour memory),
e.g. mse-rel+mfb+mask+gt+rf6.

Why: the long-rain hours in Italy come from a few days (85% of the "> half of the
past day rainy" group in 2022 is one storm, 6 May 2022). A model trained on one
Italian season (phase2_mixed) has almost none of them to learn from. Here each
calendar month (May..Sep) is held out in turn, in BOTH years at once; the model for
that month is trained on Sweden (1 Jun-15 Jul) plus the other four months of both
Italian seasons. Every Italian month is then scored only by the model that never
saw it, so the whole of 2021 and 2022 gets honest out-of-sample estimates.

The last 7 days of every training month are kept apart for validation (checkpoint
and rain/no-rain threshold), together with Sweden 16-31 July. Sweden August is
never used and is scored for every model.

Writes runs/phase2/cv/<loss>_m<MONTH>_s<SEED>.pt (+ .json) and
results/phase2_italy_cv_<loss>.json.
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
import phase2_mixed as MX  # noqa: E402

from targ import train_torch as T  # noqa: E402
from targ.classical import ClassicalParams  # noqa: E402
from targ.models.tcn_torch import TCN  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "phase2" / "cv"
OUT = ROOT / "results"
MONTHS = (5, 6, 7, 8, 9)
YEARS = ("2021", "2022")
VAL_DAYS = 7
STEPS = int(__import__("os").environ.get("PHASE2_STEPS", 4000))


def pieces(times, year, held_out):
    """Training and validation index ranges for one season, skipping the held-out month."""
    tr, va = [], []
    for m in MONTHS:
        if m == held_out:
            continue
        a = int(np.searchsorted(times, np.datetime64(f"{year}-{m:02d}-01")))
        end = pd.Timestamp(f"{year}-{m:02d}-01") + pd.offsets.MonthBegin(1)
        b = int(np.searchsorted(times, np.datetime64(end)))
        v = int(np.searchsorted(times, np.datetime64(end - pd.Timedelta(days=VAL_DAYS))))
        tr.append((a, v))
        va.append((v, b))
    return tr, va


def train(months: list[int], seeds: list[int], loss: str = "mse"):
    RUNS.mkdir(parents=True, exist_ok=True)
    se = MX.sweden()
    flags = loss.split("+")[1:]
    its = {y: MX.italy_train(y, reliable_only="-rel" in loss) for y in YEARS}
    if "gt" in flags:
        its = {y: MX.gauge_target(it, y) for y, it in its.items()}
    if "mfb" in flags:
        se = MX.mfb_adjust(se)
    if "mask" in flags:
        se = MX.mask_ambiguous(se)
        its = {y: MX.mask_ambiguous(it) for y, it in its.items()}
    for month in months:
        for seed in seeds:
            if (RUNS / f"{loss}_m{month}_s{seed}.pt").exists():
                continue
            train_one(se, its, month, seed, loss)


def train_one(se, its, month: int, seed: int, loss: str):
    base = loss.split("-")[0].split("+")[0]     # rate loss; "-rel" and "+flags" change the data
    kw = json.loads((OUT / "phase2_openmrg.json").read_text())["variants"]["physics"]["model_kw"]
    if "rf6" in loss.split("+")[1:]:
        kw = {**kw, "n_levels": 6}
    for y, it in its.items():
        it["tr_pieces"], it["va_pieces"] = pieces(it["ds"].time.values, y, month)

    def val_preds(model):
        out = {"se": T.predict(model, se["dyn"], se["stat"], se["phys"], se["va"])}
        for y, it in its.items():
            out[y] = [T.predict(model, it["dyn"], it["stat"], it["phys"], r) for r in it["va_pieces"]]
        return out

    def val_fn(model):
        vp = val_preds(model)
        tot = 0.0
        P, R = vp["se"]
        logit = torch.logit(torch.from_numpy(P).clamp(1e-6, 1 - 1e-6))
        tot += float(T.loss_fn(logit, torch.from_numpy(R),
                               torch.from_numpy(se["target"][:, se["va"][0]:se["va"][1]]), 0.05, base))
        it_loss = []
        for y, it in its.items():
            for (P, R), (a, b) in zip(vp[y], it["va_pieces"]):
                logit = torch.logit(torch.from_numpy(P).clamp(1e-6, 1 - 1e-6))
                it_loss.append(float(T.loss_fn(logit, torch.from_numpy(R), torch.from_numpy(it["target"][:, a:b]),
                                               0.05, base)))
        return tot + float(np.mean(it_loss))

    domains = [{"dyn": [se["dyn"], se["dyn_q"]], "stat": se["stat"], "phys": se["phys"],
                "target": se["target"], "t_range": se["tr"], "weight": 1.0}]
    for it in its.values():
        domains.append({"dyn": [it["dyn"]], "stat": it["stat"], "phys": it["phys"], "target": it["target"],
                        "t_ranges": it["tr_pieces"], "weight": 0.5})
    print(f"== {loss} hold out month {month}, seed {seed}", flush=True)
    model, hist = T.train_multi(domains, val_fn=val_fn, steps=STEPS, seed=seed, model_kw=kw, log_every=500,
                                rate_loss=base)
    f = RUNS / f"{loss}_m{month}_s{seed}.pt"
    torch.save(model.state_dict(), f.with_suffix(".tmp"))
    # threshold: best mean MCC over Sweden validation and the Italian validation days
    vp = val_preds(model)
    best = None
    for thr in MX.THRESHOLDS:
        P, R = vp["se"]
        wet = MX.P2.as_da((P > thr).astype(float), se["ds"], *se["va"])
        m_se = MX.wet_dry_scores(se["radar_val"].values,
                                 MX.to_radar_times(wet, se["radar_val"].time.values).values)["MCC"]
        est, ref = [], []
        for y, it in its.items():
            for (P, R), (a, b) in zip(vp[y], it["va_pieces"]):
                est.append(DL.block15(np.where(P > thr, R, 0.0)).ravel())
                ref.append(DL.block15(it["target"][:, a:b]).ravel())
        m_it = MX.wet_dry_scores(np.concatenate(ref), (np.concatenate(est) >= 0.1).astype(float))["MCC"]
        if best is None or (m_se + m_it) / 2 > best[0]:
            best = ((m_se + m_it) / 2, float(thr))
    f.with_suffix(".tmp").rename(f)
    f.with_suffix(".json").write_text(json.dumps({"month": month, "seed": seed, "loss": loss, "threshold": best[1], "model_kw": kw,
                                                  "val_mcc_mean": best[0], "history": hist}, indent=1))
    print("saved", f.name, "threshold", best[1], flush=True)


def to_hourly(x15: np.ndarray) -> np.ndarray:
    """15-min blocks (end-labelled, starting at 00:00) -> hourly means; NaN unless all 4 exist."""
    n, k = x15.shape
    h = x15[:, :k // 4 * 4].reshape(n, k // 4, 4)
    return np.where(np.isfinite(h).all(axis=2), h.mean(axis=2), np.nan)


def hourly_table(radar15: np.ndarray, est15: np.ndarray):
    import phase2_longrain as LR
    rh, eh = to_hourly(radar15), to_hourly(est15)
    wet = np.where(np.isfinite(rh), (rh >= 0.1).astype(float), np.nan)
    wf = pd.DataFrame(wet.T).rolling(24, min_periods=12).mean().shift(1).to_numpy().T
    return LR.table(rh, {"ai": eh}, wf)


def evaluate(loss: str = "mse"):
    kw = json.loads((OUT / "phase2_openmrg.json").read_text())["variants"]["physics"]["model_kw"]
    cp = ClassicalParams(**json.loads((OUT / "phase1_openmrg.json").read_text())["params"])
    models = {}
    for f in sorted(glob.glob(str(RUNS / f"{loss}_m*_s*.pt"))):
        meta = json.loads(Path(f).with_suffix(".json").read_text())
        m = TCN(5, 3, **{**kw, **meta.get("model_kw", {})}).to(T.device())
        m.load_state_dict(torch.load(f))
        m.eval()
        models[(meta["month"], meta["seed"])] = (m, meta["threshold"])
    seeds = sorted({s for _, s in models})
    assert all((mo, s) in models for mo in MONTHS for s in seeds), sorted(models)
    res = {"loss": loss, "seeds": seeds, "steps": STEPS, "per_seed": {}, "ensemble": {}, "sweden_august": {},
           "longrain_hourly_table": {}}

    se = MX.sweden()
    sw = []
    for (mo, s), (m, thr) in sorted(models.items()):
        P, R = T.predict(m, se["dyn"], se["stat"], se["phys"], se["te"])
        sw.append(MX.sweden_scores(P, R, thr, se))
    res["sweden_august"] = {k: MX.spread(sw, k) for k in sw[0]}
    del se

    for y in YEARS:
        c = DL.prepare([f"{y}{mm:02d}" for mm in MONTHS], cp)
        t0 = pd.Timestamp(c["ds"].time.values[0])
        month_of_min = (t0 + pd.to_timedelta(np.arange(c["tl"].shape[1]), "min")).month.values
        gate = {s: np.zeros(c["tl"].shape, np.float32) for s in seeds}
        rate = {s: np.zeros(c["tl"].shape, np.float32) for s in seeds}
        for (mo, s), (m, thr) in models.items():
            P, R = T.predict(m, c["dyn"], c["stat"], c["phys"])
            cols = month_of_min == mo
            gate[s][:, cols] = (P[:, cols] > thr)
            rate[s][:, cols] = R[:, cols]
        rows = []
        for s in seeds:
            # gate already applied per fold, so evaluate with threshold 0.5 on the 0/1 gate
            rows.append(MX.italy_scores(c, gate[s], rate[s], 0.5))
        res["per_seed"][y] = rows
        res["spread_" + y] = {k: MX.spread(rows, k) for k in rows[0] if rows[0][k] is not None}
        Pe = np.mean([gate[s] for s in seeds], axis=0)
        Re = np.mean([rate[s] for s in seeds], axis=0)
        res["ensemble"][y] = MX.italy_scores(c, Pe, Re, 0.5)     # majority vote of the seeds
        res["ensemble"][y + "_full"] = DL.evaluate(c, Pe, Re, 0.5)
        # the same hourly long-rain table as phase2_longrain.py (same links, same groups)
        gated = np.where(Pe > 0.5, Re, 0.0)
        gated[~np.isfinite(c["tl"])] = np.nan
        res["longrain_hourly_table"][y] = hourly_table(c["radar"], DL.block15(gated))
        # check: the kept Sweden-only model through the same code (should match phase2_longrain.json)
        m_se, thr_se = DL.model_and_thr(ROOT / "runs" / "phase2" / "physics_selected.pt")
        P, R = DL.predict(c, m_se, thr_se)
        g_se = np.where(P > thr_se, R, 0.0)
        g_se[~np.isfinite(c["tl"])] = np.nan
        res["longrain_hourly_table"][y + "_sweden_only_kept_model"] = hourly_table(c["radar"], DL.block15(g_se))
        del c
    (OUT / f"phase2_italy_cv_{loss}.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("spread") or k == "sweden_august"},
                     indent=1, default=float)[:5000])


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train([int(x) for x in sys.argv[2].split(",")], [int(x) for x in sys.argv[3].split(",")],
              *(sys.argv[4:5]))
    else:
        evaluate(*(sys.argv[2:3]))
