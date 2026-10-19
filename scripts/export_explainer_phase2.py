"""Data files for the Phase 2 chapters of docs/explainer (all from real results).

    python scripts/export_explainer_phase2.py

storm_ai.dat     sub-link 718, 25 Aug 2015: AI rain (5-min mean), rain probability,
                 the AI's dB correction, excess loss over the 24-h median
italy_storm.dat  the Italian example storm (15-min values)
nl_daily.dat     Netherlands: network-average daily rain, AI / classical / radar
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from targ import features as feat
from targ.classical import total_loss
from targ.data import openmrg
from targ.models.tcn_torch import TCN

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "explainer"


def storm():
    p2 = json.loads((ROOT / "results" / "phase2_openmrg.json").read_text())
    v = p2["variants"]["physics"]
    ds = openmrg.load_cml_1min().sel(cml_id=[718], time=slice("2015-08-22", "2015-08-25T23:59")).load()
    tl = total_loss(ds, 5).values
    dyn = feat.dynamic_features(tl, ds.length.values, ds.frequency.values, ds.polarization.values)
    stat = feat.static_features(ds.length.values, ds.frequency.values, ds.polarization.values)
    phys = feat.physics_params(ds.length.values, ds.frequency.values, ds.polarization.values)
    m = TCN(4, 3, **v["model_kw"])
    m.load_state_dict(torch.load(ROOT / "runs" / "phase2" / "physics.pt", map_location="cpu"))
    m.eval()
    feats = {}
    m.head.register_forward_hook(lambda mod, i, o: feats.setdefault("out", o))
    with torch.no_grad():
        logit, rate = m(torch.from_numpy(dyn), torch.from_numpy(stat), torch.from_numpy(phys))
    corr = (-1.0 + 5.0 * torch.sigmoid(feats["out"][0, 1])).numpy()
    prob = torch.sigmoid(logit[0]).numpy()
    R = np.where(prob > v["threshold"], rate[0].numpy(), 0.0)
    t = pd.DatetimeIndex(ds.time.values)
    day = t >= pd.Timestamp("2015-08-25")
    R5 = pd.Series(R).rolling(5, center=True, min_periods=3).mean().values
    hour = (t[day] - pd.Timestamp("2015-08-25")).total_seconds() / 3600
    out = pd.DataFrame({"hour": hour, "ai": R5[day], "prob": prob[day], "corr": corr[day],
                        "excess": dyn[0, 0, day]})
    out.round(4).to_csv(DOC / "storm_ai.dat", sep=" ", index=False)
    i = int(np.argmin(np.abs(hour - 8.5)))
    print("08:30 →", out.iloc[i].to_dict(), "a,b,L", phys[0].tolist(), "threshold", v["threshold"])


def italy():
    ex = json.loads((ROOT / "results" / "phase2_italy_example.json").read_text())
    t = pd.to_datetime(ex["t15"])
    hour = (t - t[0].normalize()).total_seconds() / 3600
    pd.DataFrame({"hour": hour, "classical": ex["classical"], "ai": ex["ai"], "radar": ex["radar"],
                  "gauge": ex["gauge_mm_h"]}).round(3).to_csv(DOC / "italy_storm.dat", sep=" ",
                                                              index=False, na_rep="nan")
    print({k: ex[k] for k in ("cml_id", "gauge", "distance_m", "length_km", "frequency_ghz")}, ex["t15"][0])


def nl():
    s = json.loads((ROOT / "results" / "phase2_summary.json").read_text())["netherlands"]["network_daily_mean_mm"]
    df = pd.DataFrame({"day": np.arange(len(s["day"])), "date": s["day"], "ai": s["ai"],
                       "classical": s["classical"], "radar": s["radar"]})
    df.round(3).to_csv(DOC / "nl_daily.dat", sep=" ", index=False)
    print(df.sort_values("radar").tail(3))
    for k in ("ai", "classical"):
        print(k, "daily r", np.corrcoef(df[k], df.radar)[0, 1].round(3), "total ratio", (df[k].sum() / df.radar.sum()).round(3))


if __name__ == "__main__":
    storm()
    italy()
    nl()
