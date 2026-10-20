"""Data for the interactive Phase 4 storm explorer (site/phase4/index.html).

    python scripts/export_phase4_explorer.py

Embeds, per city: hourly simulated truth, AI map (all links), AI map (3% of links),
the links (both sets), city-average curves (truth, AI, classical, satellite) and the
headline scores. Writes site/phase4/index.html from site/phase4/template.html.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "data" / "processed" / "phase4"
RES = ROOT / "results"
SITE = ROOT / "site" / "phase4"
NAMES = {"algiers": ("Algiers", "Bab El Oued flood, 9–10 Nov 2001"),
         "ghardaia": ("Ghardaïa", "M'zab valley flood, 29 Sep – 1 Oct 2008")}


def q(a, nd=1):
    """Round to nd decimals; NaN -> None (JSON null)."""
    a = np.round(np.asarray(a, float), nd)
    return [None if not np.isfinite(v) else (int(v) if nd == 0 else float(v)) for v in a.ravel()]


def city(c):
    z = np.load(P4 / f"sim_{c}.npz", allow_pickle=True)
    r = json.loads((RES / f"phase4_{c}.json").read_text())
    links = pd.read_csv(RES / f"phase4_links_{c}.csv")
    sparse = set(z["sparse_ids"])
    city_dat = pd.read_csv(ROOT / "docs" / "explainer" / f"p4_city_{c}.dat", sep=" ")
    H, ny, nx = z["truth"].shape
    return {
        "key": c, "name": NAMES[c][0], "event": NAMES[c][1],
        "hours": [str(pd.Timestamp(h))[:16] for h in z["hours"]],
        "ny": ny, "nx": nx, "lat": [float(z["lat"][0]), float(z["lat"][-1])],
        "lon": [float(z["lon"][0]), float(z["lon"][-1])],
        "truth": q(z["truth"]), "ai": q(z["ai"]), "ai_sparse": q(z["ai_sparse"]),
        "links": [[round(a, 4), round(b, 4), round(cc, 4), round(d, 4), int(i in sparse)]
                  for a, b, cc, d, i in zip(links.site_0_lon, links.site_0_lat, links.site_1_lon,
                                            links.site_1_lat, links.link_id)],
        "series": {k: q(city_dat[k], 2) for k in ("truth", "ai", "classical", "satellite")},
        "scores": {"n_links": r["n_links"], "n_sparse": r["maps"]["0.03"]["n_links"], "n_sites": r["n_sites"],
                   "reliable": r["n_reliable"], "sub_links": r["n_sub_links"],
                   "link_r_ai": r["links"]["ai"]["hourly_r"], "link_r_cl": r["links"]["classical"]["hourly_r"],
                   "map_r_ai": r["maps"]["1.00"]["ai"]["pixel_hourly_r"],
                   "map_r_sparse": r["maps"]["0.03"]["ai"]["pixel_hourly_r"],
                   "total_truth": r["maps"]["1.00"]["ai"]["city_total_truth_mm"],
                   "total_ai": r["maps"]["1.00"]["ai"]["city_total_est_mm"],
                   "total_cl": r["maps"]["1.00"]["classical"]["city_total_est_mm"]},
    }


def main():
    cities = [city(c) for c in ("algiers", "ghardaia") if (P4 / f"sim_{c}.npz").exists()]
    html = (SITE / "template.html").read_text()
    html = html.replace("/*DATA*/null", json.dumps(cities, separators=(",", ":")))
    (SITE / "index.html").write_text(html)
    print("wrote", SITE / "index.html", f"{len(html) / 1e6:.2f} MB", [c["key"] for c in cities])


if __name__ == "__main__":
    main()
