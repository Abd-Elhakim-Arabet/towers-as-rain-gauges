"""Phase 4, step 1: real storms over Algeria from NASA IMERG (satellite, half-hourly, 0.1 deg).

Two well-documented flash floods, one per kind of place:
  algiers   Bab El Oued, 9-10 November 2001 (dense coastal city, steep streets to the sea)
  ghardaia  M'zab valley, flood on 1 October 2008 (desert valley along a dry riverbed); IMERG puts
            most rain on 29 September and a second burst on 1 October
Windows start 3 days before the storm (the AI dry level looks back 3 days) and are
generous; the satellite data itself shows when each storm peaked.

Writes data/processed/phase4/storm_<city>.nc and results/phase4_storms.json
(totals and peaks over the city box, so the chosen dates can be checked).

    python scripts/phase4_storms.py [algiers|ghardaia ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from targ.data import imerg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# box (lat0, lon0, lat1, lon1): the region around the city, big enough to see the storm move;
# city (lat0, lon0, lat1, lon1): the tower sweep area of scripts/fetch_opencellid_area.py
EVENTS = {
    "algiers": {"name": "Bab El Oued flood", "box": (35.8, 2.0, 37.2, 4.0),
                "city": (36.65, 2.90, 36.82, 3.25), "start": "2001-11-06T00:00", "end": "2001-11-11T00:00"},
    "ghardaia": {"name": "Ghardaia (M'zab) flood", "box": (31.8, 2.8, 33.2, 4.6),
                 "city": (32.40, 3.60, 32.55, 3.78), "start": "2008-09-26T00:00", "end": "2008-10-02T00:00"},
}


def main(keys=None):
    out = ROOT / "data" / "processed" / "phase4"
    out.mkdir(parents=True, exist_ok=True)
    f = ROOT / "results" / "phase4_storms.json"
    summary = json.loads(f.read_text()) if f.exists() else {}
    for key in (keys or EVENTS):
        ev = EVENTS[key]
        da = imerg.fetch(key, ev["box"], ev["start"], ev["end"])
        da.to_netcdf(out / f"storm_{key}.nc")
        lat0, lon0, lat1, lon1 = ev["city"]
        # city mean: IMERG cells whose centre is within 0.05 deg of the city box
        c = da.sel(lat=slice(lat0 - 0.05, lat1 + 0.05), lon=slice(lon0 - 0.05, lon1 + 0.05)).mean(("lat", "lon"))
        h = c.resample(time="1h").mean()
        d = c.resample(time="1D").mean() * 24
        summary[key] = {
            "event": ev["name"], "window": [ev["start"], ev["end"]], "box": ev["box"], "city_box": ev["city"],
            "city_total_mm": float(c.sum() * 0.5),
            "city_daily_mm": {str(t)[:10]: round(float(v), 1) for t, v in zip(d.time.values, d.values)},
            "city_peak_hour": str(h.idxmax().values)[:16], "city_peak_mm_h": float(h.max()),
            "box_max_cell_total_mm": float(da.sum("time").max() * 0.5),
            "box_max_halfhour_mm_h": float(da.max()),
            "missing_halfhours": int(np.isnan(da.values).all(axis=(1, 2)).sum()),
        }
        print(key, json.dumps(summary[key], indent=1), flush=True)
        f.write_text(json.dumps(summary, indent=1))     # after each city


if __name__ == "__main__":
    main(sys.argv[1:] or None)
