"""Phase 6: flood reruns with full detail for the dashboard (SIMULATION; see docs/phase6/PLAN.md).

Same ground, rain and settings as Phase 5's gauge-matched runs (scripts/phase5/run_floods.py), but the water is
saved every hour at the full 30 m (Algiers) / 60 m (Ghardaia) resolution, with its speed and direction, so the
dashboard can draw a moving water surface. Phase 5's own runs and results are not touched.

    python scripts/phase6/run_dashboard_floods.py [algiers|ghardaia ...] [--force]
Writes data/processed/phase6/<city>/gauge_<source>.npz and checks that the basin outflow matches Phase 5's run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase5"))

from floodrisk import config, rain, solver  # noqa: E402
import run_floods as RF  # noqa: E402  (setup() only: ground, terrain, basins, communes)

PROC6 = ROOT / "data" / "processed" / "phase6"
SOURCES = ("truth", "ai", "classical", "imerg")


def fields_for(key, S):
    """Hourly rain on the flood grid per source, gauge-matched, exactly as Phase 5 builds it."""
    city = config.CITIES[key]
    summ = json.loads((config.RESULTS / f"{key}.json").read_text())
    p4 = rain.Phase4Rain(key)
    out = {}
    for s in SOURCES:
        f, _ = p4.fields(s, S["g"].lat, S["g"].lon)
        f = np.concatenate([f, np.zeros((city.spin_dry_h, *f.shape[1:]), np.float32)])
        out[s] = f * summ["gauge_factor"]
    return out, summ


def run_city(key, force=False):
    city = config.CITIES[key]
    out_dir = PROC6 / key
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"== {city.name}", flush=True)
    S = RF.setup(city)
    g = S["g"]
    fields, summ = fields_for(key, S)
    bkey = city.outlets[0].key
    for s in SOURCES:
        f = out_dir / f"gauge_{s}.npz"
        if f.exists() and not force:
            print(f"  {s}: cached", flush=True)
            continue
        p = solver.Params(snap_s=3600.0, snap_coarsen=1, snap_flow=True)
        res = solver.simulate(S["z"], S["n"], S["cn"], g.dx, g.dy, solver.HourlyRain(fields[s]),
                              fields[s].shape[0] * 3600.0, sink=S["sea"],
                              gauges={k: b["mask"] for k, b in S["basins"].items()}, params=p)
        np.savez_compressed(f, t=res.t_record, q=res.q_gauge[bkey], h=res.snaps, uv=res.snaps_flow,
                            t_snaps=res.t_snaps, hmax=res.hmax.astype(np.float16),
                            hrmax=res.hrmax.astype(np.float16), rain=fields[s].astype(np.float16))
        ref = np.load(config.PROC / key / f"gauge_{s}.npz")[f"q_{bkey}"]
        n = min(ref.size, res.q_gauge[bkey].size)
        diff = float(np.abs(ref[:n] - res.q_gauge[bkey][:n]).max())
        print(f"  {s}: {res.wall_s:.0f} s, {res.snaps.shape[0]} frames, peak {res.q_gauge[bkey].max():.1f} m3/s, "
              f"max difference from Phase 5 run {diff:.2e} m3/s", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for k in args or list(config.CITIES):
        run_city(k, force="--force" in sys.argv)
