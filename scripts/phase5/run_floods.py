"""Phase 5: flood model runs on the Phase 4 storms (SIMULATION; see docs/phase5/PLAN.md).

For each city:
  1. ground (terrain, land cover, soil) -> filled terrain, drainage, checked basin, communes
  2. rain: every Phase 4 source (truth / ai / classical / ai_sparse / imerg), two scenarios:
       satellite  the storm as Phase 4 simulated it
       gauge      every source x one factor, so the basin storm total matches the gauge record
  3. a 2D flood run per (scenario, source), plus sensitivity runs on the truth
  4. flash-flood rain thresholds from made-up uniform storms, and the alarm each source would have raised
  5. scores against the run on the true rain

    python scripts/phase5/run_floods.py [algiers|ghardaia ...] [--force]
Writes results/phase5/<city>.json and data/processed/phase5/<city>/*.npz (runs, for figures and the website).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from floodrisk import config, ground, hazard, rain, runoff, solver, terrain  # noqa: E402

SOURCES = ("truth", "ai", "classical", "ai_sparse", "imerg")
SNAP_SOURCES = ("truth", "ai", "classical", "imerg")


def setup(city: config.City) -> dict:
    g = ground.build(city)
    sea = g.sea
    lake = (g.landcover == 80) & ~sea
    zc, filled = terrain.condition(g.z, sea, keep=lake)
    zf = terrain.fill(g.z, sea)
    d8 = terrain.d8(zf, g.dx, g.dy, sea)
    cell = g.dx * g.dy
    acc = terrain.accumulate(zf, d8) * cell / 1e6
    basins = {}
    for o in city.outlets:
        r, c = g.index(o.lat, o.lon)
        r, c = terrain.snap(acc, r, c, max(1, int(round(o.snap_m / g.dy))))
        basins[o.key] = {"mask": terrain.basin(zf, d8, r, c), "rc": (int(r), int(c)),
                         "lat": float(g.lat[r]), "lon": float(g.lon[c]), "area_km2": float(acc[r, c])}
    labels, names = hazard.commune_raster(config.COMMUNES, g.lat, g.lon)
    zone = np.isin(labels, [names.index(n) + 1 for n in city.alarm_communes if n in names])
    return {"g": g, "sea": sea, "lake": lake, "z": zc, "z_raw": g.z.astype(np.float64), "filled": filled,
            "cell": cell, "basins": basins, "labels": labels, "names": names, "zone": zone,
            "built_ha": hazard.built_area_ha(g.built, cell),
            "cn": runoff.curve_numbers(g.landcover, g.hsg), "n": runoff.manning(g.landcover)}


def run(S, fields, name, out_dir, force=False, cn=None, n=None, z=None, drain=None, snaps=False):
    f = out_dir / f"{name}.npz"
    if f.exists() and not force:
        return dict(np.load(f, allow_pickle=True))
    g = S["g"]
    p = solver.Params(snap_s=1800.0 if snaps else 1e12, snap_coarsen=3)
    t0 = time.time()
    res = solver.simulate(S["z"] if z is None else z, S["n"] if n is None else n, S["cn"] if cn is None else cn,
                          g.dx, g.dy, solver.HourlyRain(fields), fields.shape[0] * 3600.0, sink=S["sea"],
                          gauges={k: b["mask"] for k, b in S["basins"].items()}, params=p, drain_mmh=drain)
    out = {"t": res.t_record, "hmax": res.hmax.astype(np.float16), "vmax": res.vmax.astype(np.float16),
           "hrmax": res.hrmax.astype(np.float16), "t_wet": res.t_wet.astype(np.float32),
           "snaps": res.snaps, "t_snaps": res.t_snaps, "budget": json.dumps(res.budget),
           **{f"q_{k}": v for k, v in res.q_gauge.items()}, **{f"vol_{k}": v for k, v in res.vol_gauge.items()}}
    np.savez_compressed(f, **out)
    print(f"  {name}: {res.wall_s:.0f} s, {res.steps} steps, water-balance error {res.budget['error_frac']:.1e}",
          flush=True)
    return out


def summarise(S, R, hours0, truth=None):
    """Numbers for one run; with `truth`, also how far it is from the run on the true rain."""
    t = R["t"]
    out = {}
    for k in S["basins"]:
        q = R[f"q_{k}"]
        i = int(np.argmax(q))
        out[k] = {"peak_m3s": float(q[i]), "peak_time": str(hours0 + pd.Timedelta(seconds=float(t[i])))[:16],
                  "volume_Mm3": float(R[f"vol_{k}"]) / 1e6}
    hmax = R["hmax"].astype(np.float32)
    hcls = hazard.hazard_class(R["hrmax"].astype(np.float32), hmax)
    wet = hmax >= hazard.WET_M
    tw = R["t_wet"]
    times = np.arange(0, float(t[-1]) + 1, 60.0)
    zone_curve = hazard.wet_area_curve(tw, S["built_ha"], times, S["zone"])
    t_flood = hazard.first_crossing(times, zone_curve, config.ALARM_HA)
    risk = hazard.commune_risk(S["labels"], S["names"], hcls, S["built_ha"])
    out.update({
        "built_ha_wet": float((S["built_ha"] * wet).sum()),
        "built_ha_significant": float((S["built_ha"] * (hcls >= 3)).sum()),
        "zone_built_ha_wet": float(zone_curve[-1]),
        "zone_flood_time": None if np.isnan(t_flood) else str(hours0 + pd.Timedelta(seconds=t_flood))[:16],
        "zone_flood_s": None if np.isnan(t_flood) else t_flood,
        "budget": json.loads(str(R["budget"])),
        "communes_top": risk[:10],
    })
    if truth is not None:
        wt = truth["hmax"].astype(np.float32) >= hazard.WET_M
        out["csi_wet"] = hazard.csi(wet, wt)
        out["csi_wet_built"] = hazard.csi(wet & (S["g"].built > 0.2), wt & (S["g"].built > 0.2))
        rt = {r["commune"]: r["built_ha_at_risk"] for r in
              hazard.commune_risk(S["labels"], S["names"],
                                  hazard.hazard_class(truth["hrmax"].astype(np.float32), truth["hmax"].astype(np.float32)),
                                  S["built_ha"])}
        rs = {r["commune"]: r["built_ha_at_risk"] for r in risk}
        common = sorted(rt)
        out["commune_spearman"] = hazard.spearman([rt[c] for c in common], [rs.get(c, 0.0) for c in common])
        top_t = {c for c, _ in sorted(rt.items(), key=lambda kv: -kv[1])[:5]}
        top_s = {c for c, _ in sorted(rs.items(), key=lambda kv: -kv[1])[:5]}
        out["commune_top5_overlap"] = len(top_t & top_s)
        for k in S["basins"]:
            pt, ps = truth[f"q_{k}"].max(), R[f"q_{k}"].max()
            out[k]["peak_err"] = float(ps / pt - 1) if pt > 0 else None
            vt, vs = float(truth[f"vol_{k}"]), float(R[f"vol_{k}"])
            out[k]["volume_err"] = float(vs / vt - 1) if vt > 0 else None
            out[k]["q_r"] = float(np.corrcoef(truth[f"q_{k}"], R[f"q_{k}"])[0, 1])
    return out


def rain_thresholds(S, city, out_dir, force=False):
    """Flash-flood rain thresholds: the uniform rain over T hours that just floods the town (>= ALARM_HA built-up
    hectares under 0.3 m), found from made-up storms on the whole domain (antecedent moisture II)."""
    f = out_dir / "ffg.json"
    done = json.loads(f.read_text()) if f.exists() and not force else {}
    g = S["g"]
    shape = g.z.shape
    out = {}
    for T in city.ffg_hours:
        if str(T) in done:                     # already computed for this duration
            out[str(T)] = done[str(T)]
            continue
        depths = [10, 20, 30, 45, 60, 80, 110, 150]
        wet_ha = []
        for D in depths:
            fields = rain.design_storm(D, T, shape, hours_after=city.spin_dry_h)
            res = solver.simulate(S["z"], S["n"], S["cn"], g.dx, g.dy, solver.HourlyRain(fields),
                                  fields.shape[0] * 3600.0, sink=S["sea"], params=solver.Params(snap_s=1e12))
            ha = float((S["built_ha"] * (res.hmax >= hazard.WET_M) * S["zone"]).sum())
            wet_ha.append(ha)
            print(f"  threshold run: {D} mm in {T} h -> {ha:.1f} ha of town under 0.3 m ({res.wall_s:.0f} s)",
                  flush=True)
            if ha >= 3 * config.ALARM_HA:
                break
        wet_ha = np.array(wet_ha)
        dd = np.array(depths[:wet_ha.size], float)
        thr = None
        i = np.flatnonzero(wet_ha >= config.ALARM_HA)
        if i.size:
            i = i[0]
            thr = float(dd[0]) if i == 0 else float(np.interp(config.ALARM_HA, wet_ha[i - 1:i + 1], dd[i - 1:i + 1]))
        out[str(T)] = {"threshold_mm": thr, "depths_mm": dd.tolist(), "town_wet_ha": wet_ha.tolist()}
    f.write_text(json.dumps(out, indent=1))
    return out


def alarm_time(fields, areas, thresholds, hours0):
    """First time the mean rain over any of `areas` (the town; the basin), summed over any threshold duration,
    reaches that duration's threshold. Two areas because a town floods two ways: from rain on the town itself, and
    from the stream that drains the basin. Returns seconds since the start (end of that hour), or nan."""
    best = np.inf
    for area in areas:
        series = np.array([fields[h][area].mean() for h in range(fields.shape[0])])
        for T, th in thresholds.items():
            if th["threshold_mm"] is None:
                continue
            T = int(T)
            roll = np.convolve(series, np.ones(T), "full")[: series.size]
            i = np.flatnonzero(roll >= th["threshold_mm"])
            if i.size:
                best = min(best, (i[0] + 1) * 3600.0)
    return float(best) if np.isfinite(best) else float("nan")


def city_runs(key, force=False):
    city = config.CITIES[key]
    out_dir = config.PROC / key
    out_dir.mkdir(parents=True, exist_ok=True)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"== {city.name}", flush=True)
    S = setup(city)
    g = S["g"]
    p4 = rain.Phase4Rain(key)
    hours0 = p4.hours[0]
    bkey = city.outlets[0].key
    bmask = S["basins"][bkey]["mask"]
    fields, coverage = {}, {}
    for s in SOURCES:
        f, cov = p4.fields(s, g.lat, g.lon)
        fields[s] = np.concatenate([f, np.zeros((city.spin_dry_h, *f.shape[1:]), np.float32)])
        coverage[s] = cov
    basin_total = float(fields["truth"].sum(0)[bmask].mean())
    factor = city.gauge_total_mm / basin_total
    print(f"  basin {S['basins'][bkey]['area_km2']:.1f} km2, true storm total over it {basin_total:.1f} mm, "
          f"gauge-matched factor {factor:.2f}", flush=True)

    thresholds = rain_thresholds(S, city, out_dir, force)
    areas = (S["zone"] & (g.built > 0.2), bmask)      # the town's built-up land; the basin
    tbox = _tower_box(p4, g)
    summary = {"city": city.name, "dem": g.dem_source, "grid": [int(g.z.shape[0]), int(g.z.shape[1])],
               "cell_m": [round(g.dx, 1), round(g.dy, 1)],
               "basins": {k: {kk: b[kk] for kk in ("lat", "lon", "area_km2")} | {
                   "area_km2_ref": next(o.area_km2_ref for o in city.outlets if o.key == k),
                   "cn_mean": float(S["cn"][b["mask"]].mean()),
                   "built_share": float(g.built[b["mask"]].mean()),
                   "tower_box_share": float((b["mask"] & tbox).sum() / b["mask"].sum())}
                          for k, b in S["basins"].items()},
               "filled_pits_Mm3": float(S["filled"].sum() * S["cell"] / 1e6),
               "tower_box_share_domain": coverage["truth"], "basin_true_total_mm": basin_total,
               "gauge_total_mm": city.gauge_total_mm, "gauge_note": city.gauge_note, "gauge_factor": factor,
               "observed": city.observed, "thresholds": thresholds, "alarm_ha": config.ALARM_HA,
               "latency_min": config.LATENCY_MIN, "hours0": str(hours0), "scenarios": {}}
    for scen, k in (("satellite", 1.0), ("gauge", factor)):
        runs = {s: run(S, fields[s] * k, f"{scen}_{s}", out_dir, force, snaps=s in SNAP_SOURCES) for s in SOURCES}
        sc = {}
        tflood = None
        for s in SOURCES:
            sc[s] = summarise(S, runs[s], hours0, None if s == "truth" else runs["truth"])
            if s == "truth":
                tflood = sc[s]["zone_flood_s"]
            ta = alarm_time(fields[s] * k, areas, thresholds, hours0)
            lat_s = config.LATENCY_MIN[s] * 60.0
            sc[s]["alarm_time"] = None if np.isnan(ta) else str(hours0 + pd.Timedelta(seconds=ta + lat_s))[:16]
            sc[s]["alarm_lead_min"] = (None if (np.isnan(ta) or tflood is None) else (tflood - ta - lat_s) / 60.0)
            sf = sc[s]["zone_flood_s"]
            sc[s]["model_alarm_lead_min"] = (None if (sf is None or tflood is None) else
                                             (tflood - sf - lat_s) / 60.0)
        summary["scenarios"][scen] = sc
        print(f"  {scen}: " + ", ".join(f"{s} peak {sc[s][bkey]['peak_m3s']:.0f} m3/s" for s in SOURCES), flush=True)

    # sensitivity: the truth, gauge-matched, one choice changed at a time
    base = fields["truth"] * factor
    sens = {"wet_soil_amc3": dict(cn=runoff.curve_numbers(g.landcover, g.hsg, amc=3)),
            "dry_soil_amc1": dict(cn=runoff.curve_numbers(g.landcover, g.hsg, amc=1)),
            "terrain_unfilled": dict(z=S["z_raw"]),
            "built_n_0.02": dict(n=runoff.manning(g.landcover, built_n=0.02)),
            "built_n_0.10": dict(n=runoff.manning(g.landcover, built_n=0.10))}
    if key == "algiers":
        sens["drains_20mmh"] = dict(drain=20.0 * g.built)
    truth_g = run(S, base, "gauge_truth", out_dir)
    summary["sensitivity"] = {}
    for name, kw in sens.items():
        R = run(S, base, f"sens_{name}", out_dir, force, **kw)
        summary["sensitivity"][name] = summarise(S, R, hours0, truth_g)
    (config.RESULTS / f"{key}.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"  wrote {config.RESULTS / f'{key}.json'}", flush=True)


def _tower_box(p4, g):
    lat, lon = p4.maps["truth"].lat, p4.maps["truth"].lon
    s = abs(lat[1] - lat[0]) / 2
    return ((g.lat[:, None] >= lat.min() - s) & (g.lat[:, None] <= lat.max() + s)
            & (g.lon[None, :] >= lon.min() - s) & (g.lon[None, :] <= lon.max() + s))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for k in args or list(config.CITIES):
        city_runs(k, force="--force" in sys.argv)
