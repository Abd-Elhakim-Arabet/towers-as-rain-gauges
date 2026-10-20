"""Phase 4: the whole chain on a simulated Algerian network (SIMULATION, not validation).

For each city (Algiers 2001, Ghardaia 2008):
  1. storm: IMERG (scripts/phase4_storms.py) -> 1-min, 1-km rain   (targ.simrain)
  2. network: real OpenCelliD sites -> plausible microwave links   (targ.simnet)
  3. signals: real dry recordings + rain loss + wet antennas        (targ.simsignal)
  4. chain: reliability score -> AI (and classical) -> rain per link -> IDW maps
  5. scores against the simulated truth, per link and per map pixel, by distance
     to the nearest link and by network density (10/25/50/100% of links)

AI = the five Phase 2 runs of the final recipe trained on the WESTERN half of Italy
(mix21rel+mfb+mask+gt+west), averaged: the dry recordings come from the eastern
half, so the AI has never seen the noise it faces here.

    python scripts/phase4_simulate.py [algiers|ghardaia ...]
Writes results/phase4_<city>.json, results/phase4_links_<city>.csv and
data/processed/phase4/sim_<city>.npz (hourly fields for the figures).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from phase4_storms import EVENTS  # noqa: E402

from targ import features as feat  # noqa: E402
from targ import maps, simnet, simrain, simsignal  # noqa: E402
from targ import reliability as rel  # noqa: E402
from targ import train_torch as T  # noqa: E402
from targ.classical import ClassicalParams, retrieve  # noqa: E402
from targ.models.tcn_torch import TCN  # noqa: E402

OUT = ROOT / "data" / "processed" / "phase4"
AI_CONFIG = "mix21rel+mfb+mask+gt+west_mse"
EVAL_START = {"algiers": "2001-11-09T00:00", "ghardaia": "2008-09-29T00:00"}
MARGIN = 0.10          # deg around the city box for the rain field (paths stay inside)
FRACTIONS = (0.01, 0.03, 0.10, 0.25, 0.50, 1.00)


def ai_ensemble(dyn, stat, phys):
    kw = json.loads((ROOT / "results" / "phase2_openmrg.json").read_text())["variants"]["physics"]["model_kw"]
    res = json.loads((ROOT / "results" / f"phase2_mixed_{AI_CONFIG}.json").read_text())
    Ps, Rs = [], []
    for seed in range(5):
        m = TCN(dyn.shape[1], stat.shape[1], **kw).to(T.device())
        m.load_state_dict(torch.load(ROOT / "runs" / "phase2" / "mixed" / f"{AI_CONFIG}_seed{seed}.pt",
                                     map_location=T.device()))
        m.eval()
        P, R = T.predict(m, dyn, stat, phys)
        Ps.append(P)
        Rs.append(R)
    thr = res["ensemble"]["threshold"]
    P, R = np.mean(Ps, 0), np.mean(Rs, 0)
    return np.where(P > thr, R, 0.0), thr


def hourly(x, times):
    """(n, T) per-minute -> (n, H) hourly means (NaN-aware), with the hour starts."""
    df = pd.DataFrame(np.asarray(x, float).T, index=pd.DatetimeIndex(times))
    h = df.resample("1h").mean()
    return h.values.T, h.index.values


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 and np.std(a[m]) > 0 and np.std(b[m]) > 0 else np.nan


def run(city: str, fibre_share: float = 0.30):
    ev = EVENTS[city]
    print(f"===== {city}: {ev['name']}", flush=True)
    imerg = xr.open_dataarray(OUT / f"storm_{city}.nc")
    lat0, lon0, lat1, lon1 = ev["city"]
    dom = (lat0 - MARGIN, lon0 - MARGIN, lat1 + MARGIN, lon1 + MARGIN)

    # 1. storm
    fine, V = simrain.downscale(imerg, dom)
    times = fine.time.values
    texture = simrain.texture_stats(fine)
    print("rain field", fine.shape, "max", float(fine.max()), "texture", texture, flush=True)

    # 2. network
    sites = pd.read_csv(ROOT / "data" / "processed" / "algeria_sites.csv")
    sites = sites[(sites.lat >= lat0) & (sites.lat <= lat1) & (sites.lon >= lon0) & (sites.lon <= lon1)]
    links = simnet.build(sites, simnet.NetParams(fibre_share=fibre_share))
    links.to_csv(ROOT / "results" / f"phase4_links_{city}.csv", index=False)
    sl = simsignal.sub_links(links)
    print(f"{len(sites)} sites -> {len(links)} links, {len(sl)} sub-links", flush=True)

    # 3. signals
    A, Rtrue = simsignal.path_rain(fine, sl)
    W = simsignal.wet_antenna(Rtrue)
    d_tsl, d_tl, d_src = simsignal.donor_bank(8640)
    n = len(times)
    tl, tsl, src = simsignal.simulate(A, W, (d_tsl[:, :n], d_tl[:, :n], d_src))
    pol = np.where(sl.pol.values == "H", "horizontal", "vertical")
    ds = xr.Dataset({"tsl": (("cml_id", "time"), tsl), "rsl": (("cml_id", "time"), tsl - tl)},
                    coords={"cml_id": sl.cml_id.values, "time": times, "length": ("cml_id", sl.length_km.values),
                            "frequency": ("cml_id", sl.freq_ghz.values), "polarization": ("cml_id", pol),
                            "site_0_lat": ("cml_id", sl.site_0_lat.values), "site_0_lon": ("cml_id", sl.site_0_lon.values),
                            "site_1_lat": ("cml_id", sl.site_1_lat.values), "site_1_lon": ("cml_id", sl.site_1_lon.values)})

    # 4. chain
    keep = rel.score_links(ds, tl_1min=tl)["keep"]
    print(f"reliability score keeps {int(keep.sum())} of {len(keep)} sub-links", flush=True)
    L, f, pl = sl.length_km.values, sl.freq_ghz.values, pol
    dyn = feat.dynamic_features(tl, L, f, pl)
    R_ai, thr = ai_ensemble(dyn, feat.static_features(L, f, pl), feat.physics_params(L, f, pl))
    R_ai = np.where(np.isfinite(tl), R_ai, np.nan)
    cp = ClassicalParams(**json.loads((ROOT / "results" / "phase1_openmrg.json").read_text())["params"])
    R_cl = retrieve(ds, cp).R.values

    # 5. scores (storm period only; the first 3 days are the dry-level spin-up)
    ev_m = times >= np.datetime64(EVAL_START[city])
    Th, ht = hourly(Rtrue[:, ev_m], times[ev_m])
    res = {"city": city, "event": ev["name"], "fibre_share": fibre_share, "n_sites": int(len(sites)),
           "n_links": int(len(links)), "n_sub_links": int(len(sl)), "n_reliable": int(keep.sum()),
           "link_length_km_pct": np.percentile(links.length_km, [5, 25, 50, 75, 95]).round(2).tolist(),
           "ai_threshold": thr, "texture_check": texture, "motion_km_h_median": (np.median(np.abs(V), 0) * 60 * 111).round(1).tolist(),
           "links": {}, "maps": {}}
    est = {}
    for name, R in (("ai", R_ai), ("classical", R_cl)):
        Rh, _ = hourly(R[:, ev_m], times[ev_m])
        est[name] = Rh
        k = keep
        tot_t, tot_e = np.nansum(Th[k]), np.nansum(np.where(np.isfinite(Rh[k]), Rh[k], 0))
        res["links"][name] = {"hourly_r": corr(Rh[k].ravel(), Th[k].ravel()),
                              "bias_pct": float(100 * (tot_e / tot_t - 1)),
                              "rmse_mm_h": float(np.sqrt(np.nanmean((Rh[k] - Th[k]) ** 2)))}
        print(name, "links", res["links"][name], flush=True)

    # truth on the city grid, hourly
    fine_c = fine.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    tv = fine_c.values[ev_m]
    H = len(ht)
    truth = tv[: H * 60].reshape(H, 60, *tv.shape[1:]).mean(1)                 # (H, ny, nx)
    glat, glon = np.meshgrid(fine_c.lat.values, fine_c.lon.values, indexing="ij")
    ref_lat, ref_lon = (lat0 + lat1) / 2, (lon0 + lon1) / 2
    dst = np.column_stack(maps.to_km(glat.ravel(), glon.ravel(), ref_lat, ref_lon))
    rng = np.random.default_rng(0)
    phys_ids = np.unique(links.link_id.values)
    saved = {}
    for frac in FRACTIONS:
        chosen = set(rng.choice(phys_ids, size=max(1, int(round(frac * len(phys_ids)))), replace=False))
        use = keep & sl.link_id.isin(chosen).values
        res["maps"][f"{frac:.2f}"] = {"n_links": len(chosen)}
        for name, Rh in est.items():
            Rp, mlat, mlon = maps.merge_directions(Rh[use], sl.site_0_lat.values[use], sl.site_0_lon.values[use],
                                                    sl.site_1_lat.values[use], sl.site_1_lon.values[use])
            src_xy = np.column_stack(maps.to_km(mlat, mlon, ref_lat, ref_lon))
            Wt = maps.idw_weights(src_xy, dst)
            M = maps.apply_idw(Wt, Rp).T.reshape(H, *truth.shape[1:])            # (H, ny, nx)
            from scipy.spatial import cKDTree
            dist = cKDTree(src_xy).query(dst)[0].reshape(truth.shape[1:])
            cover = np.isfinite(M).all(0)
            tot_t, tot_m = truth.sum(0), np.nansum(M, 0)
            city_t, city_m = truth.mean((1, 2)), np.nanmean(M, (1, 2))
            by_d = {}
            for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 10)):
                mm = (dist >= lo) & (dist < hi) & cover
                by_d[f"{lo}-{hi}km"] = {"share_of_city": float(((dist >= lo) & (dist < hi)).mean()),
                                        "hourly_r": corr(M[:, mm].ravel(), truth[:, mm].ravel()),
                                        "total_bias_pct": float(100 * (tot_m[mm].sum() / max(tot_t[mm].sum(), 1e-9) - 1))
                                        if mm.any() else None}
            res["maps"][f"{frac:.2f}"][name] = {
                "covered_share": float(cover.mean()),
                "pixel_hourly_r": corr(M[:, cover].ravel(), truth[:, cover].ravel()),
                "pixel_hourly_rmse": float(np.sqrt(np.nanmean((M[:, cover] - truth[:, cover]) ** 2))),
                "event_total_r": corr(tot_m[cover], tot_t[cover]),
                "event_total_bias_pct": float(100 * (tot_m[cover].sum() / tot_t[cover].sum() - 1)),
                "city_mean_hourly_r": corr(city_m, city_t),
                "city_total_truth_mm": float(city_t.sum()), "city_total_est_mm": float(np.nansum(city_m)),
                "by_distance": by_d}
            print(f"{frac:.2f} {name}", {k: (round(v, 3) if isinstance(v, float) else v)
                                          for k, v in res["maps"][f"{frac:.2f}"][name].items() if k != "by_distance"},
                  flush=True)
            if frac == 1.0:
                saved[name] = M.astype(np.float32)
                saved["dist_km"] = dist.astype(np.float32)
            if frac == 0.03:
                saved[name + "_sparse"] = M.astype(np.float32)
                saved["dist_km_sparse"] = dist.astype(np.float32)
                saved["sparse_ids"] = np.array(sorted(chosen))
    # 6. keep what the figures need
    imerg_city = imerg.sel(time=slice(times[ev_m][0], None))
    np.savez_compressed(
        OUT / f"sim_{city}.npz", truth=truth.astype(np.float32), ai=saved["ai"], classical=saved["classical"],
        dist_km=saved["dist_km"], ai_sparse=saved["ai_sparse"], classical_sparse=saved["classical_sparse"],
        dist_km_sparse=saved["dist_km_sparse"], sparse_ids=saved["sparse_ids"], lat=fine_c.lat.values, lon=fine_c.lon.values, hours=ht,
        imerg=imerg_city.values.astype(np.float32), imerg_lat=imerg.lat.values, imerg_lon=imerg.lon.values,
        imerg_time=imerg_city.time.values, keep=keep, link_true_h=Th.astype(np.float32),
        link_ai_h=est["ai"].astype(np.float32), link_cl_h=est["classical"].astype(np.float32),
        tl_example=tl[:, ev_m][:, ::1].astype(np.float32)[np.argsort(-np.nansum(Rtrue, 1))[:6]],
        a_example=A[:, ev_m][np.argsort(-np.nansum(Rtrue, 1))[:6]],
        w_example=W[:, ev_m][np.argsort(-np.nansum(Rtrue, 1))[:6]],
        rtrue_example=Rtrue[:, ev_m][np.argsort(-np.nansum(Rtrue, 1))[:6]],
        rai_example=R_ai[:, ev_m][np.argsort(-np.nansum(Rtrue, 1))[:6]],
        example_ids=sl.cml_id.values[np.argsort(-np.nansum(Rtrue, 1))[:6]],
        donor_src=src)
    (ROOT / "results" / f"phase4_{city}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["algiers", "ghardaia"]):
        run(c)
