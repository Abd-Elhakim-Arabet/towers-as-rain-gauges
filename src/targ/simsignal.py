"""Phase 4: what a simulated link would record during a real storm.

For every simulated sub-link, minute by minute:
  total loss = real dry-weather recording of an Italian link   (noise, drift, gaps,
                                                                 1-dB rounding, faults)
             + rain loss along the path (ITU-R P.838, integrated point by point)
             + wet-antenna loss (builds up in rain, dries slowly)
             capped where the received level would drop below -100 dBm (receiver floor),
             then rounded to the recording's 1-dB step.

Why real recordings: if the noise were invented with the same simple formulas the
AI and the classical method assume, the test would be too easy. The recordings
come from the eastern half of the Italian network in 2022 (dry windows only: the
radar shows rain less than 1% of the time), which the models used here never saw:
they were trained on the western half (Section 18.6 of the explainer).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import map_coordinates

from targ import features as feat

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SignalParams:
    step_km: float = 0.1             # path sampling
    wa_max_db: tuple = (1.0, 2.5)    # wet-antenna loss when soaked, both antennas (Italy: 1.5-2 dB)
    wa_rate_mm_h: float = 1.0        # rain rate at which ~63% of the maximum is reached
    wa_rise_min: float = 3.0         # wetting time
    wa_dry_min: float = 45.0         # drying time after rain stops
    seed: int = 0


def sub_links(links: pd.DataFrame) -> pd.DataFrame:
    """Two sub-links (directions) per link, like the real data."""
    a = links.assign(cml_id=links.link_id + "_1")
    b = links.assign(cml_id=links.link_id + "_2",
                     site_0_lon=links.site_1_lon, site_0_lat=links.site_1_lat,
                     site_1_lon=links.site_0_lon, site_1_lat=links.site_0_lat)
    return pd.concat([a, b]).sort_values("cml_id").reset_index(drop=True)


def path_rain(fine: xr.DataArray, sl: pd.DataFrame, p: SignalParams = SignalParams()):
    """Rain loss [dB] and true path-mean rain [mm/h], shape (n_sub, T)."""
    a, b = feat.itu_a_b(sl.freq_ghz.values, sl.pol.values)
    lat, lon = fine.lat.values, fine.lon.values
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    pts_r, pts_c, owner = [], [], []
    for i, r in enumerate(sl.itertuples()):
        n = max(2, int(np.ceil(r.length_km / p.step_km)) + 1)
        f = (np.arange(n) + 0.5) / n
        pts_r.append(((r.site_0_lat + f * (r.site_1_lat - r.site_0_lat)) - lat[0]) / dlat)
        pts_c.append(((r.site_0_lon + f * (r.site_1_lon - r.site_0_lon)) - lon[0]) / dlon)
        owner.append(np.full(n, i))
    pr, pc, owner = np.concatenate(pts_r), np.concatenate(pts_c), np.concatenate(owner)
    starts = np.r_[0, np.flatnonzero(np.diff(owner)) + 1]
    counts = np.diff(np.r_[starts, len(owner)])
    ap, bp = a[owner], b[owner]
    v = fine.values
    T = v.shape[0]
    A = np.empty((len(sl), T), np.float32)
    Rm = np.empty((len(sl), T), np.float32)
    for t in range(T):
        R = np.clip(map_coordinates(v[t], [pr, pc], order=1, mode="nearest"), 0, None)
        Rm[:, t] = np.add.reduceat(R, starts) / counts
        A[:, t] = np.add.reduceat(ap * R ** bp, starts) / counts * sl.length_km.values
    return A, Rm


def wet_antenna(Rm: np.ndarray, p: SignalParams = SignalParams()) -> np.ndarray:
    rng = np.random.default_rng(p.seed + 1)
    wmax = rng.uniform(*p.wa_max_db, size=(Rm.shape[0],)).astype(np.float32)
    target = wmax[:, None] * (1 - np.exp(-Rm / p.wa_rate_mm_h))
    W = np.zeros_like(Rm)
    up, down = 1 - np.exp(-1 / p.wa_rise_min), 1 - np.exp(-1 / p.wa_dry_min)
    w = np.zeros(Rm.shape[0], np.float32)
    for t in range(Rm.shape[1]):
        g = target[:, t]
        w = w + np.where(g > w, up, down) * (g - w)
        W[:, t] = w
    return W


def donor_bank(n_min: int, max_windows: int = 800, seed: int = 0):
    """Dry-weather recordings (tsl, tl) of length n_min from eastern Italian sub-links, 2022."""
    cache = ROOT / "data" / "interim" / "p4" / f"donors_{n_min}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["tsl"], z["tl"], z["src"]
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import phase2_mixed as M  # noqa: E402
    from targ.classical import total_loss
    from targ.data import openrainer
    months = [f"2022{m:02d}" for m in range(5, 10)]
    ds = openrainer.cml(months)
    ds = ds.isel(cml_id=M.side_index(ds, "east"))
    tl = total_loss(ds, 5).values.astype(np.float32)
    tsl = ds.tsl.ffill("time", limit=5).transpose("cml_id", "time").values.astype(np.float32)
    radar = openrainer.radadj_along(ds, months)
    times = ds.time.values
    # radar rain per minute (15-min values spread over their 15 minutes)
    rpos = np.searchsorted(times, radar.time.values)
    wet_min = np.zeros(tl.shape, bool)
    rv = (radar.values >= 0.1)
    for k in range(1, 16):
        q = rpos - k
        ok = (q >= 0) & (q < len(times))
        wet_min[:, q[ok]] |= rv[:, ok]
    rng = np.random.default_rng(seed)
    out_tsl, out_tl, src = [], [], []
    starts = np.arange(0, tl.shape[1] - n_min, 1440)
    cand = [(i, s) for i in range(tl.shape[0]) for s in starts]
    rng.shuffle(cand)
    for i, s in cand:
        w = slice(s, s + n_min)
        if wet_min[i, w].mean() > 0.01 or np.isfinite(tl[i, w]).mean() < 0.9:
            continue
        out_tsl.append(tsl[i, w])
        out_tl.append(tl[i, w])
        src.append(f"{ds.cml_id.values[i]}@{str(times[s])[:10]}")
        if len(src) >= max_windows:
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, tsl=np.array(out_tsl), tl=np.array(out_tl), src=np.array(src))
    return np.array(out_tsl), np.array(out_tl), np.array(src)


def simulate(A: np.ndarray, W: np.ndarray, donors, seed: int = 0, rsl_floor_dbm: float = -100.0):
    """Total loss and TSL for each sub-link: donor recording + rain + wet antenna, 1-dB steps.
    The received level cannot drop below the receiver's floor (the Italian recordings bottom
    out at -100 to -103 dBm): in extreme rain the measured loss stops growing."""
    d_tsl, d_tl, d_src = donors
    rng = np.random.default_rng(seed + 2)
    pick = rng.integers(0, len(d_tl), size=A.shape[0])
    tsl = d_tsl[pick]
    tl = d_tl[pick] + A + W
    tl = np.minimum(tl, tsl - rsl_floor_dbm)   # rsl = tsl - tl >= floor
    tl = np.round(tl)                          # recordings are in whole dB
    return tl.astype(np.float32), tsl, d_src[pick]
