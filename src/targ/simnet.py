"""Phase 4: a simulated microwave-link network on real Algerian tower positions.

We know where towers are (OpenCelliD) but not which towers are joined by
microwave links. Real mobile backhaul is mostly a tree: a few sites have fibre
("hubs"); every other site sends its traffic over one microwave hop to a site
nearer the hub, and so on. We build that tree per operator:

1. Hubs: a share of the sites (default 30%) spread over the city (k-means
   centres, snapped to the nearest site).
2. Sites are connected in order of distance to the nearest hub. Each picks, among
   already-connected sites of the same operator, the one whose distance best
   matches a hop length drawn from real city networks (Gothenburg and the
   Netherlands). The result is a tree rooted at the hubs, and hop lengths look like
   real ones instead of always being "nearest neighbour" (OpenCelliD sites are
   dense: ~0.5 km apart for one operator in central Algiers, which would give
   unrealistically short links).
3. Frequency and polarisation are copied from a real link of similar length
   (pooled from Sweden, Italy and the Netherlands): longer hops get lower
   frequencies, as in real networks.
Each link carries two sub-links (one per direction), as in the real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POOL_FILE = ROOT / "results" / "phase4_link_pool.csv"
EARTH_R = 6_371_000.0


def real_link_pool() -> pd.DataFrame:
    """Length [km], frequency [GHz], polarisation of real links (one row per link)."""
    if POOL_FILE.exists():
        return pd.read_csv(POOL_FILE)
    from targ.data import netherlands, openmrg, openrainer
    rows = []
    m = openmrg.metadata().drop_duplicates("Link")
    rows.append(pd.DataFrame({"country": "sweden", "length_km": m.Length_km.values,
                              "freq_ghz": m.Frequency_GHz.values,
                              "pol": [p[0].upper() for p in m.Polarization.values]}))
    d = openrainer.cml(["202106"])
    link = pd.Series(d.cml_id.values).str.split("_").str[0]
    it = pd.DataFrame({"link": link, "length_km": d.length.values, "freq_ghz": d.frequency.values,
                       "pol": [str(p)[0].upper() for p in d.polarization.values]}).drop_duplicates("link")
    rows.append(it.drop(columns="link").assign(country="italy"))
    n = netherlands.load_cml()
    rows.append(pd.DataFrame({"country": "netherlands", "length_km": n.length.values,
                              "freq_ghz": n.frequency.values, "pol": "V"}))
    pool = pd.concat(rows, ignore_index=True)
    pool.to_csv(POOL_FILE, index=False)
    return pool


def xy_km(lat, lon, lat0, lon0):
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R * np.cos(np.radians(lat0)) / 1000
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R / 1000
    return x, y


@dataclass
class NetParams:
    fibre_share: float = 0.30          # share of sites that are hubs (no microwave uplink)
    min_hop_km: float = 0.3            # shorter hops are not microwave in practice (cable)
    max_hop_km: float = 20.0
    length_pool: tuple = ("sweden", "netherlands")   # city-like networks for hop lengths
    seed: int = 0


def build(sites: pd.DataFrame, p: NetParams = NetParams()) -> pd.DataFrame:
    """sites: lon, lat, operator. Returns one row per link with both ends, length, freq, pol."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(p.seed)
    pool = real_link_pool()
    lengths = pool[pool.country.isin(p.length_pool)].length_km.values
    lengths = lengths[(lengths >= p.min_hop_km) & (lengths <= p.max_hop_km)]
    pool_sorted = pool.sort_values("length_km").reset_index(drop=True)
    lat0, lon0 = float(sites.lat.mean()), float(sites.lon.mean())
    links = []
    for op, g in sites.groupby("operator"):
        g = g.reset_index(drop=True)
        x, y = xy_km(g.lat.values, g.lon.values, lat0, lon0)
        P = np.c_[x, y]
        n = len(g)
        k = max(1, int(round(p.fibre_share * n)))
        km = KMeans(n_clusters=k, n_init=3, random_state=p.seed).fit(P)
        hubs = np.unique([int(np.argmin(((P - c) ** 2).sum(1))) for c in km.cluster_centers_])
        D_hub = np.sqrt(((P[:, None, :] - P[None, hubs, :]) ** 2).sum(-1))
        home = hubs[np.argmin(D_hub, axis=1)]
        d_home = D_hub.min(axis=1)
        connected = np.zeros(n, bool)
        connected[hubs] = True
        for s in np.argsort(d_home):
            if connected[s]:
                continue
            cand = np.where(connected)[0]
            d = np.sqrt(((P[cand] - P[s]) ** 2).sum(1))
            ok = (d >= p.min_hop_km) & (d <= p.max_hop_km)
            if not ok.any():
                continue                     # no plausible microwave hop: treated as cabled
            cand, d = cand[ok], d[ok]
            target = rng.choice(lengths)
            j = int(np.argmin(np.abs(d - target)))
            t = cand[j]
            connected[s] = True
            # frequency / polarisation from one of the 20 real links closest in length
            i = int(np.searchsorted(pool_sorted.length_km.values, d[j]))
            lo, hi = max(0, i - 10), min(len(pool_sorted), i + 10)
            ref = pool_sorted.iloc[rng.integers(lo, hi)]
            links.append({"operator": op, "site_0_lon": g.lon[s], "site_0_lat": g.lat[s],
                          "site_1_lon": g.lon[t], "site_1_lat": g.lat[t], "length_km": float(d[j]),
                          "freq_ghz": float(ref.freq_ghz), "pol": ref.pol, "ref_country": ref.country})
    out = pd.DataFrame(links)
    out.insert(0, "link_id", [f"dz{i:05d}" for i in range(len(out))])
    return out
