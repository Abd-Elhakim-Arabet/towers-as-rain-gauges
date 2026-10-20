"""Tests for the Phase 4 simulation: network, rain downscaling, link signals."""

import numpy as np
import pandas as pd
import xarray as xr

from targ import features as feat
from targ import simnet, simrain, simsignal


def _sites(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"lat": 36.7 + rng.uniform(0, 0.15, n), "lon": 3.0 + rng.uniform(0, 0.2, n),
                         "operator": rng.choice(["A", "B"], n)})


def test_network_is_a_tree_per_operator_with_plausible_hops(monkeypatch):
    pool = pd.DataFrame({"country": ["sweden"] * 50 + ["netherlands"] * 50,
                         "length_km": np.linspace(0.5, 10, 100), "freq_ghz": np.linspace(38, 15, 100), "pol": "V"})
    monkeypatch.setattr(simnet, "real_link_pool", lambda: pool)
    s = _sites()
    L = simnet.build(s, simnet.NetParams(fibre_share=0.2))
    for op, g in L.groupby("operator"):
        ends = set(zip(g.site_0_lat, g.site_0_lon))
        assert len(ends) == len(g)                    # each site has at most one uplink: no cycles
    assert L.length_km.between(0.3, 20).all()
    # longer hops get lower frequencies, as in the pool
    assert np.corrcoef(L.length_km, L.freq_ghz)[0, 1] < 0


def test_downscaling_keeps_the_satellite_amounts():
    rng = np.random.default_rng(1)
    t = pd.date_range("2020-01-01", periods=6, freq="30min")
    lat = np.round(np.arange(36.0, 37.01, 0.1), 2)
    lon = np.round(np.arange(2.5, 3.51, 0.1), 2)
    v = rng.gamma(0.8, 3.0, (6, len(lat), len(lon))).astype(np.float32)
    da = xr.DataArray(v, dims=("time", "lat", "lon"), coords={"time": t, "lat": lat, "lon": lon})
    fine, V = simrain.downscale(da, (36.3, 2.8, 36.7, 3.2), simrain.TextureParams(wet_threshold=0.0))
    assert fine.shape[0] == 180
    hh = fine.values.reshape(6, 30, *fine.shape[1:]).mean(1)
    # IMERG cells fully inside the fine domain: their half-hour means are kept
    errs = []
    for la in (36.4, 36.5, 36.6):
        for lo in (2.9, 3.0, 3.1):
            fl, fo = fine.lat.values, fine.lon.values      # a cell spans [centre - 0.05, centre + 0.05)
            m = ((fl >= la - 0.05 - 1e-6) & (fl < la + 0.05 - 1e-6))[:, None] & \
                ((fo >= lo - 0.05 - 1e-6) & (fo < lo + 0.05 - 1e-6))[None]
            sim = hh[:, m].mean(1)
            errs.append(np.abs(sim / da.sel(lat=la, lon=lo).values - 1))
    assert np.max(errs) < 1e-3


def test_uniform_rain_gives_the_textbook_loss():
    t = pd.date_range("2020-01-01", periods=3, freq="1min")
    lat, lon = np.round(np.arange(36.0, 36.3, 0.01), 3), np.round(np.arange(3.0, 3.3, 0.01), 3)
    fine = xr.DataArray(np.full((3, len(lat), len(lon)), 10.0, np.float32), dims=("time", "lat", "lon"),
                        coords={"time": t, "lat": lat, "lon": lon})
    sl = pd.DataFrame({"site_0_lat": [36.05], "site_0_lon": [3.05], "site_1_lat": [36.2], "site_1_lon": [3.2],
                       "length_km": [20.0], "freq_ghz": [23.0], "pol": ["V"]})
    A, R = simsignal.path_rain(fine, sl)
    a, b = feat.itu_a_b(np.array([23.0]), np.array(["V"]))
    assert np.allclose(R, 10.0) and np.allclose(A, a[0] * 10.0 ** b[0] * 20.0, rtol=1e-4)


def test_wet_antenna_builds_up_and_dries_slowly():
    R = np.zeros((1, 300), np.float32)
    R[0, 10:100] = 5.0
    W = simsignal.wet_antenna(R)
    assert W[0, 99] > 0.9 and W[0, 9] == 0                       # soaked during rain
    assert 0.1 < W[0, 140] / W[0, 99] < 0.5                       # about half gone 40 min later
