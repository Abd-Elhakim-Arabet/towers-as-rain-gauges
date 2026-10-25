"""Phase 5 (floodrisk): solver, losses and drainage on cases with known answers."""

import numpy as np
import pytest

from floodrisk import runoff, solver, terrain


def P(**kw):
    return solver.Params(device="cpu", snap_s=1e9, **kw)


def test_scs_textbook():
    # TR-55: P = 100 mm, CN = 80 -> S = 63.5, Ia = 12.7, Q = 87.3^2 / 150.8 = 50.5 mm
    assert runoff.scs_runoff(np.array([100.0]), np.array([80.0]))[0] == pytest.approx(50.5, abs=0.1)
    assert runoff.scs_runoff(np.array([10.0]), np.array([80.0]))[0] == 0.0          # below the initial loss


def test_lake_at_rest():
    y, x = np.mgrid[0:30, 0:30]
    z = 0.01 * ((x - 15) ** 2 + (y - 15) ** 2)                 # a bowl
    h0 = np.clip(1.0 - z, 0, None)                             # flat water surface at 1 m
    rain = solver.HourlyRain(np.zeros((1, 30, 30)))
    r = solver.simulate(z, np.full(z.shape, 0.03), np.full(z.shape, 100.0), 10, 10, rain, 600, h0=h0, params=P())
    assert r.vmax.max() < 1e-3
    assert abs(r.budget["error_m3"]) < 1e-3 * h0.sum() * 100


def test_mass_conserved():
    rng = np.random.default_rng(0)
    z = np.cumsum(rng.normal(0, 0.3, (40, 50)), 1) + 0.05 * np.arange(40)[:, None]
    rain = solver.HourlyRain(rng.uniform(5, 60, (2, 40, 50)))
    cn = rng.uniform(60, 100, z.shape)
    r = solver.simulate(z, np.full(z.shape, 0.05), cn, 20, 25, rain, 3 * 3600, params=P())
    assert r.budget["runoff_m3"] > 0
    assert abs(r.budget["error_frac"]) < 1e-3


def test_plane_outflow_equals_rain():
    # steady rain on a 1% slope draining east: at equilibrium, flow out of the slope = rain x area
    nx, ny, dx = 40, 10, 10.0
    z = np.tile(0.01 * dx * (nx - np.arange(nx)), (ny, 1)).astype(float)
    mask = np.zeros((ny, nx), bool)
    mask[:, :-1] = True
    rain = solver.HourlyRain(np.full((4, ny, nx), 36.0))         # 36 mm/h = 1e-5 m/s
    r = solver.simulate(z, np.full(z.shape, 0.03), np.full(z.shape, 100.0), dx, dx, rain, 4 * 3600,
                        gauges={"plane": mask}, params=P())
    expect = 1e-5 * mask.sum() * dx * dx
    assert r.q_gauge["plane"][-1] == pytest.approx(expect, rel=0.02)


def test_valley_drainage():
    y, x = np.mgrid[0:21, 0:31]
    z = np.abs(x - 15) * 1.0 + (20 - y) * 0.5                  # V valley draining south
    zf = terrain.fill(z)
    d = terrain.d8(zf, 10, 10)
    acc = terrain.accumulate(zf, d)
    r, c = terrain.snap(acc, 20, 15, 2)
    assert (r, c) == (20, 15)
    m = terrain.basin(zf, d, r, c)
    assert m.sum() > 0.8 * z.size                               # almost everything drains through the outlet
