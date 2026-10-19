import numpy as np

from targ import maps


def test_idw_basic():
    src = np.array([[0.0, 0.0], [2.0, 0.0], [50.0, 50.0]])
    dst = np.array([[1.0, 0.0], [0.0, 0.0], [30.0, 0.0]])
    W = maps.idw_weights(src, dst, max_km=10)
    v = np.array([[2.0, np.nan], [4.0, 4.0], [100.0, 100.0]])
    out = maps.apply_idw(W, v)
    assert np.isclose(out[0, 0], 3.0)           # halfway between 2 and 4
    assert np.isclose(out[0, 1], 4.0)           # missing link skipped
    assert out[1, 0] < 2.5                      # close to the first link
    assert np.isnan(out[2]).all()               # nothing within 10 km


def test_merge_directions():
    R = np.array([[1.0, 3.0], [3.0, np.nan], [5.0, 5.0]])
    lat0, lon0 = np.array([45, 45.1, 46.0]), np.array([9, 9.1, 9.0])
    lat1, lon1 = np.array([45.1, 45, 46.1]), np.array([9.1, 9, 9.1])
    Rp, mlat, mlon = maps.merge_directions(R, lat0, lon0, lat1, lon1)
    assert Rp.shape == (2, 2)
    i = int(np.argmin(np.abs(mlat - 45.05)))
    assert np.allclose(Rp[i], [2.0, 3.0])
