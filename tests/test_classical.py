import numpy as np
import pycomlink as pycml
import xarray as xr

from targ.classical import ClassicalParams, retrieve, rolling_std


def test_rolling_std_is_centred():
    x = np.zeros((1, 200))
    x[0, 100] = 10.0  # single spike
    s = rolling_std(x, window=20)
    nz = np.flatnonzero(s[0] > 0)
    # a centred window sees the spike from ~10 samples before to ~10 after
    assert 88 <= nz.min() <= 92 and 108 <= nz.max() <= 111


def _synthetic(R_true=10.0, f=38.0, L=2.0, n=600):
    a, b = pycml.processing.k_R_relation.a_b(np.array([f]), pol=np.array(["vertical"]),
                                             approx_type="ITU_2005")
    A = a[0] * R_true ** b[0] * L
    rng = np.random.default_rng(0)
    rsl = -45.0 + rng.normal(0, 0.05, n)
    rsl[300:420] -= A + rng.normal(0, 1.0, 120)  # 2-h rain event with fluctuations
    t = np.datetime64("2015-08-01T00:00") + np.arange(n).astype("timedelta64[m]")
    return xr.Dataset(
        {"rsl": (("cml_id", "time"), rsl[None]), "tsl": (("cml_id", "time"), np.full((1, n), 10.0))},
        coords={"cml_id": [1], "time": t, "length": ("cml_id", [L]),
                "frequency": ("cml_id", [f]), "polarization": ("cml_id", ["vertical"])}), A


def test_retrieves_known_rain_rate_without_wet_antenna():
    ds, A = _synthetic()
    out = retrieve(ds, ClassicalParams(window_min=30, threshold_db=0.3, waa="none"))
    R_event = np.nanmean(out.R.values[0, 320:400])
    assert abs(R_event - 10.0) / 10.0 < 0.15
    assert np.nanmax(out.R.values[0, :250]) == 0.0  # dry before the event


def test_wet_antenna_reduces_rain():
    ds, _ = _synthetic()
    none = retrieve(ds, ClassicalParams(window_min=30, threshold_db=0.3, waa="none")).R
    pst = retrieve(ds, ClassicalParams(window_min=30, threshold_db=0.3, waa="pastorek")).R
    assert np.nansum(pst) < np.nansum(none)
