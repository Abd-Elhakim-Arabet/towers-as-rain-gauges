import numpy as np
import torch

from targ import features as feat
from targ.models.tcn_torch import TCN


def test_tcn_is_causal():
    """Changing the future must not change any earlier output."""
    torch.manual_seed(0)
    m = TCN(4, 3, head="physics").eval()
    x = torch.randn(2, 4, 1500)
    stat, phys = torch.randn(2, 3), torch.tensor([[0.37, 0.86, 1.0]] * 2)
    lg1, r1 = m(x, stat, phys)
    x2 = x.clone()
    x2[:, :, 1000:] += 5.0
    lg2, r2 = m(x2, stat, phys)
    assert torch.allclose(lg1[:, :1000], lg2[:, :1000], atol=1e-5)
    assert torch.allclose(r1[:, :1000], r2[:, :1000], atol=1e-5)
    assert not torch.allclose(lg1[:, 1000:], lg2[:, 1000:])


def test_physics_head_matches_itu_inverse():
    """With the correction forced to 0 dB, the rate is the ITU inverse power law."""
    m = TCN(4, 3, head="physics").eval()
    with torch.no_grad():
        m.head.weight.zero_()
        m.head.bias[:] = torch.tensor([0.0, float(np.log(0.2 / 0.8))])  # sigmoid → 0.2 → c = 0 dB
    x = torch.zeros(1, 4, 10)
    x[0, 0] = 5.37
    a, b, L = 0.370, 0.860, 1.07
    _, r = m(x, torch.zeros(1, 3), torch.tensor([[a, b, L]]))
    assert np.isclose(float(r[0, -1]), (5.37 / (a * L)) ** (1 / b), rtol=1e-3)


def test_dynamic_features_use_only_past():
    rng = np.random.default_rng(0)
    tl = 50 + rng.normal(0, 0.1, (3, 3000))
    f1 = feat.dynamic_features(tl, np.ones(3), np.full(3, 38.0), np.array(["vertical"] * 3))
    tl2 = tl.copy()
    tl2[:, 2000:] += 10
    f2 = feat.dynamic_features(tl2, np.ones(3), np.full(3, 38.0), np.array(["vertical"] * 3))
    assert np.allclose(f1[:, :, :2000], f2[:, :, :2000])


def test_low_reference_survives_long_rain():
    """Rain 70% of the last 3 days: the median is fooled, the low percentile is not."""
    rng = np.random.default_rng(1)
    T = 6 * 1440
    tl = 50 + rng.normal(0, 0.1, (1, T))
    rain = np.zeros(T, bool)
    minute_of_day = np.arange(T) % 1440
    rain[3 * 1440:] = minute_of_day[3 * 1440:] < 1010      # last 3 days: rain 70% of each day
    tl[0, rain] += 6.0
    ref = feat.low_reference(tl)
    med = feat.bn.move_median(tl, 1440, axis=-1)
    assert abs(ref[0, -1] - 50) < 0.5          # still the dry level
    assert med[0, -1] > 55                     # median now sits on the rain
    # causal: future changes do not move earlier references
    tl2 = tl.copy()
    tl2[0, 4000:] += 20
    assert np.allclose(feat.low_reference(tl2)[0, :3990], ref[0, :3990], equal_nan=True)


def test_neighbour_check_removes_lonely_rain_only():
    from targ.neighbours import neighbour_check, neighbour_matrix
    # three paths ~2 km apart, one far away (50 km); path 1 has two directions
    lat0 = np.array([45.00, 45.00, 45.02, 45.04, 45.50])
    lon0 = np.array([9.00, 9.01, 9.00, 9.00, 9.00])
    lat1 = np.array([45.00, 45.00, 45.02, 45.04, 45.50])
    lon1 = np.array([9.01, 9.00, 9.01, 9.01, 9.01])
    A = neighbour_matrix(lat0, lon0, lat1, lon1)
    assert A[0, 1] == 0 and A[0, 2] == 1 and A[0, 4] == 0      # same path / neighbour / far
    R = np.zeros((5, 120))
    R[0, 10:20] = 5.0          # path 0 alone (its other direction does not count)
    R[1, 10:20] = 5.0
    R[0, 60:70] = 5.0          # path 0 with support from path 2, 10 min later
    R[2, 70:75] = 2.0
    R[4, 30:40] = 9.0          # far link: no neighbour with data -> cannot be checked
    R[3, :] = np.nan
    R[2, :5] = np.nan
    out, info = neighbour_check(R, A, step_min=1)
    assert (out[0, 10:20] == 0).all() and (out[1, 10:20] == 0).all()
    assert (out[0, 60:70] == 5).all() and (out[2, 70:75] == 2).all()
    assert (out[4, 30:40] == 9).all()
