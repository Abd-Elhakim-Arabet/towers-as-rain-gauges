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
