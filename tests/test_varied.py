"""Tests for the Phase 2 follow-up: varied training, cross-validation, neighbour inputs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from targ import features as feat
from targ import train_torch as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_neighbour_features_count_raised_neighbours():
    dyn = np.zeros((3, 5, 100), np.float32)
    dyn[:, 3] = 1                               # all present
    dyn[0, 0, 40:] = 3.0                        # link 0 raised from minute 40
    dyn[1, 0, 50:] = 2.0                        # link 1 raised from minute 50
    A = sparse.csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], np.float32))
    f = feat.neighbour_features(dyn, A)
    assert f[2, 0, 30] == 0 and f[2, 0, 45] == 0.5 and f[2, 0, 70] == 1.0
    assert (f[:, 1] == 1).all()
    # the same moment only: nothing from later minutes
    dyn2 = dyn.copy()
    dyn2[0, 0, 80:] = 9.0
    assert np.array_equal(feat.neighbour_features(dyn2, A)[:, :, :80], f[:, :, :80])


def test_log_rate_loss_weighs_light_rain():
    logit = torch.zeros(1, 4)
    y = torch.tensor([[1.0, 1.0, 50.0, float("nan")]])
    under_light = torch.tensor([[0.5, 0.5, 50.0, 0.0]])
    under_heavy = torch.tensor([[1.0, 1.0, 45.0, 0.0]])
    # in mm/h the heavy minute dominates; in log space halving light rain costs more
    assert T.loss_fn(logit, under_light, y, 1.0, "mse") < T.loss_fn(logit, under_heavy, y, 1.0, "mse")
    assert T.loss_fn(logit, under_light, y, 1.0, "log") > T.loss_fn(logit, under_heavy, y, 1.0, "log")


def test_cross_validation_never_trains_on_the_held_out_month():
    import phase2_italy_cv as CV
    times = pd.date_range("2022-05-01", "2022-09-30T23:59", freq="1min").values
    tr, va = CV.pieces(times, "2022", held_out=7)
    t = pd.DatetimeIndex(times)
    for a, b in tr + va:
        assert not (t[a:b].month == 7).any()
    # validation is the last 7 days of each training month, never mixed with training
    for (a, b), (c, d) in zip(tr, va):
        assert b == c and (t[d - 1] - t[c]).days == 6


def test_train_multi_uses_only_allowed_pieces():
    rng = np.random.default_rng(0)
    n, T_ = 4, 6000
    dyn = rng.normal(size=(n, 5, T_)).astype(np.float32)
    stat = np.zeros((n, 3), np.float32)
    phys = np.tile(np.array([[0.1, 1.0, 2.0]], np.float32), (n, 1))
    target = np.full((n, T_), np.nan, np.float32)
    target[:, 4000:] = 5.0                      # only outside the allowed piece
    m, _ = T.train_multi([{"dyn": [dyn], "stat": stat, "phys": phys, "target": target,
                           "t_ranges": [(0, 3500)], "weight": 1.0}],
                         steps=3, batch=2, crop=1200, warm=1100, model_kw={"n_levels": 6, "head": "physics"})
    # every target seen was NaN, so the loss was 0 and nothing changed: weights equal a fresh model's
    torch.manual_seed(0)
    fresh = T.TCN(5, 3, n_levels=6, head="physics")
    assert all(torch.allclose(a.cpu(), b) for a, b in zip(m.state_dict().values(), fresh.state_dict().values()))


def test_new_links_split_keeps_sides_apart():
    import xarray as xr
    import phase2_mixed as M
    lon0 = np.array([9.5, 10.90, 10.95, 11.5, 12.0])
    ds = xr.Dataset(coords={"cml_id": list("abcde")},
                    data_vars={"site_0_lon": ("cml_id", lon0), "site_1_lon": ("cml_id", lon0)})
    w, e = M.side_index(ds, "west"), M.side_index(ds, "east")
    assert list(w) == [0] and list(e) == [3, 4]            # links inside the gap belong to neither
    d = {"ds": ds, "tl": np.arange(5 * 4).reshape(5, 4), "blk": {"ai": np.arange(5)}, "tr": (0, 3)}
    s = M.subset_links(d, e)
    assert list(s["ds"].cml_id.values) == ["d", "e"] and s["tl"].shape == (2, 4)
    assert list(s["blk"]["ai"]) == [3, 4] and s["tr"] == (0, 3)
