"""Framework-free inputs for the Phase 2 neural networks.

Every channel uses only the past (causal), so a model built on them can run live.
Per sub-link and minute:

  excess      TL minus a causal "dry level": the 20th percentile of the previous
              3 days (of 15-min block medians) [dB]. Rain only ever adds loss, so a
              low percentile stays at the dry level even when it rained most of the
              time; a median fails once more than half of its window was rainy.
  r_itu       the ITU-R P.838-3 rain rate that `excess` would mean if it were all rain
              [mm/h, log1p]. Hands the network the physics so it can learn corrections
              (wet antenna, noise) instead of relearning the power law per frequency.
  std15       standard deviation of TL over the previous 15 min [dB] ("jumpiness").
  present     1 where TL exists, 0 where it was missing (missing values are set to 0).
  excess_24h  TL minus the median of the previous 24 h [dB] (the original reference).
              Next to `excess` it tells the network whether the recent past was rainy.

Static per sub-link: log(length km), frequency/40 GHz, polarisation (H = 1).
"""

from __future__ import annotations

import bottleneck as bn
import numpy as np
import pycomlink as pycml

CHANNELS = ("excess", "r_itu", "std15", "present", "excess_24h")
STATICS = ("log_length", "freq_40", "pol_h")
DRY_WINDOW_MIN = 1440
LOW_BLOCK_MIN = 15          # block medians of 15 min ...
LOW_WINDOW_BLOCKS = 288     # ... over the previous 3 days
LOW_QUANTILE = 0.2


def itu_a_b(freq_ghz: np.ndarray, pol: np.ndarray):
    return pycml.processing.k_R_relation.a_b(np.asarray(freq_ghz, float),
                                             pol=np.asarray(pol).astype(str),
                                             approx_type="ITU_2005")


def dynamic_features(tl: np.ndarray, length_km: np.ndarray, freq_ghz: np.ndarray,
                     pol: np.ndarray, dry_window: int = DRY_WINDOW_MIN,
                     std_window: int = 15) -> np.ndarray:
    """tl: (n_links, n_time) 1-min total loss [dB]. Returns float32 (n_links, 4, n_time)."""
    tl = np.asarray(tl, dtype=np.float64)
    med = bn.move_median(tl, window=dry_window, min_count=dry_window // 4, axis=-1)
    excess = tl - low_reference(tl)
    excess_24h = tl - med
    a, b = itu_a_b(freq_ghz, pol)
    k = np.clip(excess, 0, None) / np.asarray(length_km, float)[:, None]
    r_itu = np.log1p(np.clip((k / a[:, None]) ** (1 / b[:, None]), 0, 300))
    std = bn.move_std(tl, window=std_window, min_count=std_window // 2, axis=-1)
    present = np.isfinite(excess).astype(np.float64)
    out = np.stack([excess, r_itu, std, present, excess_24h], axis=1)
    out = np.nan_to_num(out, nan=0.0)
    out[:, 0] = np.clip(out[:, 0], -10, 40)
    out[:, 4] = np.clip(out[:, 4], -10, 40)
    out[:, 2] = np.clip(out[:, 2], 0, 10)
    return out.astype(np.float32)


def low_reference(x: np.ndarray, block: int = LOW_BLOCK_MIN, window: int = LOW_WINDOW_BLOCKS,
                  q: float = LOW_QUANTILE) -> np.ndarray:
    """Causal low-percentile dry level along the last axis.

    x is cut into blocks of `block` steps; each block's median is taken; the q-quantile
    of the previous `window` complete blocks is the reference for every step of the
    next block. Only finished blocks are used, so nothing comes from the future.
    """
    import pandas as pd

    x = np.asarray(x, dtype=np.float64)
    n, T = x.shape
    nb = -(-T // block)
    pad = np.full((n, nb * block), np.nan)
    pad[:, :T] = x
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bmed = np.nanmedian(pad.reshape(n, nb, block), axis=2)
    ref_b = (pd.DataFrame(bmed.T).rolling(window, min_periods=window // 8).quantile(q)
             .shift(1).to_numpy().T)
    return np.repeat(ref_b, block, axis=1)[:, :T]


def static_features(length_km, freq_ghz, pol) -> np.ndarray:
    pol_h = (np.asarray(pol).astype(str) == "horizontal").astype(float)
    return np.stack([np.log(np.asarray(length_km, float)), np.asarray(freq_ghz, float) / 40.0,
                     pol_h], axis=1).astype(np.float32)


def physics_params(length_km, freq_ghz, pol) -> np.ndarray:
    """(n_links, 3): ITU-R P.838-3 a, b and length [km], for the physics head."""
    a, b = itu_a_b(freq_ghz, pol)
    return np.stack([a, b, np.asarray(length_km, float)], axis=1).astype(np.float32)


def quantise(x: np.ndarray, step: float) -> np.ndarray:
    """Round to a fixed step, as many operators' equipment does (e.g. 1 dB)."""
    return (x / step).round() * step if hasattr(x, "round") else np.round(np.asarray(x) / step) * step


# --- 15-min min/max variant (Netherlands-style data) --------------------------------

MINMAX_CHANNELS = ("excess_mid", "excess_min", "excess_max", "r_itu_mid", "present", "excess_mid_24h")
MINMAX_DRY_WINDOW = 96   # 24 h of 15-min steps (median reference, extra channel)
MINMAX_LOW_WINDOW = 288  # 3 days of 15-min steps (low-percentile reference, main)


def minmax_features(loss_max: np.ndarray, loss_min: np.ndarray, length_km, freq_ghz, pol,
                    dry_window: int = MINMAX_DRY_WINDOW) -> np.ndarray:
    """Inputs from 15-min minimum / maximum loss (loss = -RSL when TSL is constant).

    loss_max comes from the 15-min minimum RSL (the weakest moment), loss_min from
    the maximum RSL. Both are referenced to the 20th percentile of loss_min over the
    previous 3 days (robust to long rain); the mid-point is also given against the
    24-h median, as in the 1-min features.
    Channel 0 is the mid-point excess (what the physics head corrects and converts).
    Returns float32 (n_links, 6, n_steps).
    """
    lmax, lmin = np.asarray(loss_max, float), np.asarray(loss_min, float)
    med = bn.move_median(lmin, window=dry_window, min_count=dry_window // 4, axis=-1)
    ref = low_reference(lmin, block=1, window=MINMAX_LOW_WINDOW)
    ex_max, ex_min = lmax - ref, lmin - ref
    a, b = itu_a_b(freq_ghz, pol)
    L = np.asarray(length_km, float)[:, None]

    def to_r(ex):
        return np.log1p(np.clip((np.clip(ex, 0, None) / L / a[:, None]) ** (1 / b[:, None]), 0, 300))

    present = (np.isfinite(ex_max) & np.isfinite(ex_min)).astype(float)
    mid = (ex_min + ex_max) / 2
    mid_24h = (lmin + lmax) / 2 - med
    out = np.stack([mid, ex_min, ex_max, to_r(mid), present, mid_24h], axis=1)
    out = np.nan_to_num(out, nan=0.0)
    out[:, :3] = np.clip(out[:, :3], -10, 40)
    out[:, 5] = np.clip(out[:, 5], -10, 40)
    return out.astype(np.float32)


# --- neighbour inputs (optional, Phase 2 follow-up) ---------------------------------

NB_CHANNELS = ("nb_raised", "nb_any")
NB_RAISED_DB = 1.0       # a neighbour counts as "raised" with >= 1 dB extra loss ...
NB_SMOOTH_MIN = 15       # ... averaged over its last 15 minutes


def neighbour_features(dyn: np.ndarray, A) -> np.ndarray:
    """Two channels from the neighbours, at the same moment (nothing from the future).

    nb_raised  share of neighbours with data whose extra loss (channel 0, last 15 min
               mean) is >= NB_RAISED_DB; 0 when no neighbour has data
    nb_any     1 if at least one neighbour has data, else 0
    A: (n, n) sparse 0/1 neighbour matrix (targ.neighbours.neighbour_matrix), so a
    link's own other direction does not count. Long rain raises many neighbours at
    once; a drifting or broken link is raised alone.
    Returns float32 (n_links, 2, T).
    """
    ex = dyn[:, 0].astype(np.float32)
    present = dyn[:, 3].astype(np.float32)
    ex_s = bn.move_mean(np.where(present > 0, ex, np.nan), window=NB_SMOOTH_MIN, min_count=1, axis=-1)
    raised = np.nan_to_num((ex_s >= NB_RAISED_DB).astype(np.float32) * (np.isfinite(ex_s)))
    has = np.isfinite(ex_s).astype(np.float32)
    num = A @ raised
    den = A @ has
    frac = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return np.stack([frac, (den > 0).astype(np.float32)], axis=1).astype(np.float32)
