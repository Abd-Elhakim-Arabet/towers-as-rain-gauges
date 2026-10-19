"""Framework-free inputs for the Phase 2 neural networks.

Every channel uses only the past (causal), so a model built on them can run live.
Per sub-link and minute:

  excess   TL minus the median TL of the previous 24 h [dB]. A causal "dry level":
           a rain event lasting a few hours barely moves a 24-h median.
  r_itu    the ITU-R P.838-3 rain rate that `excess` would mean if it were all rain
           [mm/h, log1p]. Hands the network the physics so it can learn corrections
           (wet antenna, noise) instead of relearning the power law per frequency.
  std15    standard deviation of TL over the previous 15 min [dB] ("jumpiness").
  present  1 where TL exists, 0 where it was missing (missing values are set to 0).

Static per sub-link: log(length km), frequency/40 GHz, polarisation (H = 1).
"""

from __future__ import annotations

import bottleneck as bn
import numpy as np
import pycomlink as pycml

CHANNELS = ("excess", "r_itu", "std15", "present")
STATICS = ("log_length", "freq_40", "pol_h")
DRY_WINDOW_MIN = 1440


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
    excess = tl - med
    a, b = itu_a_b(freq_ghz, pol)
    k = np.clip(excess, 0, None) / np.asarray(length_km, float)[:, None]
    r_itu = np.log1p(np.clip((k / a[:, None]) ** (1 / b[:, None]), 0, 300))
    std = bn.move_std(tl, window=std_window, min_count=std_window // 2, axis=-1)
    present = np.isfinite(excess).astype(np.float64)
    out = np.stack([excess, r_itu, std, present], axis=1)
    out = np.nan_to_num(out, nan=0.0)
    out[:, 0] = np.clip(out[:, 0], -10, 40)
    out[:, 2] = np.clip(out[:, 2], 0, 10)
    return out.astype(np.float32)


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

MINMAX_CHANNELS = ("excess_mid", "excess_min", "excess_max", "r_itu_mid", "present")
MINMAX_DRY_WINDOW = 96   # 24 h of 15-min steps


def minmax_features(loss_max: np.ndarray, loss_min: np.ndarray, length_km, freq_ghz, pol,
                    dry_window: int = MINMAX_DRY_WINDOW) -> np.ndarray:
    """Inputs from 15-min minimum / maximum loss (loss = -RSL when TSL is constant).

    loss_max comes from the 15-min minimum RSL (the weakest moment), loss_min from
    the maximum RSL. Both are referenced to the 24-h median of loss_min.
    Channel 0 is the mid-point excess (what the physics head corrects and converts).
    Returns float32 (n_links, 5, n_steps).
    """
    lmax, lmin = np.asarray(loss_max, float), np.asarray(loss_min, float)
    ref = bn.move_median(lmin, window=dry_window, min_count=dry_window // 4, axis=-1)
    ex_max, ex_min = lmax - ref, lmin - ref
    a, b = itu_a_b(freq_ghz, pol)
    L = np.asarray(length_km, float)[:, None]

    def to_r(ex):
        return np.log1p(np.clip((np.clip(ex, 0, None) / L / a[:, None]) ** (1 / b[:, None]), 0, 300))

    present = (np.isfinite(ex_max) & np.isfinite(ex_min)).astype(float)
    mid = (ex_min + ex_max) / 2
    out = np.stack([mid, ex_min, ex_max, to_r(mid), present], axis=1)
    out = np.nan_to_num(out, nan=0.0)
    out[:, :3] = np.clip(out[:, :3], -10, 40)
    return out.astype(np.float32)
