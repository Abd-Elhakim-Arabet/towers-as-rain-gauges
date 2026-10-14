"""Shared splits and metrics, so classical and AI methods are scored identically."""

from __future__ import annotations

import numpy as np
import poligrain as plg
import xarray as xr

# OpenMRG time split (fixed in Phase 1, reused by every later phase).
# August is never used for tuning.
OPENMRG_SPLITS = {
    "train": slice("2015-06-01", "2015-07-15"),
    "val": slice("2015-07-16", "2015-07-31"),
    "test": slice("2015-08-01", "2015-08-31"),
}
OPENMRG_CALIB = slice("2015-06-01", "2015-07-31")  # train + val, for classical tuning

WET_MM_H = 0.1  # reference rain rate at or above this = wet


def to_radar_times(x_1min: xr.DataArray, radar_times, how: str = "mean") -> xr.DataArray:
    """Average 1-min CML values over the 5 minutes centred on each radar scan time."""
    rolled = x_1min.rolling(time=5, center=True, min_periods=3)
    rolled = rolled.mean() if how == "mean" else rolled.max()
    return rolled.sel(time=radar_times)


def hourly(x_1min: xr.DataArray, min_coverage: float = 0.8) -> xr.DataArray:
    """Hourly mean rain rate [mm/h = mm per hour]; NaN if < min_coverage of minutes present."""
    mean = x_1min.resample(time="1h").mean()
    cov = x_1min.notnull().resample(time="1h").mean()
    return mean.where(cov >= min_coverage)


def wet_dry_scores(ref_rain: np.ndarray, est_wet: np.ndarray) -> dict:
    """MCC/TPR/FPR. ref is rain rate (wet if ≥ WET_MM_H); est is a 0..1 wet indicator."""
    m = plg.validation.calculate_wet_dry_metrics(
        np.ravel(ref_rain).astype(float), np.ravel(est_wet).astype(float),
        ref_thresh=WET_MM_H, est_thresh=0.5)
    return {"MCC": m["matthews_correlation_coefficient"], "TPR": m["true_positive_ratio"],
            "FPR": m["false_positive_ratio"], "N_wet_ref": int(m["N_wet_ref"]),
            "N": int(m["N_all"] - m["N_nan"])}


def rain_scores(ref: np.ndarray, est: np.ndarray) -> dict:
    """Pearson r, RMSE, MAE, bias over pairs where ref or est ≥ WET_MM_H (poligrain rule)."""
    ref, est = np.ravel(ref).astype(float), np.ravel(est).astype(float)
    m = plg.validation.calculate_rainfall_metrics(ref, est, ref_thresh=WET_MM_H,
                                                  est_thresh=WET_MM_H)
    ok = ~(np.isnan(ref) | np.isnan(est))
    return {"r": m["pearson_correlation_coefficient"], "RMSE": m["root_mean_square_error"],
            "MAE": m["mean_absolute_error"], "bias_pct": m["percent_bias"],
            "ref_total": float(ref[ok].sum()), "est_total": float(est[ok].sum()),
            "N": int(m["N_all"] - m["N_nan"])}
