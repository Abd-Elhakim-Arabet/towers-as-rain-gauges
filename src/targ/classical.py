"""Classical CML rain retrieval: rolling-std wet/dry → baseline → wet-antenna → k-R.

This is the standard pipeline (Schleiss & Berne 2010 wet/dry; pycomlink baseline,
wet-antenna and ITU-R P.838-3 k-R functions). It is the reference the AI models
in Phase 2 have to beat. All inputs are 1-min data in OpenSense format.
"""

from __future__ import annotations

from dataclasses import dataclass

import bottleneck as bn
import numpy as np
import pycomlink as pycml
import xarray as xr


@dataclass(frozen=True)
class ClassicalParams:
    window_min: int = 60          # rolling std window (centered → uses 30 min of future data)
    threshold_db: float = 0.8     # rolling std above this = wet
    n_average_last_dry: int = 5   # baseline = mean of the last 5 dry minutes before an event
    waa: str = "pastorek"         # "none" | "schleiss" | "pastorek"
    waa_max_db: float = 2.3       # Schleiss 2013 max WAA
    waa_tau_min: float = 15.0     # Schleiss 2013 time constant
    pastorek_A_max: float = 14.0  # pycomlink default for Pastorek 2021
    max_gap_fill_min: int = 5     # forward-fill short RSL gaps before processing


def total_loss(ds: xr.Dataset, max_gap_fill_min: int) -> xr.DataArray:
    """TL = TSL − RSL [dB], with short gaps forward-filled."""
    tl = (ds.tsl - ds.rsl).transpose("cml_id", "time")
    return tl.ffill("time", limit=max_gap_fill_min)


def rolling_std(tl: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling std along the last axis (time)."""
    s = bn.move_std(tl, window=window, min_count=window // 2, axis=-1)
    shift = window // 2  # bottleneck windows are trailing; shift to center them
    out = np.full_like(s, np.nan)
    out[..., :-shift] = s[..., shift:]
    return out


def wet_dry(tl: np.ndarray, p: ClassicalParams) -> np.ndarray:
    """Float array: 1 = wet, 0 = dry, NaN where TL is missing."""
    s = rolling_std(tl, p.window_min)
    wet = (s > p.threshold_db).astype(float)
    wet[np.isnan(s) | np.isnan(tl)] = np.nan
    return wet


def retrieve(ds: xr.Dataset, p: ClassicalParams = ClassicalParams()) -> xr.Dataset:
    """Run the full classical chain; returns wet, baseline, A, waa, R per cml_id × time."""
    tl_da = total_loss(ds, p.max_gap_fill_min)
    tl = tl_da.values.astype(np.float64)
    wet = wet_dry(tl, p)

    n = tl.shape[0]
    baseline = np.full_like(tl, np.nan)
    waa = np.zeros_like(tl)
    for i in range(n):
        w = np.nan_to_num(wet[i], nan=0.0).astype(bool)
        baseline[i] = pycml.processing.baseline.baseline_constant(
            tl[i], w, n_average_last_dry=p.n_average_last_dry)
        if p.waa == "schleiss":
            waa[i] = pycml.processing.wet_antenna.waa_schleiss_2013(
                rsl=tl[i], baseline=baseline[i], wet=w, waa_max=p.waa_max_db,
                delta_t=1.0, tau=p.waa_tau_min)

    A_obs = np.clip(tl - baseline, 0, None)
    A_obs[np.nan_to_num(wet, nan=0.0) == 0] = 0.0  # attenuation only counts when wet

    if p.waa == "pastorek":
        pol = np.where(ds.polarization.values == "horizontal", "H", "V")
        for i in range(n):
            waa[i] = pycml.processing.wet_antenna.waa_pastorek_2021_from_A_obs(
                A_obs=A_obs[i], f_Hz=ds.frequency.values[i] * 1e9, pol=pol[i],
                L_km=ds.length.values[i], A_max=p.pastorek_A_max)
    A = np.clip(A_obs - waa, 0, None)

    # ITU-R P.838-3 (2005) k-R coefficients per sub-link
    a, b = pycml.processing.k_R_relation.a_b(
        ds.frequency.values.astype(float), pol=ds.polarization.values.astype(str),
        approx_type="ITU_2005")
    # Same power law as pycomlink's calc_R_from_A (which only takes 1-D input).
    R = (A / (a[:, None] * ds.length.values[:, None])) ** (1 / b[:, None])
    R[R < 0.1] = 0.0
    R = np.where(np.isnan(tl), np.nan, R)

    coords = {"cml_id": tl_da.cml_id, "time": tl_da.time}
    dims = ("cml_id", "time")
    return xr.Dataset({"tl": (dims, tl), "wet": (dims, wet), "baseline": (dims, baseline),
                       "waa": (dims, waa), "A": (dims, A), "R": (dims, R)}, coords=coords)


def quality_mask(ds_calib: xr.Dataset, p: ClassicalParams, max_wet_fraction: float = 0.3,
                 min_availability: float = 0.5) -> xr.DataArray:
    """True = keep sub-link. Computed on calibration data only.

    A sub-link is dropped if it is flagged wet far more often than it can rain
    (OpenMRG radar never exceeds ~0.09 wet fraction on any path in June–July),
    or if it has too little data.
    """
    tl = total_loss(ds_calib, p.max_gap_fill_min)
    wet = wet_dry(tl.values, p)
    wet_fraction = np.nanmean(wet, axis=1)
    availability = np.mean(~np.isnan(tl.values), axis=1)
    keep = (wet_fraction <= max_wet_fraction) & (availability >= min_availability)
    return xr.DataArray(keep, coords={"cml_id": tl.cml_id}, dims="cml_id", name="qc_keep")
