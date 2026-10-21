"""Phase 4: 1-minute, ~500-m rain fields from half-hourly 0.1-degree IMERG.

Satellite rain is too coarse for links that are 1-10 km long and read every
minute. We add the missing detail in three steps, all documented and tunable:

1. Motion: one storm-motion vector per half-hour, from optical flow (pysteps
   Lucas-Kanade) on the IMERG frames, median over raining pixels.
2. Large scale, every minute: each IMERG frame is shifted along the motion and
   blended with the next one, so rain moves smoothly instead of jumping every
   30 minutes.
3. Small scale: multiplied by exp(sigma*N - sigma^2/2), where N is a random field
   with the power-spectrum slope of real radar rain at scales below one IMERG cell
   (larger scales come from IMERG itself, as in RainFARM), carried along with the
   storm and slowly renewed (AR(1) with time scale tau). sigma, slope and the wet share
   are measured on real radar (scripts/phase4_rain_texture.py); tau is set so the
   field decorrelates in time like real radar rain does.
Finally each IMERG cell's half-hour mean is restored (smooth rescaling), so the
simulation keeps the satellite's amounts and only adds detail.

Real radar shows the patchiness only down to 1 km; below that the same spectral law is
extended (as RainFARM does), which is plausible but unchecked.
The result is a plausible storm, not the real one at 500 m: it is the "truth" the
simulated network is judged against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates


@dataclass
class TextureParams:
    sigma: float = 0.60              # input strength; tuned so std of log(15-min pixel / 10-km mean) ~ 0.59, as real radar Nov 2021
    slope: float = -3.87             # radially averaged power-spectrum slope, real radar
    tau_min: float = 20.0            # renewal time of the small-scale pattern (Lagrangian)
    wet_threshold: float = 0.1       # mm/h; below is dry
    fine_deg: float = 0.005           # ~500 m
    mass_passes: int = 6             # passes of the smooth mass correction
    seed: int = 0


def storm_motion(da: xr.DataArray) -> np.ndarray:
    """Per half-hour motion (dlon, dlat) in degrees per minute; shape (T, 2)."""
    from pysteps import motion
    lk = motion.get_method("LK")
    T = da.sizes["time"]
    out = np.zeros((T, 2))
    v = np.nan_to_num(da.values)
    for t in range(T - 1):
        if (v[t] >= 0.5).sum() < 10 or (v[t + 1] >= 0.5).sum() < 10:
            out[t] = out[t - 1] if t else 0.0
            continue
        V = lk(np.stack([v[t], v[t + 1]]))             # (2, ny, nx): pixels per 30 min (x, y)
        m = (v[t] >= 0.5)
        out[t] = [np.median(V[0][m]) * 0.1 / 30, np.median(V[1][m]) * 0.1 / 30]
    out[-1] = out[-2]
    # IMERG latitude increases with row index, so +y pixels = northwards
    return out


def _spectral_noise(rng, n, slope, cutoff_px=10.0):
    """Gaussian field with power ~ k**slope, only at scales the satellite cannot see
    (wavelength < cutoff_px fine pixels = one IMERG cell), as in RainFARM."""
    k = np.sqrt(np.add.outer(np.fft.fftfreq(n) ** 2, np.fft.fftfreq(n) ** 2))
    k[0, 0] = 1.0
    amp = k ** (slope / 2)
    amp[k < 1.0 / cutoff_px] = 0.0
    f = np.fft.ifft2(amp * np.fft.fft2(rng.standard_normal((n, n)))).real
    return (f - f.mean()) / f.std()


def downscale(da: xr.DataArray, domain, p: TextureParams = TextureParams()):
    """da: IMERG (time [half-hour starts], lat, lon) mm/h. domain: (lat0, lon0, lat1, lon1).
    Returns xarray (time [1-min], lat, lon) in mm/h on a p.fine_deg grid, and the motion."""
    rng = np.random.default_rng(p.seed)
    lat0, lon0, lat1, lon1 = domain
    flat = np.round(np.arange(lat0, lat1 + 1e-9, p.fine_deg), 4)
    flon = np.round(np.arange(lon0, lon1 + 1e-9, p.fine_deg), 4)
    LON, LAT = np.meshgrid(flon, flat)
    clat, clon = da.lat.values, da.lon.values
    cv = np.nan_to_num(da.values.astype(np.float32))
    V = storm_motion(da)
    T = da.sizes["time"]
    nmin = T * 30
    times = da.time.values[0] + np.arange(nmin) * np.timedelta64(1, "m")

    def sample(frame, dlon, dlat):
        """Coarse frame bilinearly sampled at fine points displaced by (-dlon, -dlat)."""
        r = (LAT - dlat - clat[0]) / 0.1
        c = (LON - dlon - clon[0]) / 0.1
        return map_coordinates(frame, [r, c], order=1, mode="nearest")

    # small-scale noise on a periodic canvas, carried with the storm
    n = 256
    rho = np.exp(-1.0 / p.tau_min)
    N = _spectral_noise(rng, n, p.slope, 0.1 / p.fine_deg)
    disp = np.zeros(2)                 # cumulative displacement in degrees (lon, lat)
    out = np.empty((nmin, len(flat), len(flon)), np.float32)
    for m in range(nmin):
        k = m // 30
        mid_k = k * 30 + 15
        if m < mid_k:
            j, w = k - 1, (m - (mid_k - 30)) / 30.0
        else:
            j, w = k, (m - mid_k) / 30.0
        a = max(j, 0)
        b = min(j + 1, T - 1)
        ta, tb = (m - (a * 30 + 15)), (m - (b * 30 + 15))   # minutes since each frame's midpoint
        La = sample(cv[a], V[a, 0] * ta, V[a, 1] * ta)
        Lb = sample(cv[b], V[b, 0] * tb, V[b, 1] * tb)
        L = (1 - w) * La + w * Lb if a != b else La
        disp += V[k]
        if m:
            N = rho * N + np.sqrt(1 - rho ** 2) * _spectral_noise(rng, n, p.slope, 0.1 / p.fine_deg)
        r = ((LAT - disp[1] - lat0) / p.fine_deg) % n
        c = ((LON - disp[0] - lon0) / p.fine_deg) % n
        Nm = map_coordinates(N, [r, c], order=1, mode="grid-wrap")
        out[m] = L * np.exp(p.sigma * Nm - p.sigma ** 2 / 2)
    # restore each IMERG cell's half-hour mean (smooth factor, bilinear)
    rr = np.clip(np.floor((LAT - clat[0] + 0.05) / 0.1 + 1e-6).astype(int), 0, len(clat) - 1)
    cc = np.clip(np.floor((LON - clon[0] + 0.05) / 0.1 + 1e-6).astype(int), 0, len(clon) - 1)
    cell = rr * len(clon) + cc
    cnt = np.bincount(cell.ravel(), minlength=len(clat) * len(clon))
    for k in range(T):
        # smooth (bilinear) passes first, so the field has no steps at cell edges,
        # then one exact pass per cell for the small remainder
        for it in range(p.mass_passes + 1):
            blk = out[k * 30:(k + 1) * 30].mean(0)
            sums = np.bincount(cell.ravel(), blk.ravel(), minlength=len(clat) * len(clon))
            sim = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan).reshape(len(clat), len(clon))
            fac = np.where(sim > 1e-3, cv[k] / np.maximum(sim, 1e-3), 1.0)
            fac = np.clip(np.where(np.isfinite(fac), fac, 1.0), 0.0, 20.0).astype(np.float32)
            if it < p.mass_passes:
                out[k * 30:(k + 1) * 30] *= sample(fac, 0.0, 0.0)[None]
            else:                      # last pass exact per cell (factors are close to 1 by now)
                out[k * 30:(k + 1) * 30] *= fac.ravel()[cell][None]
    out[out < p.wet_threshold] = 0.0
    fine = xr.DataArray(out, dims=("time", "lat", "lon"), coords={"time": times, "lat": flat, "lon": flon},
                        name="rain", attrs={"units": "mm/h", "note": "simulated 1-min rain, IMERG + texture"})
    return fine, V


def texture_stats(fine: xr.DataArray, block_px: int = 10) -> dict:
    """The statistics measured on real radar by scripts/phase4_rain_texture.py, on a simulated
    field: first averaged to ~1 km (the radar's pixel), then 15-min means, 10x10-pixel blocks
    with block mean >= 0.5 mm/h."""
    v = fine.values
    k = max(1, int(round(0.01 / float(fine.lat.values[1] - fine.lat.values[0]))))
    if k > 1:
        ny_, nx_ = v.shape[1] // k * k, v.shape[2] // k * k
        v = v[:, :ny_, :nx_].reshape(v.shape[0], ny_ // k, k, nx_ // k, k).mean((2, 4))
    T15 = v.shape[0] // 15
    v15 = v[:T15 * 15].reshape(T15, 15, *v.shape[1:]).mean(1)
    ny, nx = v15.shape[1] // block_px * block_px, v15.shape[2] // block_px * block_px
    x = v15[:, :ny, :nx]
    b = x.reshape(T15, ny // block_px, block_px, nx // block_px, block_px).mean((2, 4))
    up = np.repeat(np.repeat(b, block_px, 1), block_px, 2)
    with np.errstate(all="ignore"):
        lr = np.log(x / up)
    raining = up >= 0.5
    m = raining & (x >= 0.1) & np.isfinite(lr)
    out = {"sigma_log_ratio": float(lr[m].std()), "wet_share_mean": float((x >= 0.1)[raining].mean())}
    for k, name in ((1, "15min"), (2, "30min")):
        mm = m[:-k] & m[k:]
        out[f"log_ratio_corr_{name}"] = float(np.corrcoef(lr[:-k][mm], lr[k:][mm])[0, 1])
    return out
