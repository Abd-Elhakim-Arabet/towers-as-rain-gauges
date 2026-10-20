"""Phase 4, step 3a: how patchy is real rain inside one satellite pixel?

IMERG gives one value per 0.1 deg (~10 km) cell and half-hour. Links are 1-10 km
long, so the simulation needs the rain's small-scale pattern too. We do not
invent its strength: we measure it on real gauge-adjusted radar (OpenRainER,
Emilia-Romagna, ~1 km, 15 min) in November 2021, an autumn Mediterranean month
like the Algiers storm.

Measured, on 10x10 km blocks (radar ~1 km pixels) with block mean >= 0.5 mm/h:
  - wet share: share of 1-km pixels with >= 0.1 mm/h
  - sigma_log: std of log(pixel / block mean) over wet pixels
  - spatial power-spectrum slope of the 1-km field (for the noise generator)
  - lag-15/30-min correlation of log(pixel / block mean) at the same place

    python scripts/phase4_rain_texture.py
Writes results/phase4_rain_texture.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
NC = ROOT / "data" / "interim" / "p4" / "RADadj_202111.nc"
BY, BX = 11, 8          # 11 x 0.009 deg lat, 8 x 0.0126 deg lon: about 0.1 x 0.1 deg


def radial_slope(f):
    """Slope of the radially averaged power spectrum (log-log), wavelengths 2-30 km."""
    F = np.abs(np.fft.fftshift(np.fft.fft2(f - f.mean()))) ** 2
    ny, nx = f.shape
    ky, kx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(ny)), np.fft.fftshift(np.fft.fftfreq(nx)), indexing="ij")
    k = np.sqrt(kx ** 2 + ky ** 2)
    bins = np.linspace(1 / 30, 1 / 2, 15)
    idx = np.digitize(k.ravel(), bins)
    p = np.array([F.ravel()[idx == i].mean() for i in range(1, len(bins))])
    kc = 0.5 * (bins[1:] + bins[:-1])
    ok = p > 0
    return float(np.polyfit(np.log(kc[ok]), np.log(p[ok]), 1)[0])


def main():
    rad = xr.open_dataset(NC).rainfall_amount * 4.0          # 15-min depth -> mm/h
    v = rad.values.astype(np.float32)                          # (time, lat, lon)
    T, ny, nx = v.shape
    ny2, nx2 = ny // BY * BY, nx // BX * BX
    v = v[:, :ny2, :nx2]
    blk = np.nanmean(v.reshape(T, ny2 // BY, BY, nx2 // BX, BX), axis=(2, 4))
    up = np.repeat(np.repeat(blk, BY, axis=1), BX, axis=2)
    raining = up >= 0.5
    wet = v >= 0.1
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(v / up)
    m = raining & wet & np.isfinite(lr)
    sigma = float(np.std(lr[m]))
    # wet share per raining block
    wet_share = np.nanmean(np.where(raining, wet, np.nan).reshape(T, ny2 // BY, BY, nx2 // BX, BX), axis=(2, 4))
    ws = wet_share[np.isfinite(wet_share)]
    # temporal correlation of the log ratio at the same pixel
    lag = {}
    for k, name in ((1, "15min"), (2, "30min"), (4, "60min")):
        a, b = lr[:-k], lr[k:]
        mm = m[:-k] & m[k:]
        lag[name] = float(np.corrcoef(a[mm], b[mm])[0, 1])
    # spectral slope on the wettest frames
    tot = np.nanmean(np.where(raining, v, 0), axis=(1, 2))
    frames = np.argsort(tot)[-40:]
    slopes = [radial_slope(np.nan_to_num(v[t])) for t in frames]
    res = {"source": "OpenRainER RADadj, November 2021, ~1 km, 15 min", "block_deg": 0.1,
           "n_wet_pixels": int(m.sum()), "sigma_log_ratio": sigma,
           "wet_share_mean": float(ws.mean()), "wet_share_median": float(np.median(ws)),
           "log_ratio_corr": lag, "spectral_slope_median": float(np.median(slopes)),
           "spectral_slope_iqr": [float(np.percentile(slopes, 25)), float(np.percentile(slopes, 75))]}
    print(json.dumps(res, indent=1))
    (ROOT / "results" / "phase4_rain_texture.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
