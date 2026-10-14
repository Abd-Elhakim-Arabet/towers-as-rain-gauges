# Phase 1 — classical baseline, OpenMRG (REAL data)

Test period: **August 2015, held out** (tuning used June–July only).

Pipeline: rolling-std wet/dry (window 60 min, centred; threshold 0.35 dB) → constant baseline (mean of last 5 dry min) → wet-antenna Pastorek 2021 (A_max 10.0 dB) → ITU-R P.838-3 k-R power law.

Quality control (from June–July only): drop sub-links flagged wet > 30% of the time or with < 50% data. Keeps 683 / 728 sub-links; 166 sub-links lie within 1 km of one of the 10 city gauges.

| Sub-links | Wet/dry MCC | TPR / FPR | vs radar 5-min: r / RMSE [mm/h] / bias | vs radar hourly | median per-link r (hourly, radar) | vs gauges hourly |
|---|---|---|---|---|---|---|
| QC (683) | 0.578 | 0.708 / 0.046 | 0.58 / 2.68 / -10% | 0.76 / 1.25 / -10% | 0.88 | 0.81 / 1.04 / -6% |
| all (728) | 0.536 | 0.713 / 0.061 | 0.31 / 4.94 / +24% | 0.33 / 3.61 / +24% | 0.88 | 0.81 / 1.03 / -6% |

Reference check (how good is the radar?): radar pixel vs gauge, hourly, August: r = 0.80, RMSE = 1.14 mm/h, bias = +3%.

Wet-antenna tuning on June–July (QC sub-links, vs gauges hourly; chosen = lowest RMSE, ties within 2% → smallest |bias|):

| WAA model | r | RMSE [mm/h] | bias |
|---|---|---|---|
| none | 0.528 | 4.529 | +247.9% |
| schleiss max 1.0 | 0.531 | 3.090 | +143.1% |
| schleiss max 1.5 | 0.509 | 2.669 | +95.0% |
| schleiss max 2.3 | 0.445 | 2.341 | +49.5% |
| pastorek A_max 4.0 | 0.636 | 2.040 | +77.3% |
| pastorek A_max 6.0 | 0.664 | 1.520 | +36.3% |
| pastorek A_max 8.0 | 0.680 | 1.248 | +8.2% |
| pastorek A_max 10.0 | 0.688 | 1.132 | -11.9% |
| pastorek A_max 14.0 | 0.690 | 1.128 | -38.2% |

Notes:
- Metrics use pairs where reference or estimate ≥ 0.1 mm/h (poligrain convention); wet = radar path average ≥ 0.1 mm/h.
- Radar = SMHI reflectivity, Z = 200 R^1.5, **not gauge-adjusted**, 2 km pixels, instantaneous 5-min scans. It is a reference, not ground truth.
- Pooled Pearson r across links is sensitive to a few bad links; the median per-link r is the robust view.
- The centred 60-min window looks 30 min ahead, so this baseline is not strictly real-time; Phase 2 models are causal.
