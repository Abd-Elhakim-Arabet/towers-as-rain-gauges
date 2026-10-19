# Phase 2 — AI model (causal TCN, physics head) — REAL data

## Sweden (OpenMRG), August 2015 held out

| | MCC | TPR / FPR | radar 5-min r / bias | radar hourly r / bias | gauges hourly r / RMSE / bias | median per-link r |
|---|---|---|---|---|---|---|
| classical (Phase 1) | 0.578 | 0.71 / 0.046 | 0.58 / -10% | 0.76 / -10% | 0.81 / 1.04 / -6% | 0.88 |
| AI physics (seed 0) | 0.614 | 0.65 / 0.029 | 0.53 / -5% | 0.72 / -5% | 0.82 / 0.96 / -9% | 0.87 |
| AI free head (ablation) | 0.587 | 0.68 / 0.039 | 0.52 / -29% | 0.68 / -29% | 0.71 / 1.28 / -29% | – |
| AI no 1-dB augmentation (ablation) | 0.603 | 0.68 / 0.035 | 0.49 / -27% | 0.67 / -28% | 0.75 / 1.20 / -40% | – |
| classical, data rounded to 1 dB | 0.336 | 0.82 / 0.230 | 0.61 / +18% | 0.78 / +17% | 0.85 / 0.82 / +26% | – |
| AI physics, data rounded to 1 dB | 0.541 | 0.67 / 0.049 | 0.54 / +12% | 0.74 / +11% | 0.83 / 0.84 / +6% | – |

5 retrainings (seeds 0–4; GPU training is not bit-reproducible, so seed 0 here differs slightly from the row above):

- wet_dry_vs_radar_5min.MCC: mean 0.617, range 0.603 – 0.634
- rain_vs_radar_5min.r: mean 0.514, range 0.507 – 0.522
- rain_vs_radar_hourly.r: mean 0.717, range 0.700 – 0.730
- rain_vs_radar_hourly.bias_pct: mean -13.319, range -20.994 – -3.011
- rain_vs_gauges_hourly.r: mean 0.803, range 0.771 – 0.817
- rain_vs_gauges_hourly.RMSE: mean 1.006, range 0.942 – 1.071
- rain_vs_gauges_hourly.bias_pct: mean -19.537, range -26.059 – -8.440

## Italy (OpenRainER), May–Sep 2021 and 2022, zero-shot

| season | method | MCC 15-min | radar hourly r / RMSE / bias | gauges hourly r / bias | median per-link r | hours > 100 mm/h |
|---|---|---|---|---|---|---|
| 2021 (81 sub-links) | classical | 0.251 | 0.22 / 5.92 / +64% | 0.45 / +39% | 0.84 | 35 |
| 2021 (81 sub-links) | ai | 0.490 | 0.34 / 4.45 / +44% | 0.24 / +65% | 0.82 | 8 |
| 2022 (72 sub-links) | classical | 0.269 | 0.12 / 12.90 / +132% | 0.31 / +64% | 0.79 | 124 |
| 2022 (72 sub-links) | ai | 0.497 | 0.33 / 5.51 / +37% | 0.50 / +28% | 0.81 | 19 |

## Netherlands, 1 Jun – 22 Aug 2014, 15-min min/max, zero-shot (1081 sub-links)

| method | MCC hourly | radar hourly r / RMSE / bias | median per-link r | hours > 100 mm/h |
|---|---|---|---|---|
| classical | 0.626 | 0.16 / 8.65 / +54% | 0.79 | 219 |
| ai | 0.556 | 0.44 / 2.77 / +17% | 0.76 | 40 |

## MindSpore

- MindSpore 2.10.0 (CPU): PyTorch weights ported, max difference 8.6e-05 mm/h over 276480 values; rain/no-rain agreement 100%.
- Training from scratch on CPU: loss 0.63 → 0.39 in 300 steps.

