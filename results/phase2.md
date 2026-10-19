# Phase 2 — AI model (causal TCN, physics head) — REAL data

## Sweden (OpenMRG), August 2015 held out

| | MCC | TPR / FPR | radar 5-min r / bias | radar hourly r / bias | gauges hourly r / RMSE / bias | median per-link r |
|---|---|---|---|---|---|---|
| classical (Phase 1) | 0.578 | 0.71 / 0.046 | 0.58 / -10% | 0.76 / -10% | 0.81 / 1.04 / -6% | 0.88 |
| AI physics (selected by validation: seed 1) | 0.615 | 0.68 / 0.032 | 0.52 / -13% | 0.71 / -13% | 0.79 / 1.04 / -20% | 0.85 |
| AI free head (ablation) | 0.618 | 0.68 / 0.031 | 0.54 / -20% | 0.72 / -20% | 0.75 / 1.15 / -19% | – |
| AI no 1-dB augmentation (ablation) | 0.634 | 0.68 / 0.029 | 0.51 / -23% | 0.70 / -23% | 0.79 / 1.08 / -27% | – |
| classical, data rounded to 1 dB | 0.336 | 0.82 / 0.230 | 0.61 / +18% | 0.78 / +17% | 0.85 / 0.82 / +26% | – |
| AI physics, data rounded to 1 dB | 0.568 | 0.65 / 0.039 | 0.53 / -7% | 0.71 / -9% | 0.80 / 0.96 / -15% | – |

Ablation rows come from single runs (phase2_openmrg.py). 5 retrainings of the main design (seeds 0–4):

- wet_dry_vs_radar_5min.MCC: mean 0.619, range 0.612 – 0.637
- rain_vs_radar_5min.r: mean 0.517, range 0.508 – 0.531
- rain_vs_radar_hourly.r: mean 0.707, range 0.687 – 0.731
- rain_vs_radar_hourly.bias_pct: mean -15.130, range -27.661 – -3.822
- rain_vs_gauges_hourly.r: mean 0.792, range 0.767 – 0.841
- rain_vs_gauges_hourly.RMSE: mean 1.050, range 0.897 – 1.147
- rain_vs_gauges_hourly.bias_pct: mean -22.068, range -32.048 – -5.963

## Italy (OpenRainER), May–Sep 2021 and 2022, zero-shot

| season | method | MCC 15-min | radar hourly r / RMSE / bias | gauges hourly r / bias | median per-link r | hours > 100 mm/h |
|---|---|---|---|---|---|---|
| 2021 (81 sub-links) | classical | 0.251 | 0.22 / 5.92 / +64% | 0.45 / +39% | 0.84 | 35 |
| 2021 (81 sub-links) | ai | 0.509 | 0.70 / 1.84 / -5% | 0.74 / -3% | 0.82 | 0 |
| 2022 (72 sub-links) | classical | 0.269 | 0.12 / 12.90 / +132% | 0.31 / +64% | 0.79 | 124 |
| 2022 (72 sub-links) | ai | 0.502 | 0.68 / 2.31 / -7% | 0.81 / -10% | 0.79 | 0 |

## Netherlands, 1 Jun – 22 Aug 2014, 15-min min/max, zero-shot (1081 sub-links)

| method | MCC hourly | radar hourly r / RMSE / bias | median per-link r | hours > 100 mm/h |
|---|---|---|---|---|
| classical | 0.626 | 0.16 / 8.65 / +54% | 0.79 | 219 |
| ai | 0.547 | 0.50 / 2.47 / +16% | 0.77 | 26 |

## Rain after a mostly rainy day (past 24 h > 50% rainy)

Rain measured as % of the reference total in those steps (before = 24-h median dry level; after = 3-day 20th-percentile dry level).

| test | share of rainy steps | classical | AI before | AI after |
|---|---|---|---|---|
| sweden_aug2015_5min | 0.0% | 66% | 47% | 34% |
| italy_2021_hourly | 2.3% | 99% | 86% | 77% |
| italy_2022_hourly | 10.6% | 62% | 64% | 55% |
| netherlands_2014_hourly | 9.7% | 75% | 48% | 69% |

## MindSpore

- MindSpore 2.10.0 (CPU): PyTorch weights ported, max difference 7.7e-05 mm/h over 276480 values; rain/no-rain agreement 100%.
- Training from scratch on CPU: loss 0.59 → 0.40 in 300 steps.

