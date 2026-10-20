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
| classical + neighbour check | 0.581 | 0.71 / 0.046 | 0.58 / -11% | 0.76 / -10% | 0.81 / 1.04 / -6% | 0.88 |
| AI + neighbour check | 0.622 | 0.68 / 0.031 | 0.52 / -13% | 0.71 / -13% | 0.80 / 1.02 / -19% | 0.85 |

Neighbour check removed 1.7% (classical) and 1.6% (AI) of rainy minutes in Sweden.

Ablation rows come from single runs (phase2_openmrg.py). 5 retrainings of the main design (seeds 0–4):

- wet_dry_vs_radar_5min.MCC: mean 0.619, range 0.612 – 0.637
- rain_vs_radar_5min.r: mean 0.517, range 0.508 – 0.531
- rain_vs_radar_hourly.r: mean 0.707, range 0.687 – 0.731
- rain_vs_radar_hourly.bias_pct: mean -15.130, range -27.661 – -3.822
- rain_vs_gauges_hourly.r: mean 0.792, range 0.767 – 0.841
- rain_vs_gauges_hourly.RMSE: mean 1.050, range 0.897 – 1.147
- rain_vs_gauges_hourly.bias_pct: mean -22.068, range -32.048 – -5.963

## Italy (OpenRainER), May–Sep 2021 and 2022, zero-shot

QC: sub-links with >= 50% data. `old links` = the earlier QC set (also dropping sub-links the classical detector calls wet > 30% of the time). `reliable` = kept by the raw-signal reliability score (targ.reliability). `_nb` = after the neighbour check (10 km, ±15 min).

| season | links | method | MCC 15-min | radar hourly r / RMSE / bias | gauges hourly r / bias | median per-link r | hours > 100 mm/h |
|---|---|---|---|---|---|---|---|
| 2021 | all (281) | classical | 0.156 | 0.22 / 4.44 / +163% | 0.42 / +103% | 0.82 | 54 |
| 2021 | all (281) | ai | 0.413 | 0.64 / 1.59 / +36% | 0.66 / +21% | 0.83 | 0 |
| 2021 | all (281) | classical_nb | 0.158 | 0.25 / 4.18 / +121% | 0.45 / +76% | 0.83 | 43 |
| 2021 | all (281) | ai_nb | 0.449 | 0.70 / 1.49 / +12% | 0.72 / +3% | 0.84 | 0 |
| 2021 | old links (81) | classical | 0.251 | 0.22 / 5.92 / +64% | 0.45 / +39% | 0.84 | 35 |
| 2021 | old links (81) | ai | 0.509 | 0.70 / 1.84 / -5% | 0.74 / -3% | 0.82 | 0 |
| 2021 | old links (81) | classical_nb | 0.254 | 0.25 / 5.51 / +51% | 0.54 / +28% | 0.84 | 26 |
| 2021 | old links (81) | ai_nb | 0.518 | 0.74 / 1.72 / -10% | 0.75 / -7% | 0.83 | 0 |
| 2021 | reliable (196) | classical | 0.184 | 0.29 / 3.98 / +72% | 0.45 / +57% | 0.84 | 47 |
| 2021 | reliable (196) | ai | 0.508 | 0.74 / 1.47 / +0% | 0.78 / -1% | 0.85 | 0 |
| 2021 | reliable (196) | classical_nb | 0.186 | 0.32 / 3.76 / +59% | 0.47 / +47% | 0.84 | 37 |
| 2021 | reliable (196) | ai_nb | 0.521 | 0.77 / 1.44 / -6% | 0.78 / -6% | 0.86 | 0 |

Neighbour check 2021: removed 20.4% (classical) / 22.3% (AI) of rainy minutes; median 6 neighbours per sub-link.

| 2022 | all (267) | classical | 0.166 | 0.12 / 8.40 / +167% | 0.29 / +77% | 0.78 | 165 |
| 2022 | all (267) | ai | 0.413 | 0.53 / 2.27 / +29% | 0.68 / -3% | 0.81 | 5 |
| 2022 | all (267) | classical_nb | 0.168 | 0.15 / 7.23 / +130% | 0.38 / +49% | 0.80 | 106 |
| 2022 | all (267) | ai_nb | 0.443 | 0.56 / 2.29 / +12% | 0.69 / -16% | 0.81 | 5 |
| 2022 | old links (72) | classical | 0.269 | 0.12 / 12.90 / +132% | 0.31 / +64% | 0.79 | 124 |
| 2022 | old links (72) | ai | 0.502 | 0.68 / 2.31 / -7% | 0.81 / -10% | 0.79 | 0 |
| 2022 | old links (72) | classical_nb | 0.271 | 0.15 / 11.48 / +92% | 0.41 / +36% | 0.82 | 77 |
| 2022 | old links (72) | ai_nb | 0.506 | 0.71 / 2.17 / -14% | 0.84 / -14% | 0.80 | 0 |
| 2022 | reliable (194) | classical | 0.190 | 0.12 / 10.02 / +106% | 0.35 / +40% | 0.80 | 152 |
| 2022 | reliable (194) | ai | 0.495 | 0.66 / 1.91 / -6% | 0.71 / -17% | 0.82 | 2 |
| 2022 | reliable (194) | classical_nb | 0.193 | 0.15 / 8.38 / +76% | 0.44 / +23% | 0.81 | 98 |
| 2022 | reliable (194) | ai_nb | 0.508 | 0.68 / 1.86 / -11% | 0.72 / -20% | 0.83 | 2 |

Neighbour check 2022: removed 17.2% (classical) / 19.4% (AI) of rainy minutes; median 7 neighbours per sub-link.


## Netherlands, 1 Jun – 22 Aug 2014, 15-min min/max, zero-shot (1908 sub-links; old QC set 1081)

| links | method | MCC hourly | radar hourly r / RMSE / bias | median per-link r | hours > 100 mm/h |
|---|---|---|---|---|---|
| all | classical | 0.440 | 0.16 / 6.34 / +151% | 0.78 | 262 |
| all | ai | 0.460 | 0.49 / 2.20 / +42% | 0.76 | 29 |
| all | classical_nb | 0.463 | 0.19 / 5.81 / +128% | 0.78 | 199 |
| all | ai_nb | 0.484 | 0.52 / 2.10 / +34% | 0.76 | 19 |
| old links | classical | 0.626 | 0.16 / 8.65 / +54% | 0.79 | 219 |
| old links | ai | 0.547 | 0.50 / 2.47 / +16% | 0.77 | 26 |
| old links | classical_nb | 0.632 | 0.20 / 7.55 / +43% | 0.79 | 163 |
| old links | ai_nb | 0.562 | 0.53 / 2.31 / +12% | 0.77 | 16 |
| reliable (1653) | classical | 0.505 | 0.19 / 6.48 / +60% | 0.78 | 237 |
| reliable (1653) | ai | 0.490 | 0.52 / 2.14 / +25% | 0.76 | 28 |
| reliable (1653) | classical_nb | 0.517 | 0.22 / 5.75 / +52% | 0.78 | 180 |
| reliable (1653) | ai_nb | 0.508 | 0.55 / 2.03 / +20% | 0.77 | 18 |

Neighbour check: removed 10.6% (classical) / 8.9% (AI) of rainy 15-min steps; median 25 neighbours per sub-link.

## Rain after a mostly rainy day (past 24 h > 50% rainy)

Rain measured as % of the reference total in those steps (before = 24-h median dry level; after = 3-day 20th-percentile dry level).

| test | share of rainy steps | classical | AI before | AI after |
|---|---|---|---|---|
| sweden_aug2015_5min | 0.0% | 66% | 47% | 34% |
| italy_2021_hourly | 2.3% | 99% | 86% | 77% |
| italy_2022_hourly | 10.6% | 62% | 64% | 55% |
| netherlands_2014_hourly | 9.7% | 75% | 48% | 69% |

## Follow-up: why runs disagree, long rain in Italy, more varied training

Runs differ in the strength of their rain (AI/radar where both say rain), not in rain/no-rain. July bias vs August bias across runs: r = 0.97.

| run | July bias | July strength | Aug bias | Aug strength | Aug missed / false (share of radar) |
|---|---|---|---|---|---|
| physics_seed0.pt | -12% | 0.87 | -28% | 0.69 | 10% / 10% |
| physics_seed1.pt | +7% | 1.01 | -14% | 0.79 | 12% / 16% |
| physics_seed2.pt | +16% | 1.12 | -5% | 0.89 | 11% / 16% |
| physics_seed3.pt | +6% | 1.02 | -18% | 0.77 | 11% / 14% |
| physics_seed4.pt | +5% | 1.02 | -14% | 0.81 | 12% / 15% |

One-season experiments (5 runs each; Sweden August, Italy 2022 held out; long-rain groups in 15-min steps):

| config | Sweden MCC (mean) | Sweden gauge bias (runs) | Italy 2022 bias (runs) | IT22 after dry day | IT22 after mostly rainy day |
|---|---|---|---|---|---|
| se_mse | 0.619 | -32% .. -6% | -6% .. +41% | 83% | 32% |
| se_log | 0.621 | -20% .. +9% | -8% .. +32% | 81% | 48% |
| mix21_mse | 0.648 | -17% .. +0% | -3% .. +4% | 95% | 28% |
| mix21_log | 0.651 | -23% .. -16% | -15% .. -5% | 83% | 20% |
| mix21nb_mse | 0.685 | -17% .. -11% | -16% .. +0% | 91% | 35% |
| mix21rel_mse | 0.648 | -18% .. -13% | -7% .. +5% | 95% | 42% |

Month-by-month cross-validation (Sweden + both Italian summers minus the judged month, reliable links only, 3 runs per month). Hourly long-rain table, AI rain as % of radar, groups by share of past 24 h rainy (<10%, 10-25%, 25-50%, >50%):

| season | Sweden-only kept model | varied training | Italy total bias (runs) |
|---|---|---|---|
| 2021 | 77% / 79% / 70% / 77% | 93% / 94% / 87% / 95% | +1% .. +4% |
| 2022 | 84% / 86% / 72% / 55% | 98% / 102% / 87% / 60% | -1% .. +4% |

Sweden August, same networks: MCC 0.648, gauges r 0.81, RMSE 1.01, bias -17% (runs -23% .. -12%).
Italy and Netherlands zero-shot results above (and Phase 3) keep the Sweden-only model on purpose.

## MindSpore

- MindSpore 2.10.0 (CPU): PyTorch weights ported, max difference 7.7e-05 mm/h over 276480 values; rain/no-rain agreement 100%.
- Training from scratch on CPU: loss 0.59 → 0.40 in 300 steps.

