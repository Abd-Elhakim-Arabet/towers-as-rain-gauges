# Phase 4: simulated Algerian network (SIMULATION, not validation)

Real tower positions (OpenCelliD) + plausible microwave tree per operator (30% fibre hubs, hop lengths from
Gothenburg/NL, frequency/polarisation from a real link of similar length) + IMERG storms downscaled to
500 m / 1 min (patchiness measured on Italian radar at 1 km) + real dry recordings from eastern Italian links
(the AI = final recipe trained on western Italy never saw them) + wet antennas + -100 dBm floor + 1-dB rounding.

Real radar texture (Nov 2021): sigma 0.59, wet 0.94, corr 15/30 min 0.39/0.22

## Bab El Oued flood (algiers)

IMERG city total 49 mm, peak 4.0 mm/h at 2001-11-10T07:00; 4092 sites -> 2864 links (5728 sub-links), median length 3.6 km; reliability keeps 4079.
Simulated texture: sigma 0.69, wet 0.91, corr 15/30 min 0.60/0.33.

| method | link hourly r | link total bias | map hourly r | map total bias | city-mean hourly r |
|---|---|---|---|---|---|
| ai | 0.87 | -24% | 0.94 | -26% | 0.98 |
| classical | 0.70 | -13% | 0.93 | -14% | 0.99 |

| share of links | n links | AI map r | classical map r | covered |
|---|---|---|---|---|
| 1% | 29 | 0.83 | 0.83 | 100% |
| 3% | 86 | 0.87 | 0.88 | 100% |
| 10% | 286 | 0.91 | 0.83 | 100% |
| 25% | 716 | 0.92 | 0.90 | 100% |
| 50% | 1432 | 0.93 | 0.93 | 100% |
| 100% | 2864 | 0.94 | 0.93 | 100% |

## Ghardaia (M'zab) flood (ghardaia)

IMERG city total 38 mm, peak 5.5 mm/h at 2008-09-29T02:00; 220 sites -> 153 links (306 sub-links), median length 2.9 km; reliability keeps 257.
Simulated texture: sigma 0.56, wet 0.97, corr 15/30 min 0.46/0.10.

| method | link hourly r | link total bias | map hourly r | map total bias | city-mean hourly r |
|---|---|---|---|---|---|
| ai | 0.91 | -14% | 0.89 | -14% | 0.99 |
| classical | 0.54 | +11% | 0.84 | +8% | 0.98 |

| share of links | n links | AI map r | classical map r | covered |
|---|---|---|---|---|
| 1% | 2 | 0.79 | 0.81 | 95% |
| 3% | 5 | 0.82 | 0.83 | 94% |
| 10% | 15 | 0.85 | 0.84 | 99% |
| 25% | 38 | 0.86 | 0.79 | 100% |
| 50% | 76 | 0.89 | 0.89 | 100% |
| 100% | 153 | 0.89 | 0.84 | 100% |

