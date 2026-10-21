# Phase 4: simulated Algerian network (SIMULATION, not validation)

Real tower positions (OpenCelliD) + plausible microwave tree per operator (30% fibre hubs, hop lengths from
Gothenburg/NL, frequency/polarisation from a real link of similar length) + IMERG storms downscaled to
500 m / 1 min (patchiness measured on Italian radar at 1 km) + real dry recordings from eastern Italian links
(the AI = final recipe trained on western Italy never saw them) + wet antennas + -100 dBm floor + 1-dB rounding.

Real radar texture (Nov 2021): sigma 0.59, wet 0.94, corr 15/30 min 0.39/0.22

## Bab El Oued flood (algiers)

IMERG city total 49 mm, peak 4.0 mm/h at 2001-11-10T07:00; 2434 sites -> 1705 links (3410 sub-links), median length 3.6 km; reliability keeps 2458.
Simulated texture: sigma 0.69, wet 0.91, corr 15/30 min 0.60/0.33.

| method | link hourly r | link total bias | map hourly r | map total bias | city-mean hourly r |
|---|---|---|---|---|---|
| ai | 0.86 | -23% | 0.92 | -24% | 0.98 |
| classical | 0.66 | -13% | 0.91 | -12% | 0.99 |

| share of links | n links | AI map r | classical map r | covered |
|---|---|---|---|---|
| 1% | 17 | 0.83 | 0.83 | 91% |
| 3% | 51 | 0.84 | 0.85 | 98% |
| 10% | 170 | 0.91 | 0.91 | 100% |
| 25% | 426 | 0.92 | 0.89 | 100% |
| 50% | 852 | 0.92 | 0.92 | 100% |
| 100% | 1705 | 0.92 | 0.91 | 100% |

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

