# Phase 4: simulated Algerian network (SIMULATION, not validation)

Real tower positions (OpenCelliD) + plausible microwave tree per operator (30% fibre hubs, hop lengths from
Gothenburg/NL, frequency/polarisation from a real link of similar length) + IMERG storms downscaled to
1 km / 1 min (patchiness measured on Italian radar) + real dry recordings from eastern Italian links
(the AI = final recipe trained on western Italy never saw them) + wet antennas + -100 dBm floor + 1-dB rounding.

Real radar texture (Nov 2021): sigma 0.59, wet 0.94, corr 15/30 min 0.39/0.22

## Bab El Oued flood (algiers)

IMERG city total 49 mm, peak 4.0 mm/h at 2001-11-10T07:00; 2434 sites -> 1705 links (3410 sub-links), median length 3.6 km; reliability keeps 2433.
Simulated texture: sigma 0.69, wet 0.92, corr 15/30 min 0.62/0.35.

| method | link hourly r | link total bias | map hourly r | map total bias | city-mean hourly r |
|---|---|---|---|---|---|
| ai | 0.87 | -23% | 0.91 | -26% | 0.97 |
| classical | 0.66 | -15% | 0.91 | -14% | 0.99 |

| share of links | n links | AI map r | classical map r | covered |
|---|---|---|---|---|
| 1% | 17 | 0.83 | 0.82 | 91% |
| 3% | 51 | 0.85 | 0.86 | 97% |
| 10% | 170 | 0.90 | 0.91 | 100% |
| 25% | 426 | 0.92 | 0.89 | 100% |
| 50% | 852 | 0.92 | 0.92 | 100% |
| 100% | 1705 | 0.91 | 0.91 | 100% |

## Ghardaia (M'zab) flood (ghardaia)

IMERG city total 38 mm, peak 5.5 mm/h at 2008-09-29T02:00; 220 sites -> 153 links (306 sub-links), median length 2.9 km; reliability keeps 256.
Simulated texture: sigma 0.56, wet 0.97, corr 15/30 min 0.47/0.11.

| method | link hourly r | link total bias | map hourly r | map total bias | city-mean hourly r |
|---|---|---|---|---|---|
| ai | 0.92 | -15% | 0.90 | -14% | 0.99 |
| classical | 0.55 | +9% | 0.84 | +7% | 0.99 |

| share of links | n links | AI map r | classical map r | covered |
|---|---|---|---|---|
| 1% | 2 | 0.82 | 0.83 | 94% |
| 3% | 5 | 0.82 | 0.82 | 93% |
| 10% | 15 | 0.86 | 0.85 | 99% |
| 25% | 38 | 0.88 | 0.81 | 99% |
| 50% | 76 | 0.89 | 0.90 | 100% |
| 100% | 153 | 0.90 | 0.84 | 100% |

