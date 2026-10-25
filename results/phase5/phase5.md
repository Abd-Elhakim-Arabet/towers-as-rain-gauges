# Phase 5: flood risk (SIMULATION)

Flood runs on the Phase 4 storms. Peaks and volumes are for the checked basin (Koriche at Bab El Oued; M'zab at Ghardaïa). 'gauge' = rain maps scaled to the gauge storm total.

## Algiers

- basin 9.1 km², curve number 85, tower rain over 100% of it
- true storm over the basin 72 mm; gauge 263 mm (factor 3.64)
- observed: Menad et al. 2012: flood marks 143 m3/s, 1.8 Mm3; plausible 9-14 m3/s/km2; their RuiCells model 88.7 m3/s

| storm | rain | peak m³/s | peak time | volume Mm³ | peak vs truth | flooded built-up CSI | communes Spearman | alarm | before the town floods |
|---|---|---|---|---|---|---|---|---|---|
| satellite | truth | 9 | 2001-11-10 09:54 | 0.26 |  |  |  | 2001-11-10 07:00 | -54 min |
| satellite | ai | 3 | 2001-11-10 10:13 | 0.08 | -68% | 0.65 | 0.93 | - | - |
| satellite | classical | 7 | 2001-11-10 10:10 | 0.18 | -27% | 0.83 | 0.93 | - | - |
| satellite | ai_sparse | 2 | 2001-11-10 09:44 | 0.07 | -73% | 0.62 | 0.91 | - | - |
| satellite | imerg | 9 | 2001-11-10 09:38 | 0.20 | +3% | 0.85 | 0.97 | 2001-11-10 13:00 | -414 min |
| gauge | truth | 62 | 2001-11-10 03:19 | 1.84 |  |  |  | 2001-11-09 21:00 | 29 min |
| gauge | ai | 32 | 2001-11-10 03:32 | 1.01 | -49% | 0.81 | 0.92 | 2001-11-09 22:05 | -36 min |
| gauge | classical | 44 | 2001-11-10 03:26 | 1.50 | -29% | 0.86 | 0.95 | 2001-11-09 20:05 | 84 min |
| gauge | ai_sparse | 36 | 2001-11-10 03:20 | 0.97 | -43% | 0.80 | 0.96 | 2001-11-09 20:05 | 84 min |
| gauge | imerg | 47 | 2001-11-10 09:05 | 1.56 | -24% | 0.84 | 0.96 | 2001-11-10 01:00 | -211 min |

| sensitivity (truth, gauge storm) | peak m³/s | vs base |
|---|---|---|
| wet_soil_amc3 | 67 | +8% |
| dry_soil_amc1 | 51 | -18% |
| terrain_unfilled | 61 | -1% |
| built_n_0.02 | 62 | +1% |
| built_n_0.10 | 60 | -3% |
| drains_20mmh | 27 | -56% |

| rain threshold | mm |
|---|---|
| 1 h | 45 |
| 3 h | 50 |
| 6 h | 52 |
| 12 h | 54 |
| 24 h | 57 |

## Ghardaïa

- basin 1386.6 km², curve number 86, tower rain over 5% of it
- true storm over the basin 50 mm; gauge 80 mm (factor 1.61)
- observed: 900 (press, all tributaries), 1,200, 1,600 m3/s (Springer 2021); 10-yr 205, 100-yr 722 m3/s; a dam break upstream added >900 m3/s

| storm | rain | peak m³/s | peak time | volume Mm³ | peak vs truth | flooded built-up CSI | communes Spearman | alarm | before the town floods |
|---|---|---|---|---|---|---|---|---|---|
| satellite | truth | 134 | 2008-10-01 16:58 | 5.40 |  |  |  | 2008-09-29 23:00 | 1716 min |
| satellite | ai | 134 | 2008-10-01 16:58 | 5.34 | -0% | 0.90 | 1.00 | 2008-09-29 23:05 | 1711 min |
| satellite | classical | 135 | 2008-10-01 16:59 | 5.43 | +0% | 0.90 | 1.00 | 2008-09-29 23:05 | 1711 min |
| satellite | ai_sparse | 134 | 2008-10-01 16:59 | 5.30 | -0% | 0.82 | 1.00 | 2008-09-29 23:05 | 1711 min |
| satellite | imerg | 135 | 2008-10-01 16:58 | 5.42 | +0% | 0.96 | 1.00 | 2008-09-30 03:00 | 1476 min |
| gauge | truth | 679 | 2008-10-01 12:57 | 27.74 |  |  |  | 2008-09-29 08:00 | 837 min |
| gauge | ai | 679 | 2008-10-01 12:58 | 27.56 | -0% | 0.98 | 1.00 | 2008-09-29 08:05 | 832 min |
| gauge | classical | 681 | 2008-10-01 12:57 | 27.91 | +0% | 0.94 | 1.00 | 2008-09-29 08:05 | 832 min |
| gauge | ai_sparse | 677 | 2008-10-01 13:00 | 27.36 | -0% | 0.90 | 1.00 | 2008-09-29 08:05 | 832 min |
| gauge | imerg | 680 | 2008-10-01 12:57 | 27.82 | +0% | 0.98 | 1.00 | 2008-09-29 12:00 | 597 min |

| sensitivity (truth, gauge storm) | peak m³/s | vs base |
|---|---|---|
| wet_soil_amc3 | 999 | +47% |
| dry_soil_amc1 | 176 | -74% |
| terrain_unfilled | 487 | -28% |
| built_n_0.02 | 692 | +2% |
| built_n_0.10 | 660 | -3% |

| rain threshold | mm |
|---|---|
| 3 h | 24 |
| 6 h | 24 |
| 12 h | 24 |
| 24 h | 24 |
