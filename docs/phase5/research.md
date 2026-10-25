# Phase 5 research notes: how flash floods are simulated, and where the data comes from

Written 2026-10-25. These notes back the choices in [PLAN.md](PLAN.md).

## 1. How flash floods are simulated

There are three families of method, from cheap to expensive.

| Family | What it does | Examples | Good for | Weak at |
|---|---|---|---|---|
| Rainfall thresholds ("flash flood guidance") | Compares the rain over 1/3/6 h on a small basin with the rain needed to flood it, given how wet the soil is | WMO Global Flash Flood Guidance System; Norbiato et al. 2008 | Warnings across a whole country, in real time | Says *whether* a basin floods, not *where* the water goes. Uniform-rain thresholds overestimate the true threshold. One German test found 91% detection but 56% false alarms. |
| Hydrological models (rain → river flow) | Computes losses (infiltration) and routes the runoff to an outlet | SCS curve number + unit hydrograph (HEC-HMS), wflow, GR4H | Peak flow at a basin outlet | No flood map. Needs a hydraulic model downstream (HEC-RAS 1D/2D). This is what Yamani et al. (2016) did for Ghardaïa. |
| 2D "rain-on-grid" hydraulic models | Rain falls on every cell of a terrain grid, and the shallow-water equations move the water downhill | LISFLOOD-FP (local-inertial "ACC" solver, GPU in 8.1), HEC-RAS 2D, TELEMAC-2D, Landlab OverlandFlow (Python), Inunda (GPU, differentiable, 2026), TRITON | Where water goes, how deep and how fast it runs, street by street | Cost; no sewers unless coupled, so small storms can be over-predicted; roughness is uncertain for very shallow flow |

The **local-inertial approximation** (Bates et al. 2010; de Almeida et al. 2012) is the standard compromise. It keeps the
water's momentum but drops the advection term. It is explicit, stable at time steps set by the wave speed, and accurate
for floods that spread over land. LISFLOOD-FP's ACC solver and Landlab OverlandFlow both use it. It is about 100 lines of
array code, so we implement it ourselves (PyTorch, GPU or CPU) and keep LISFLOOD-FP as an optional outside check
(GPLv3, non-commercial).

**Losses.** The SCS curve number (CN) method is the usual choice where soil data is thin. It has one parameter per cell,
read from land cover and the soil's hydrologic group (A–D). GCN250 (Jaafar et al. 2019, *Scientific Data*) builds a
global 250 m CN map this way from ESA CCI land cover and HYSOGs250m. We apply the same tables to the finer ESA WorldCover
10 m map.

**Hazard to people.** Flood hazard is classed from water depth and speed together, using the UK DEFRA/EA FD2320 hazard
rating HR = d·(v + 0.5) + DF and the Australian H1–H6 classes.

**Towers and floods.** Commercial-link rain has already been fed into runoff models:
- Pastorek et al. (2019) used 16 links on a 1.3 km² Prague catchment. Short links inside the catchment reproduced the
  runoff peaks best, and the runoff error falls as the links get longer.
- Pastorek et al. (2020) and Fencl et al. (2017) found that correcting link rain against gauges removes the bias, but can
  blunt the heavy-rain peaks.
- Smiatek et al. (2017, *ERL*) found that link rain improved high-flow simulation in German river basins.
- A 2025 *J. Hydrology* study fed raw link attenuation into machine-learning runoff models. They matched physics models
  driven by gauges.

Nobody has done this for a North African flash flood. That is the gap Part 5 fills, as a simulation.

## 2. What happened in the two storms (what we can check against)

### Algiers, Bab El Oued, 9–10 November 2001

Main source: Menad, Douvinet, Beltrando & Arnaud-Fassetta (2012), *Géomorphologie* 18(3):337–350,
doi:10.4000/geomorphologie.9954.

**The catchment**
- Oued Koriche, about 10.2 km², draining into western Bab El Oued. Its outlet is upstream of the Chevalley–Triolet
  expressway.
- Sub-catchments: Frais Vallon, Scotto-Nadal, Birtraria, Sidi Medjber, Baranes, Chemin-du-Fort and Koriche.

**Rain**

| Station | Total |
|---|---|
| Bouzaréah | 263 mm, 19:30 on 9 Nov to 10:30 on 10 Nov (previous record 135 mm) |
| Port d'Alger and Bir Mourad Raïs | 159 and 208 mm (the paper's table and text swap these two) |
| Dar El Beida (inland) | 31 mm |

At Bouzaréah, the evening of 9 Nov brought 114 mm in 5 h. The morning of 10 Nov brought two bursts of about 55 mm in
30 min (about 110 mm/h), at about 08:00–08:30 and 09:30–10:00.

**Flow at the outlet**
- Flood marks: peak 143 m³/s and volume about 1.8 Mm³ (water only). The national water agency (ANRH) put the total, mud
  included, at 2.6 Mm³, of which about 30% was solid.
- The ANRH figure of 730 m³/s at Triolet is judged implausible by Menad et al. (73 m³/s per km²). They propose 9–14 m³/s
  per km², which is 90–140 m³/s.
- Their own model (RuiCells) gave 88.7 m³/s, which is 60% of the flood marks.

**Land and outcome**
- Land cover: 78% urban in 2005 (54% dense). Redoing the storm on 1986 land cover lowered the peak by only 6%.
- 781 dead and 115 missing. Mud deposits were 0.85–2.45 m thick (Machane et al. 2008).

### Ghardaïa, M'zab valley, 1 October 2008

The sources disagree, so ranges are given.

**Rain**
- Over the two days: about 80 mm (ANRH).
- On the morning of the flood: a burst of 28–40 mm at about 03:00–04:00 on 1 Oct, depending on the source.

**Peak flow and the dam**
- Peak flow at Ghardaïa: estimates range from about 900 m³/s (all tributaries combined, press), through 1,200 m³/s, to
  1,600 m³/s with water more than 8 m deep (Springer chapter, 2021). The 10-year and 100-year floods are about 205 and
  722 m³/s.
- A dam break upstream added more than 900 m³/s, according to one source. Sediment had raised the wadi bed by up to
  1.5 m.

**Satellite maps and outcome**
- The International Charter "Space and Major Disasters" was activated for Ghardaïa on 4 October 2008 (ESA, SERTIT
  maps). Charter maps are the only independent flood outline for this storm.
- About 33–89 dead, depending on the date of the report.

### What this means for our runs

- The Phase 4 storms come from a satellite that averages over 10-km squares. Over Koriche the simulated storm totals are
  far below the 263 mm measured at Bouzaréah, so the flood on the satellite storm will be milder than the real one.
- We therefore run two versions of each storm:
  1. **As seen by the satellite.** This is the fair test of towers against truth.
  2. **Matched to the gauge.** Every rain map is multiplied by the same factor, so the basin storm total matches the
     gauge record. This is the fair test of the flood model against the flood marks.

## 3. Data sources

| Layer | Source | Resolution | Access | Notes |
|---|---|---|---|---|
| Terrain | Copernicus GLO-30 DEM | 30 m | public cloud COG (AWS), no key | A surface model: it includes buildings. Already used in Phase 4. |
| Terrain, bare earth (optional) | FABDEM (Hawker et al. 2022) | 30 m | data.bris.ac.uk download, CC BY-NC-SA 4.0 | Best of the free 30 m terrain maps in a 2024 test against 65 lidar surveys. Server timed out on 2026-10-25, so it is kept as an optional backend. |
| Land cover | ESA WorldCover 2021 v200 | 10 m | public cloud COG (AWS), no key | Built-up, trees, grass, bare, water... |
| Soil hydrologic group | HYSOGs250m (ORNL DAAC) | 250 m | NASA Earthdata token (we already have one for IMERG); read as a window | Codes 1–4 = A–D, 11–14 = dual groups (A/D...), 255 = no data |
| Ready-made curve numbers (cross-check) | GCN250 | 250 m | figshare | Same idea at a coarser land-cover scale |
| River network, check (optional) | MERIT Hydro / HydroSHEDS | 90 m | registration | We derive our own drainage from the 30 m terrain instead |
| Rain | Phase 4 hourly fields (truth / AI / classical / sparse network) at 500 m; IMERG at 10 km outside the tower box | | already on disk | The single link between Phase 4 and Phase 5 |
| Satellite flood outlines | Global Flood Database (Tellman et al. 2021, MODIS 250 m, 2000–2018) | 250 m | Google Earth Engine account | **Manual**: needs an Earth Engine login. Probably useless for narrow urban floods. |
| Ghardaïa 2008 flood outline | International Charter / SERTIT maps | map sheets | disasterscharter.org (timed out on 2026-10-25) | **Manual** if the site stays down |
| Gauge records, flood marks, sewer network | ANRH, ONM, SEAAL | | not public | **Manual** / institutional request; the literature values above stand in |

## Sources

- Menad W. et al. (2012) Géomorphologie 18(3):337–350. https://journals.openedition.org/geomorphologie/9954
- Flood Analysis and Mitigation Strategies in Algeria (2021), Springer. https://link.springer.com/chapter/10.1007/978-981-16-2904-4_3
- Floods and mudflows on November 11, 2001 in Bab El Oued. https://www.researchgate.net/publication/319211040
- Yamani et al. (2016) Mapping of vulnerability of flooded area in arid region, Ghardaïa. https://link.springer.com/article/10.1007/s40808-016-0183-x
- M'zab flood risk susceptibility (2024) WST. https://doi.org/10.2166/wst.2024.343
- International Charter activation, Floods in Algeria (2008). https://disasterscharter.org/activations/floods-in-algeria
- ReliefWeb, Algeria flash floods Oct 2008. https://reliefweb.int/disaster/ff-2008-000178-dza
- LISFLOOD-FP 8.1 (Shaw et al. 2023, GMD). https://gmd.copernicus.org/articles/16/2391/2023/
- Landlab OverlandFlow (Adams et al. 2017, GMD). https://gmd.copernicus.org/articles/10/1645/2017/
- Inunda GPU solver (2026). https://arxiv.org/pdf/2607.09614
- Urban rain-on-grid with street sensors, NYC (2026). https://www.sciencedirect.com/science/article/pii/S0309170826000199
- GCN250 (Jaafar et al. 2019). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6690949/
- HYSOGs250m. https://catalog.data.gov/dataset/global-hydrologic-soil-groups-hysogs250m-for-curve-number-based-runoff-modeling-3f96f
- FABDEM paper. https://www.fathom.global/academic-papers/a-30-m-global-map-of-elevation-with-forests-and-buildings-removed/
- Global DEM accuracy test, 2024. https://doaj.org/article/87fa65b8ab4b4f3a888fbe40764fd41a
- Global Flood Database. https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1
- WMO Flash Flood Guidance System. https://wmo.int/node/25418
- CML for urban drainage (Pastorek et al. 2019). https://pubmed.ncbi.nlm.nih.gov/31541849/
- CML for river runoff (2017, ERL). https://iopscience.iop.org/article/10.1088/1748-9326/aa5f46
- Roughness for rain-on-grid (2022, Water). https://www.doi.org/10.3390/w14244017
