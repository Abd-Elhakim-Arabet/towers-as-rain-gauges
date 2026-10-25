# Phase 5 plan: from rain maps to flood risk

The goal is to turn the tower rain maps into "where will water run, how deep, how dangerous, and how early would we
know?". The research behind each choice is in [research.md](research.md).

> **Label.** Phase 5 runs on the Phase 4 *simulated* storms, so every Phase 5 number is a simulation too. The only
> outside checks are the literature values for the two real floods (flood marks, peak flows).

## Separation from Phases 0–4

Phase 5 must not change anything that already works. It lives in its own places and talks to Phase 4 in exactly one
spot.

| Part | Phase 5 location | Touches earlier phases? |
|---|---|---|
| Code | `src/floodrisk/`, a separate package; nothing in `src/targ/` imports it, and it does not import `targ` | no |
| Link to Phase 4 | `floodrisk/rain.py::Phase4Rain` reads `data/processed/phase4/sim_<city>.npz` (read-only) | reads its output file only |
| Scripts | `scripts/phase5/` | no |
| Data | `data/raw/flood/` (downloads), `data/processed/phase5/` (runs); both git-ignored | no |
| Results | `results/phase5/` | no |
| Tests | `tests/test_floodrisk.py` | no |
| Website | `site/flood/` with its own page at `/flood/` on the local site | one nav link and one copy step in `serve_site.py` |
| Paper | `docs/explainer/part5.tex` + `p5_numbers.tex` + `fig_p5_*` | one new `\input{part5}` line after Part 4, and new glossary entries; nothing earlier is edited |
| Tower refresh | not wired in; Phase 5 is rerun by hand (`scripts/phase5/run_all.sh`) | no |

**Separable.** Without Phase 4 the package still runs, because the rain is pluggable:
- `ImergRain`: the satellite only, for any box and dates;
- `DesignStorm`: a made-up storm, used by the tests;
- `Phase4Rain`: the tower maps.

## Steps

### 5.1 Ground: terrain, land cover, soil (`floodrisk.ground`)
- Download, crop and cache three layers:
  - Copernicus GLO-30 terrain (FABDEM bare-earth as an option);
  - ESA WorldCover 10 m land cover;
  - HYSOGs250m soil groups.
- Put them all on one 30 m grid per city.
- Domains:
  - **Algiers:** the whole Phase 4 city box (about 31 × 19 km), so the risk map covers the city. Koriche is the
    checked catchment.
  - **Ghardaïa:** the M'zab basin upstream of the town, which reaches outside the Phase 4 tower box. Its grid is
    60–90 m if the basin is too large for 30 m.
- Choice: a 30 m terrain map cannot see individual streets. In a city it is a *surface* model, so blocks of buildings
  act as rough bumps. We say so, and we test FABDEM where it is reachable.

### 5.2 Drainage network (`floodrisk.terrain`)
- Fill the pits, compute D8 flow directions and the upstream area of each cell, and draw each basin.
- Snap the outlets to the cell of largest upstream area near the named place: Koriche at Triolet, and the M'zab at
  Ghardaïa.
- Check: the Koriche area must come out near 10.2 km².

### 5.3 Losses: what soaks in (`floodrisk.runoff`)
- Each cell gets a curve number from WorldCover class × soil group (USDA TR-55 tables, as in GCN250).
- The rain excess is the cumulative SCS-CN form, applied every time step (Ia = 0.2 S).
- Antecedent soil moisture: AMC II by default, with AMC III as a sensitivity run (Algiers had rain before the flood).
- Sewers are unknown (SEAAL data is not public). Two cases: drains blocked (0 mm/h, as reported in 2001) and a nominal
  urban drain capacity, as a sensitivity run.

### 5.4 Moving the water (`floodrisk.solver`)
- A 2D local-inertial shallow-water solver (Bates 2010 / de Almeida 2012) in PyTorch, on the GPU when free, otherwise
  the CPU.
- Manning roughness comes from land cover.
- The time step is set by the wave speed (CFL 0.7).
- Outputs:
  - maximum depth and maximum speed in every cell;
  - flow at the outlets, every minute;
  - depth snapshots every 10–15 min for the website.
- Tests:
  - a lake at rest stays at rest;
  - mass is conserved (rain in = water stored + water out);
  - steady rain on a tilted plane gives outflow = rain × area.

### 5.5 Rain inputs: the link to Phase 4 (`floodrisk.rain`)
- Sources, all hourly at 500 m, regridded to the flood grid:
  - `truth`: the simulated real rain;
  - `ai`: the towers + AI;
  - `classical`: the towers with the classical method;
  - `ai_sparse`: towers + AI on a thin network;
  - `imerg`: the satellite alone, 10 km.
- Outside the tower box (the Ghardaïa basin) every source falls back to IMERG. We report this, because towers only
  help where towers are.
- Two scenarios:
  - **satellite storm:** as in Phase 4;
  - **gauge-matched storm:** every source multiplied by one factor, so the Koriche storm total = 263 mm (Bouzaréah) and
    the Ghardaïa basin two-day total = 80 mm (ANRH).

### 5.6 Hazard, risk and warnings (`floodrisk.hazard`)
- **Hazard per cell:** the DEFRA rating HR = d(v+0.5)+DF, in 4 classes (low / moderate / significant / extreme), and
  the Australian H1–H6 classes for reference.
- **Exposure:** WorldCover built-up cells. **Risk per commune:** the built-up area under significant+ hazard
  (geoBoundaries communes, as in Phase 4).
- **Alarm rule:**
  - A basin is "in flood" when its outlet flow passes a set level (the 10-year flood where it is known, otherwise
    the flow that first puts built-up cells under 0.3 m).
  - The alarm time for each rain source is compared with the time computed from the true rain.
  - Data delays are added: towers 5 min, satellite IMERG-Early 4 h.
- **Scores:**
  - peak flow and volume error;
  - flooded-area agreement against truth (critical success index);
  - commune risk ranking (Spearman);
  - alarm lead or lag in minutes.

### 5.7 Checks against the real floods
- **Algiers, gauge-matched:** the Koriche peak should be compared with 143 m³/s (flood marks), 90–140 m³/s (the
  plausible range) and 88.7 m³/s (Menad et al.'s model). The volume should be compared with 1.8 Mm³.
- **Ghardaïa, gauge-matched:** compare the peak with 900–1,600 m³/s. Our model has no dam break and no sediment, so it
  should come out on the low side.

### 5.8 Website (`site/flood/`, its own page)
- An animated flood depth over the satellite basemap (MapLibre), and a hazard map.
- Outlet flow over time for each rain source, with the alarm times marked.
- A commune risk table.
- It shows the same two storms as the replay and the 3D storm, but on its own page.

### 5.9 Paper (`part5.tex`)
- Same tone as Parts 1–4: every term explained, real numbers only, and the simulation clearly labelled.
- Sections: why rain is not yet a flood; the ground; losses; moving water; the two storms; does tower rain give the same
  flood?; warnings and lead time; what we could not check.
- New glossary entries.

## Data that needs a person (manual access)

| What | Why it matters | Status |
|---|---|---|
| ANRH / ONM gauge records and flood marks (Algiers 2001, Ghardaïa 2008) | Real check of rain and peak flow | Not public: an institutional request. Literature values used instead |
| SEAAL Algiers sewer network | Drains change small floods | Not public; handled as a sensitivity case |
| Charter / SERTIT flood maps, Ghardaïa 2008 | The only independent flood outline | Site timed out; may need a manual download |
| Google Earth Engine account (Global Flood Database) | MODIS flood outlines | Needs a login; low value for narrow urban floods |
| FABDEM (bare-earth terrain) | Removes buildings from the terrain | Server timed out; retry automatically, else manual download |
| Paywalled Ghardaïa hydrology papers (Springer) | Calibration numbers | The SNDL proxy in Chrome works for Springer if wanted |

## Order of work and commits
1. Plan and research (this file). 2. Package + tests. 3. Ground + drainage. 4. Runs. 5. Paper. 6. Website.

- Commit after each step.
- Commit code, tests, small json/csv results, paper sources and figures, and the site template and page.
- Never commit downloads, grids or run outputs (`data/` is git-ignored).
- The full test suite must pass before every commit.
