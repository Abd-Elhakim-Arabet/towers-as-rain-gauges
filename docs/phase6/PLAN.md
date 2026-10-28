# Phase 6 plan: the operations dashboard

One page that shows the whole chain on one 3D scene: **towers → rain → flood → alarm**. It is the demo for the
judges and the place where every earlier phase meets. Everything in it is still the Phase 4–5 **simulation**,
and the page says so.

## Separation (same rule as Phase 5)

| | Phase 6 owns | Reads from earlier phases (never writes) |
|---|---|---|
| Code | `scripts/phase6/`, `site/dashboard/template.html` | `floodrisk` (setup, solver), `results/phase5/*.json`, Phase 4 files via `floodrisk.rain.Phase4Rain` |
| Data | `data/processed/phase6/` (its own flood reruns), `site/dashboard/data/` (built, gitignored) | `data/processed/phase4/sim_*.npz`, `data/processed/algeria_sites.csv`, `results/phase4_links_*.csv` |
| Page | `/dashboard/` on the local site | nothing: the replay, 3D storm and flood pages stay as they are |

The only change outside Phase 6 is additive: `floodrisk.solver.Params.snap_flow` (default off) also saves the
water's speed and direction with each snapshot. Phase 5 runs and results are unchanged.

## What inspired the design

- **A 3D scene with a time slider and linked charts.** This is the common pattern in research flood viewers
  (FlowsDT digital twin, Sheffield and Exeter flood visualisation tools, Hydro3DJS): map, flow curve and time
  move together.
- **Water as a surface, not coloured squares.** Realistic-water renderers (ArcGIS `WaterSymbol3DLayer`, WebGL
  water demos) use depth-dependent colour (Beer–Lambert), Fresnel sky reflection, moving ripples and sun glints.
  Flood water is muddy, so ours is brown-green, not blue.
- **Focus plus context** (HORA 3D): a guided tour flies to the places that matter instead of leaving viewers in a
  30 km map.
- **Operations dashboards** (flash-flood guidance, Delft-FEWS, Google Flood Hub): one status (normal / watch /
  warning / flooding), the rain measured against the flash-flood threshold, lead time, and who is affected.

## Steps

1. **Flood reruns with full detail** (`scripts/phase6/run_dashboard_floods.py`). Gauge-matched storm, sources
   truth / AI / classical / satellite, the same inputs as Phase 5 but hourly snapshots at full 30 m resolution with
   water speed. Check: the basin outflow equals Phase 5's run.
2. **Export** (`scripts/phase6/export_dashboard.py`) to `site/dashboard/data/<city>/`:
   - terrain tiles made from the flood model's own DEM (Copernicus GLO-30), so the water sits exactly on the ground;
   - the ground height under the water, and one PNG per hour per source (depth plus flow direction);
   - hourly rain per source (Phase 4 maps on the flood grid, × the gauge factor), towers, links and link rain;
   - time series: basin rain, rain against the threshold, flow out of the basin, flooded built-up land, communes affected;
   - alarms and thresholds from `results/phase5/<city>.json`.
3. **Page** (`site/dashboard/template.html`): MapLibre GL + deck.gl.
   - Layers: satellite imagery, our terrain, OSM 3D buildings, towers, links, rain, and a custom water layer
     (GPU shader that blends hourly frames smoothly).
   - Panels: status banner and clock, rain-source switch, live figures, flow chart, communes affected.
   - A timeline with rain bars and alarm/flood markers, and a guided tour that walks through the chain.
4. **Wire-up**: `/dashboard/` on `scripts/serve_site.py`, nav link, gitignore the built files.
5. **Iterate on the look** with screenshots until nothing jars: alignment of water on the ground, colours, labels,
   panel overlap, phone width.
6. **Paper**: Part 6 (new `part6.tex`, only `\input` lines added to `explainer.tex`) with screenshots and what each
   panel means; README row.

## Honesty rules on the page

- A permanent "Simulation" badge in the header and an About box: real towers, terrain, buildings and storms;
  simulated link signals and simulated truth.
- Every number comes from the exported runs; nothing is typed in by hand except labels.
- The satellite source shows its 4-hour delay and its late alarm; Ghardaïa shows that towers cover 5% of the basin.
