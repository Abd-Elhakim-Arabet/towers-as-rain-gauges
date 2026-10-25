# Towers as Rain Gauges

Flash-flood early warning from the microwave links that already connect cell towers.
Rain attenuates the signal between two towers, and operators already log the received
and transmitted signal levels (RSL/TSL). So every link is an accidental rain gauge.
AI turns those signals into a live rain map, and a flood-risk layer tells you which
neighborhoods and roads are most exposed in the next hour.

Built for the Huawei ICT Competition (Innovation Track).

> **Honesty note.** Validation uses **real European data** (Sweden, Italy, Netherlands).
> The **Algeria part is a simulation**: real tower positions and real satellite rainfall
> drive a *virtual* link network. It shows how the system would work there. It does not prove
> accuracy in Algeria. No Algerian operator data was used.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Repo, environment, data downloads | done (NL: June–Aug 2014 subset; KNMI hourly radar 2014) |
| 1 | Classical baseline on OpenMRG | done — see [results/phase1_openmrg.md](results/phase1_openmrg.md) |
| 2 | AI models (causal TCN, physics head; PyTorch + MindSpore twin), cross-domain tests on Italy and the Netherlands | done — see [results/phase2.md](results/phase2.md) |
| 3 | Rain maps (IDW) + nowcasting (pysteps) | done — see [results/phase3_maps.json](results/phase3_maps.json), [results/phase3_nowcast.json](results/phase3_nowcast.json) |
| 4 | Virtual Algerian network (simulation) | done — see [results/phase4.md](results/phase4.md), explainer Part 4, interactive replay in [site/phase4/index.html](site/phase4/index.html) |
| 5 | Flood-risk layer: 2D flood model (own package `src/floodrisk/`) on the Phase 4 storms, hazard, alarms (simulation) | done — see [results/phase5/phase5.md](results/phase5/phase5.md), explainer Part 5, plan in [docs/phase5/PLAN.md](docs/phase5/PLAN.md), local page `/flood/` |
| 6 | Dashboard | – |
| 7 | Report, pitch, video script | – |

## Layout

```
configs/      datasets.yaml (data registry), backend.yaml (local ↔ Huawei switch)
scripts/      download_data.py, download_nl_subset.py, phase entry points
src/targ/     the Python package
  backends/   framework (torch|mindspore), storage (local|obs), compute (local|modelarts)
data/raw/     downloaded data (git-ignored) + MANIFEST.json
docs/         DATA.md, ENVIRONMENT.md, report
tests/
```

## Local first, Huawei when available

The team has no Huawei Cloud access yet, so every Huawei-specific piece sits behind a
backend with a local default:

| Layer | Local (now) | Huawei (later) | Switch |
|---|---|---|---|
| Deep-learning framework | PyTorch | MindSpore | `framework:` in `configs/backend.yaml` or `TARG_FRAMEWORK` |
| Storage | local disk | OBS | `storage:` / `TARG_STORAGE` |
| Training | local GPU/CPU | ModelArts training job | `compute:` / `TARG_COMPUTE` |

Code outside `src/targ/backends/` never imports torch, MindSpore or Huawei SDKs directly.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e .
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
# optional MindSpore (separate venv, CPU build locally):
uv venv --python 3.12 .venv-ms && uv pip install --python .venv-ms/bin/python -r requirements-mindspore.txt -e .

.venv/bin/python scripts/download_data.py openmrg openrainer netherlands_cml
.venv/bin/python -m pytest
.venv/bin/python scripts/phase1_baseline.py   # Phase 1, ~5 min (first run builds a 1-min cache, +6 min)
.venv/bin/python scripts/phase2_openmrg.py     # Phase 2 Sweden: train + score (~10 min on a GPU)
.venv/bin/python scripts/phase2_seeds.py       # Phase 2: 5 retrainings, spread of the scores
.venv/bin/python scripts/phase2_italy.py       # Phase 2: zero-shot test on OpenRainER (Italy)
.venv/bin/python scripts/phase2_netherlands.py # Phase 2: min/max model, zero-shot test on the Netherlands
.venv/bin/python scripts/phase2_mindspore.py export && .venv-ms/bin/python scripts/phase2_mindspore.py check
.venv/bin/python scripts/phase2_diagnose_seeds.py     # why runs disagree on amounts
.venv/bin/python scripts/phase2_diagnose_longrain.py  # why long rain is under-measured in Italy
.venv/bin/python scripts/phase2_mixed.py se_mse mix21_mse mix21rel_mse   # varied-training experiments
.venv/bin/python scripts/phase2_italy_cv.py train 5,6,7,8,9 0,1,2 mse-rel && .venv/bin/python scripts/phase2_italy_cv.py eval mse-rel
.venv/bin/python scripts/phase2_mixed.py "mix21rel+mfb+mask+gt_mse"   # final recipe (flags: mfb, mask, gt, rf6, west/east)
.venv/bin/python scripts/phase2_mixed.py "mix21rel+mfb+mask+gt+west_mse" "mix21rel+mfb+mask+gt+east_mse"   # new-links test
.venv/bin/python scripts/phase2_italy_cv.py train 5,6,7,8,9 0,1,2 mse-rel+mfb+mask+gt && .venv/bin/python scripts/phase2_italy_cv.py eval mse-rel+mfb+mask+gt
.venv/bin/python scripts/export_explainer_varied.py && .venv/bin/python scripts/export_explainer_varied.py numbers
.venv/bin/python scripts/phase2_summary.py     # results/phase2.md
.venv/bin/python scripts/phase3_maps.py       # Phase 3: link rain maps vs gauges / radar
.venv/bin/python scripts/phase3_nowcast.py    # Phase 3: 0-60 min nowcasts, Netherlands
.venv/bin/python scripts/phase4_storms.py     # Phase 4: IMERG storms (Algiers 2001, Ghardaia 2008), needs EARTHDATA_TOKEN
.venv/bin/python scripts/phase4_rain_texture.py  # Phase 4: rain patchiness measured on Italian radar
.venv/bin/python scripts/phase4_simulate.py   # Phase 4: simulated network -> signals -> AI/classical -> maps, scores
.venv/bin/python scripts/export_explainer_phase4.py && .venv/bin/python scripts/export_phase4_explorer.py  # figures + replay page
.venv/bin/python scripts/export_storm3d.py  # 3D storm (satellite, terrain, 3D buildings, towers, links, rain; live tiles) + GeoJSON for GeoLibre
.venv/bin/python scripts/serve_site.py       # explainer PDF, storm replay, 3D storm on http://localhost:8000
```

Exact versions are in `requirements.lock.txt`. See `docs/ENVIRONMENT.md` for machine-specific notes.

## Data

See [docs/DATA.md](docs/DATA.md) for sources, versions, licenses and sizes.
