# Reproduce the results

The quick install in [SETUP.md](SETUP.md) is enough to look at the pages and run the live watch. To rebuild every
number in the explainer from the raw data, run the steps below in order. You need about 60 GB of disk for the raw
datasets ([DATA.md](DATA.md) lists sources, licences and sizes), free API keys in `.env` for some sources
(`OPENCELLID_TOKEN`, `EARTHDATA_TOKEN`, `KNMI_API_KEY`), and ideally a GPU.

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

Exact versions are in `requirements.lock.txt`. See [ENVIRONMENT.md](ENVIRONMENT.md) for machine-specific notes.

## Where each step's results live

| Step | Results |
|---|---|
| Classical baseline (Sweden) | [results/phase1_openmrg.md](../results/phase1_openmrg.md) |
| AI model, tests in Sweden, Italy and the Netherlands | [results/phase2.md](../results/phase2.md) |
| Rain maps and nowcasting | [results/phase3_maps.json](../results/phase3_maps.json), [results/phase3_nowcast.json](../results/phase3_nowcast.json) |
| Simulated Algerian network | [results/phase4.md](../results/phase4.md) |
| Flood risk | [results/phase5/phase5.md](../results/phase5/phase5.md), plan in [phase5/PLAN.md](phase5/PLAN.md) |
| Control room | plan in [phase6/PLAN.md](phase6/PLAN.md); build with `scripts/phase6/run_dashboard_floods.py` then `scripts/phase6/export_dashboard.py` |
| Live watch | plan and results in [phase7/PLAN.md](phase7/PLAN.md) |
