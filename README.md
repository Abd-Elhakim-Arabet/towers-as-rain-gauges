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
| 0 | Repo, environment, data downloads | done (NL subset pending: 4TU blocking us) |
| 1 | Classical baseline on OpenMRG | – |
| 2 | AI wet/dry + rain-rate models, cross-domain tests | – |
| 3 | Rain maps + nowcasting | – |
| 4 | Virtual Algerian network (simulation) | – |
| 5 | Flood-risk layer | – |
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
```

Exact versions are in `requirements.lock.txt`. See `docs/ENVIRONMENT.md` for machine-specific notes.

## Data

See [docs/DATA.md](docs/DATA.md) for sources, versions, licenses and sizes.
