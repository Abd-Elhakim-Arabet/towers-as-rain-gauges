# Environment

Checked 2026-10-14 on the lab machine: 24 CPU cores, 125 GB RAM, 2 × RTX 5090 (32 GB), Ubuntu, Python 3.12.3.

## Main venv `.venv`
- Created with `uv`. Exact package versions are in `requirements.lock.txt`.
- Key versions: torch 2.11.0+cu128, pycomlink 0.6.0, poligrain 0.3.1, pysteps 1.21.5,
  xarray 2026.9.0, numpy 2.5.4, lightgbm 4.7.0, scikit-learn 1.9.1.
- **Torch is borrowed, not installed**, because the network is slow (CUDA wheels are about 3 GB).
  `.venv/lib/python3.12/site-packages/zz_shared_torch.pth` adds
  `~/Desktop/pfe/o2vg-demo/.venv/lib/python3.12/site-packages` to the path. It is appended
  last, so this venv's own packages take precedence. On another machine, install torch normally (see README).

## MindSpore venv `.venv-ms`
- mindspore 2.10.0 (PyPI, cp312 manylinux, CPU), kept apart from the main stack.
- Locally the MindSpore models run on CPU. They are small 1D models, so this is enough for
  porting and checking that both frameworks give the same results. GPU/Ascend training is meant for ModelArts.
