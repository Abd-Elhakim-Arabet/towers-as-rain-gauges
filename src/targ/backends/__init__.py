"""Swappable backends: framework, storage and compute layers.

Everything outside this package asks for a backend through these functions and
never imports torch/mindspore/OBS/ModelArts SDKs directly. That keeps the
Huawei switch a one-line config change (configs/backend.yaml or env vars).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


@lru_cache
def config() -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "backend.yaml").read_text())
    for layer in ("framework", "storage", "compute"):
        cfg[layer] = os.environ.get(f"TARG_{layer.upper()}", cfg[layer])
    return cfg


def get_framework() -> str:
    """Name of the deep-learning framework in use: 'torch' or 'mindspore'."""
    fw = config()["framework"]
    if fw not in ("torch", "mindspore"):
        raise ValueError(f"unknown framework {fw!r}")
    return fw


def get_storage():
    from .storage import LocalStorage, OBSStorage

    kind = config()["storage"]
    if kind == "local":
        return LocalStorage(ROOT / config()["local"]["root"])
    if kind == "obs":
        return OBSStorage(**config()["obs"])
    raise ValueError(f"unknown storage {kind!r}")


def get_compute():
    from .compute import LocalRunner, ModelArtsRunner

    kind = config()["compute"]
    if kind == "local":
        return LocalRunner(ROOT)
    if kind == "modelarts":
        return ModelArtsRunner(**config()["modelarts"])
    raise ValueError(f"unknown compute {kind!r}")
