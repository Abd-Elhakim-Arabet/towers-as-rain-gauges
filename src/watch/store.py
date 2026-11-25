"""Small files the Live watch page reads: JSON written in one move, and the two map pictures."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

RAIN_STOPS = [(0.2, (150, 210, 255, 0)), (1, (120, 190, 250, 110)), (5, (70, 150, 240, 170)), (15, (250, 220, 80, 200)),
              (40, (245, 130, 50, 225)), (100, (200, 40, 60, 240))]          # mm/h -> colour, as in the replay
DEPTH_STOPS = [(0.03, (120, 190, 255, 0)), (0.1, (110, 180, 250, 150)), (0.3, (50, 120, 240, 205)),
               (1.0, (25, 60, 190, 230)), (3.0, (20, 20, 110, 245))]         # metres of water


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":"), default=float), encoding="utf-8")
    os.replace(tmp, path)


def _colour(v: np.ndarray, stops, log: bool = True) -> np.ndarray:
    x = np.array([s[0] for s in stops], float)
    c = np.array([s[1] for s in stops], float)
    xv = np.log(np.maximum(v, 1e-6)) if log else v
    xs = np.log(x) if log else x
    out = np.stack([np.interp(xv, xs, c[:, k]) for k in range(4)], -1)
    out[v < x[0]] = 0
    return out.astype(np.uint8)


def _png(path: Path, rgba: np.ndarray) -> None:
    from PIL import Image
    tmp = path.with_suffix(".tmp.png")
    Image.fromarray(rgba, "RGBA").save(tmp, optimize=False)
    os.replace(tmp, path)


def rain_png(path: Path, mmh_north_up: np.ndarray) -> None:
    _png(path, _colour(np.nan_to_num(mmh_north_up), RAIN_STOPS))


def depth_png(path: Path, h: np.ndarray, sea: np.ndarray) -> None:
    _png(path, _colour(np.where(sea, 0.0, h), DEPTH_STOPS))
