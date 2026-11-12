"""The watched area: a box around a city, with its terrain, soil and land cover made ready for the flood model."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from floodrisk import ground, hazard, runoff, terrain

from . import geocode

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "data" / "live"           # one folder per watched city; never committed
MAX_CELLS = 800_000                     # above this the grid goes from ~30 m to ~60 m cells
ALARM_HA = 5.0                          # "the town floods": at least this much built-up land under 0.3 m ...
ALARM_SHARE = 0.01                      # ... and at least this share of all built-up land in the box


@dataclass(frozen=True)
class Area:
    key: str
    name: str
    lat: float
    lon: float
    box: tuple[float, float, float, float]      # lat0, lon0, lat1, lon1
    agg: int                                    # 1 = ~30 m cells, 2 = ~60 m

    @property
    def dir(self) -> Path:
        return LIVE / self.key

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "area.json").write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def load(key: str) -> "Area":
        d = json.loads((LIVE / key / "area.json").read_text())
        return Area(**{**d, "box": tuple(d["box"])})


def slug(name: str) -> str:
    s = unicodedata.normalize("NFKD", name.split(",")[0]).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "city"


def make(name: str, lat: float, lon: float, half_km: float) -> Area:
    box = geocode.box(lat, lon, half_km)
    ny, nx = (box[2] - box[0]) * 3600, (box[3] - box[1]) * 3600
    return Area(f"{slug(name)}-{lat:.2f}-{lon:.2f}".replace(".", "_"), name, lat, lon, box, 1 if ny * nx <= MAX_CELLS else 2)


def prepare(area: Area) -> dict:
    """Everything the flood model needs for this box. Downloads on first use (a few minutes), then cached."""
    g = ground.build_box(area.key, area.box, area.agg, cache=area.dir / "ground.npz")
    sea = g.sea
    lake = (g.landcover == 80) & ~sea
    z, _ = terrain.condition(g.z, sea, keep=lake)
    cell = g.dx * g.dy
    built_ha = hazard.built_area_ha(g.built, cell)
    town = (g.built > 0.2) & ~sea
    alarm_ha = max(ALARM_HA, ALARM_SHARE * float(built_ha[town].sum()))
    # does a large stream enter the box from outside? then part of the flood comes from where we cannot see
    zf = terrain.fill(g.z, sea)
    acc = terrain.accumulate(zf, terrain.d8(zf, g.dx, g.dy, sea)) * cell / 1e6
    edge = np.zeros(z.shape, bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    box_km2 = z.size * cell / 1e6
    return {"g": g, "sea": sea, "z": z, "cell": cell, "built_ha": built_ha, "town": town, "alarm_ha": alarm_ha,
            "cn": runoff.curve_numbers(g.landcover, g.hsg), "n": runoff.manning(g.landcover),
            "acc_km2": acc, "box_km2": box_km2, "biggest_stream_km2": float(acc[~sea].max()),
            "edge_note": bool((g.z[edge & ~sea] > g.z[~sea].mean()).mean() > 0.3)}
