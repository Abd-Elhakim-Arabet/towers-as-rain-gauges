"""The two cities: model domains, checked outlets and what was observed (literature values).

Observed values: docs/phase5/research.md (Menad et al. 2012 for Algiers; Charter / Springer 2021 / press for Ghardaia).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "flood"
PROC = ROOT / "data" / "processed" / "phase5"
RESULTS = ROOT / "results" / "phase5"
PHASE4 = ROOT / "data" / "processed" / "phase4"       # the only Phase 4 location Phase 5 reads
COMMUNES = ROOT / "data" / "raw" / "boundaries" / "dza_adm3_communes.geojson"   # geoBoundaries ADM3 (shared input)
ALARM_HA = 5.0              # "the town floods": this many hectares of built-up land under >= 0.3 m of water
LATENCY_MIN = {"truth": 0, "ai": 5, "classical": 5, "ai_sparse": 5, "classical_sparse": 5, "imerg": 240}
# minutes from rain falling to the rain map being available: towers report every minute (+ processing);
# IMERG Early Run is published about 4 hours after observation


@dataclass(frozen=True)
class Outlet:
    key: str
    name: str
    lat: float
    lon: float
    snap_m: float = 400.0               # look this far for the main channel
    area_km2_ref: float | None = None   # published basin area, to check the outline


@dataclass(frozen=True)
class City:
    key: str
    name: str
    box: tuple[float, float, float, float]     # (lat0, lon0, lat1, lon1) of the flood model
    agg: int                                   # 1 = native 1-arc-second (~30 m), 2 = ~60 m
    dem_tiles: tuple[str, ...]
    outlets: tuple[Outlet, ...]
    gauge_total_mm: float                      # the storm total we match in the gauge-matched scenario
    gauge_note: str
    observed: dict = field(default_factory=dict)
    spin_dry_h: int = 6                        # dry hours added after the rain, so the flood can drain
    alarm_communes: tuple[str, ...] = ()       # geoBoundaries names of the town the alarm is about
    ffg_hours: tuple[int, ...] = (1, 3, 6, 12, 24)   # rain durations for the flash-flood rain thresholds


CITIES = {
    "algiers": City(
        key="algiers", name="Algiers",
        box=(36.65, 2.90, 36.82, 3.25), agg=1,
        dem_tiles=("N36_00_E002_00", "N36_00_E003_00"),
        outlets=(Outlet("koriche", "Oued Koriche at Triolet (Bab El Oued)", 36.7896, 3.0504, 150, 10.2),),
        gauge_total_mm=263.0,
        gauge_note="Bouzareah, 19:30 9 Nov to 10:30 10 Nov 2001 (Menad et al. 2012)",
        alarm_communes=("Bab Al Wadi",),
        observed={"koriche": {"peak_m3s": 143.0, "peak_range_m3s": (90.0, 140.0), "peak_model_m3s": 88.7,
                              "volume_Mm3": 1.8,
                              "src": "Menad et al. 2012: flood marks 143 m3/s, 1.8 Mm3; plausible 9-14 m3/s/km2; "
                                     "their RuiCells model 88.7 m3/s"}},
    ),
    "ghardaia": City(
        key="ghardaia", name="Ghardaïa",
        box=(32.40, 2.98, 32.82, 3.80), agg=2,
        dem_tiles=("N32_00_E002_00", "N32_00_E003_00"),
        outlets=(Outlet("mzab", "Oued M'zab at Ghardaïa", 32.4892, 3.6775, 200, None),),
        gauge_total_mm=80.0,
        gauge_note="Ghardaïa, two-day total 29 Sep - 1 Oct 2008 (ANRH, via Springer 2021)",
        alarm_communes=("Ghardaia", "Bounoura", "El Atteuf"), ffg_hours=(3, 6, 12, 24),
        observed={"mzab": {"peak_m3s": 1200.0, "peak_range_m3s": (900.0, 1600.0),
                           "src": "900 (press, all tributaries), 1,200, 1,600 m3/s (Springer 2021); "
                                  "10-yr 205, 100-yr 722 m3/s; a dam break upstream added >900 m3/s"}},
        spin_dry_h=12,
    ),
}
