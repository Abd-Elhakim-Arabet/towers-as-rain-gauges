"""City name -> a short list of places to choose from (OpenStreetMap Nominatim).

A name alone is ambiguous ("Tunis" also matches the country), so the caller shows the candidates and the user picks.
"""

from __future__ import annotations

import requests

API = "https://nominatim.openstreetmap.org/search"
UA = "towers-as-rain-gauges/0.7 (flood watch research demo)"
MAX_HALF_KM = 15.0          # the watched box is at most 30 x 30 km
MIN_HALF_KM = 4.0
PLACES = {"city", "town", "village", "municipality", "suburb", "borough", "administrative", "hamlet", "county"}


def search(name: str, limit: int = 6) -> list[dict]:
    """Candidates: name, lat, lon and the half-size (km) of the box we would watch."""
    r = requests.get(API, params={"q": name, "format": "jsonv2", "limit": 12, "accept-language": "en",
                                  "addressdetails": 0}, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    out, big = [], []
    for d in r.json():
        if d.get("category") not in ("place", "boundary") or d.get("type") not in PLACES:
            continue
        if d.get("addresstype") == "country" or int(d.get("place_rank") or 30) <= 4:
            continue                            # "Tunis" also matches Tunisia: a country is never a place to watch
        lat, lon = float(d["lat"]), float(d["lon"])
        s, n, w, e = map(float, d["boundingbox"])
        half_km = max((n - s) * 111.2, (e - w) * 111.2 * _cos(lat)) / 2
        c = {"name": d["display_name"], "lat": lat, "lon": lon, "kind": d.get("addresstype") or d["type"],
             "half_km": round(min(max(half_km, MIN_HALF_KM), MAX_HALF_KM), 1)}
        # Very large areas (a state, or Tokyo, whose outline reaches islands 1,000 km out) are kept only when nothing
        # smaller matches; then we watch the 30 km around OpenStreetMap's point for it, which is its centre.
        (big if half_km > 150 else out).append(c)
    return (out or big)[:limit]


def _cos(lat: float) -> float:
    import math
    return math.cos(math.radians(lat))


def box(lat: float, lon: float, half_km: float) -> tuple[float, float, float, float]:
    """(lat0, lon0, lat1, lon1), snapped to whole arc-seconds so the flood grid lines up with the terrain data."""
    half_km = min(max(half_km, MIN_HALF_KM), MAX_HALF_KM)
    dlat, dlon = half_km / 111.2, half_km / (111.2 * _cos(lat))
    snap = lambda v: round(v * 3600) / 3600       # noqa: E731
    return snap(lat - dlat), snap(lon - dlon), snap(lat + dlat), snap(lon + dlon)
