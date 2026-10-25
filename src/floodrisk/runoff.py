"""What soaks in and how rough the ground is.

Losses: the SCS curve-number method (USDA TR-55), cumulative form, one curve number (CN) per cell from WorldCover
class x hydrologic soil group, as GCN250 (Jaafar et al. 2019) does with coarser land cover.
    S = 25400/CN - 254  [mm],  Ia = 0.2 S,  Q = (P - Ia)^2 / (P - Ia + S) for P > Ia
P is the rain fallen on the cell since the start; the rain that runs off in a time step is the growth of Q.
Roughness: Manning's n per WorldCover class.
"""

from __future__ import annotations

import numpy as np

# WorldCover class -> CN for soil groups A, B, C, D (antecedent moisture II, TR-55 / GCN250 pairing)
CN_TABLE = {
    10: (30, 55, 70, 77),    # tree cover            ~ woods, good
    20: (35, 56, 70, 77),    # shrubland             ~ brush, fair
    30: (49, 69, 79, 84),    # grassland             ~ pasture, fair
    40: (67, 78, 85, 89),    # cropland              ~ row crops, straight, good
    50: (89, 92, 94, 95),    # built-up              ~ commercial/business, 85% impervious
    60: (77, 86, 91, 94),    # bare / sparse         ~ fallow, bare soil
    70: (98, 98, 98, 98),    # snow and ice
    80: (100, 100, 100, 100),  # open water
    90: (85, 85, 85, 85),    # herbaceous wetland
    95: (85, 85, 85, 85),    # mangroves
    100: (77, 86, 91, 94),   # moss and lichen       ~ bare
}
# Manning's n per class. Built-up is high on purpose: on a 30 m grid the buildings themselves are not resolved,
# so their drag goes into the roughness (a common choice for coarse urban rain-on-grid; tested as a sensitivity).
MANNING = {10: 0.10, 20: 0.06, 30: 0.035, 40: 0.04, 50: 0.05, 60: 0.03, 70: 0.03, 80: 0.03, 90: 0.06, 95: 0.08,
           100: 0.03}


def hsg_index(hsg: np.ndarray, dual: str = "D") -> np.ndarray:
    """HYSOGs code -> 0..3 (A..D). Dual groups (A/D...) are 'D when undrained': we take D by default."""
    h = np.asarray(hsg).astype(int)
    out = np.full(h.shape, 2, int)                      # no data (255) -> C
    for code, i in ((1, 0), (2, 1), (3, 2), (4, 3)):
        out[h == code] = i
    for code, i in ((11, 0), (12, 1), (13, 2), (14, 3)):
        out[h == code] = 3 if dual == "D" else i
    return out


def curve_numbers(landcover: np.ndarray, hsg: np.ndarray, amc: int = 2, dual: str = "D") -> np.ndarray:
    lc = np.asarray(landcover).astype(int)
    hi = hsg_index(hsg, dual)
    cn = np.full(lc.shape, 85.0)
    for k, row in CN_TABLE.items():
        m = lc == k
        cn[m] = np.asarray(row, float)[hi[m]]
    if amc == 3:                                         # wet soil (Chow et al. 1988)
        cn = 23 * cn / (10 + 0.13 * cn)
    elif amc == 1:                                       # dry soil
        cn = 4.2 * cn / (10 - 0.058 * cn)
    return np.clip(cn, 30, 100)


def manning(landcover: np.ndarray, built_n: float | None = None) -> np.ndarray:
    lc = np.asarray(landcover).astype(int)
    n = np.full(lc.shape, 0.04)
    for k, v in MANNING.items():
        n[lc == k] = built_n if (k == 50 and built_n is not None) else v
    return n


def scs_runoff(p_cum_mm, cn, ia_ratio: float = 0.2):
    """Cumulative runoff Q [mm] for cumulative rain P [mm] (works on numpy arrays or torch tensors)."""
    s = 25400.0 / cn - 254.0
    ia = ia_ratio * s
    pe = (p_cum_mm - ia).clip(min=0) if hasattr(p_cum_mm, "clip") else np.maximum(p_cum_mm - ia, 0)
    return pe * pe / (pe + s + 1e-9)
