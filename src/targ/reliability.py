"""Per-link reliability from the raw signal and the neighbours (no rain method involved).

Idea: real rain is shared. A link whose signal keeps dropping while no link around it
does is probably faulty. The score uses only the raw signal, so it does not favour
the classical method or the AI.

Every 15 minutes, per link:
  level    median total loss of the 15 minutes (min/max data: mid-point)
  excess   level minus the link's own dry level (20th percentile of the previous
           3 days of levels, as in targ.features)
  event    excess >= EVENT_DB

Per link:
  n_events            number of event blocks
  unconfirmed_share   share of its event blocks with no event on any of its
                      N_NEAREST nearest neighbours (other paths within 10 km)
                      within +-1 block
  event_share         share of its 15-min blocks that are events
Rule: drop if (n_events >= MIN_EVENTS and unconfirmed_share > MAX_UNCONFIRMED)
or event_share > MAX_EVENT_SHARE (it rains far less than 20% of the time, so a
link "in event" that often is drifting or broken). Links with too few events to
judge are kept.

Chosen on Sweden June-July only (scripts/phase2_reliability_sweden.py): first
"any neighbour within 10 km", which in a dense city (192 neighbours) confirms
almost anything by chance; then the 5 nearest plus the event-share rule. On that
calibration data the dropped links' raw signal follows the radar far worse
(median r 0.33) than the kept ones (0.58).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from scipy.ndimage import maximum_filter1d

from .features import low_reference

EVENT_DB = 1.5
MIN_EVENTS = 8
MAX_UNCONFIRMED = 0.5
MAX_EVENT_SHARE = 0.2
N_NEAREST = 5
BLOCK_MIN = 15


def levels_15min(tl_1min: np.ndarray) -> np.ndarray:
    """(n, T) 1-min total loss -> (n, T // 15) block medians."""
    tl = np.asarray(tl_1min, float)
    n, T = tl.shape
    nb = T // BLOCK_MIN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(tl[:, :nb * BLOCK_MIN].reshape(n, nb, BLOCK_MIN), axis=2)


def link_reliability(level: np.ndarray, A: sparse.csr_matrix, event_db: float = EVENT_DB,
                     min_events: int = MIN_EVENTS, max_unconfirmed: float = MAX_UNCONFIRMED,
                     max_event_share: float = MAX_EVENT_SHARE) -> dict:
    """level: (n, B) 15-min levels; A: neighbour matrix (use neighbours.nearest_matrix
    with k=N_NEAREST). Returns per-link arrays and the keep mask."""
    level = np.asarray(level, float)
    ex = level - low_reference(level, block=1, window=288)
    event = np.isfinite(ex) & (ex >= event_db)
    nb_event = maximum_filter1d(event.astype(np.float32), size=3, axis=1, mode="constant")
    supported = (A @ nb_event) > 0
    n_events = event.sum(axis=1)
    unconf = (event & ~supported).sum(axis=1)
    share = np.where(n_events > 0, unconf / np.maximum(n_events, 1), 0.0)
    judged = n_events >= min_events
    event_share = n_events / np.maximum(np.isfinite(level).sum(axis=1), 1)
    lonely = judged & (share > max_unconfirmed)
    always = event_share > max_event_share
    return {"n_events": n_events, "unconfirmed_share": share, "judged": judged,
            "event_share": event_share, "lonely": lonely, "always": always, "keep": ~(lonely | always),
            "rule": {"event_db": event_db, "min_events": min_events, "max_unconfirmed": max_unconfirmed,
                     "max_event_share": max_event_share, "n_nearest": N_NEAREST, "block_min": BLOCK_MIN}}


def score_links(ds, tl_1min=None, level_15=None) -> dict:
    """Convenience: reliability for an OpenSense dataset (give 1-min TL or 15-min levels)."""
    from .neighbours import nearest_matrix

    A = nearest_matrix(ds.site_0_lat.values, ds.site_0_lon.values,
                       ds.site_1_lat.values, ds.site_1_lon.values, k=N_NEAREST)
    lev = level_15 if level_15 is not None else levels_15min(tl_1min)
    return link_reliability(lev, A)
