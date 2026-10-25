"""Drainage from a terrain grid: pit filling, D8 flow directions, upstream area, basins.

Grids are (rows north->south, cols west->east). Cell sizes dx (east-west) and dy (north-south) in metres.
"""

from __future__ import annotations

import heapq

import numpy as np
from numba import njit

# D8 neighbours: E, SE, S, SW, W, NW, N, NE
DR = np.array([0, 1, 1, 1, 0, -1, -1, -1], np.int64)
DC = np.array([1, 1, 0, -1, -1, -1, 0, 1], np.int64)


@njit(cache=True)
def _fill(z, sink, eps):
    """Priority-flood (Barnes et al. 2014) with a tiny gradient, so every cell drains to the edge or a sink.

    sink: cells where water leaves the grid (sea); edge cells drain off the grid too.
    """
    ny, nx = z.shape
    out = z.copy()
    done = np.zeros((ny, nx), np.bool_)
    heap = [(0.0, 0, 0)]
    heap.pop()
    for r in range(ny):
        for c in range(nx):
            if sink[r, c] or r == 0 or c == 0 or r == ny - 1 or c == nx - 1:
                done[r, c] = True
                heapq.heappush(heap, (out[r, c], r, c))
    while len(heap) > 0:
        e, r, c = heapq.heappop(heap)
        for k in range(8):
            rr = r + DR[k]
            cc = c + DC[k]
            if rr < 0 or cc < 0 or rr >= ny or cc >= nx or done[rr, cc]:
                continue
            done[rr, cc] = True
            if out[rr, cc] <= e:
                out[rr, cc] = e + eps
            heapq.heappush(heap, (out[rr, cc], rr, cc))
    return out


def fill(z: np.ndarray, sink: np.ndarray | None = None, eps: float = 1e-4) -> np.ndarray:
    z = np.asarray(z, np.float64)
    sink = np.zeros(z.shape, bool) if sink is None else np.asarray(sink, bool)
    return _fill(z, sink, eps)


@njit(cache=True)
def _d8(zf, dx, dy, sink):
    ny, nx = zf.shape
    d = np.full((ny, nx), -1, np.int64)
    diag = np.sqrt(dx * dx + dy * dy)
    for r in range(ny):
        for c in range(nx):
            if sink[r, c]:
                continue
            best = 0.0
            for k in range(8):
                rr = r + DR[k]
                cc = c + DC[k]
                if rr < 0 or cc < 0 or rr >= ny or cc >= nx:
                    continue
                dist = dx if DR[k] == 0 else (dy if DC[k] == 0 else diag)
                s = (zf[r, c] - zf[rr, cc]) / dist
                if s > best:
                    best = s
                    d[r, c] = k
    return d


def d8(zf: np.ndarray, dx: float, dy: float, sink: np.ndarray | None = None) -> np.ndarray:
    """Steepest-descent direction 0..7 (see DR/DC), -1 = drains off the grid or into a sink."""
    sink = np.zeros(zf.shape, bool) if sink is None else np.asarray(sink, bool)
    return _d8(np.asarray(zf, np.float64), float(dx), float(dy), sink)


@njit(cache=True)
def _accumulate(order, d, w):
    ny, nx = d.shape
    acc = w.copy()
    for i in range(order.size):          # high to low: each cell passes its total downhill
        r = order[i] // nx
        c = order[i] % nx
        k = d[r, c]
        if k >= 0:
            acc[r + DR[k], c + DC[k]] += acc[r, c]
    return acc


def accumulate(zf: np.ndarray, d: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Upstream total of `weight` (default 1 per cell) for every cell."""
    w = np.ones(zf.shape) if weight is None else np.asarray(weight, np.float64)
    order = np.argsort(-zf, axis=None, kind="stable")
    return _accumulate(order, d, w)


@njit(cache=True)
def _basin(order, d, r0, c0):
    ny, nx = d.shape
    m = np.zeros((ny, nx), np.bool_)
    m[r0, c0] = True
    for i in range(order.size):          # low to high: a cell joins if its downhill neighbour has joined
        r = order[i] // nx
        c = order[i] % nx
        k = d[r, c]
        if k >= 0 and m[r + DR[k], c + DC[k]]:
            m[r, c] = True
    return m


def basin(zf: np.ndarray, d: np.ndarray, r0: int, c0: int) -> np.ndarray:
    """Mask of every cell that drains through (r0, c0)."""
    order = np.argsort(zf, axis=None, kind="stable")
    return _basin(order, d, int(r0), int(c0))


def snap(acc: np.ndarray, r: int, c: int, radius: int) -> tuple[int, int]:
    """The cell with the largest upstream area within `radius` cells of (r, c)."""
    r0, r1 = max(r - radius, 0), min(r + radius + 1, acc.shape[0])
    c0, c1 = max(c - radius, 0), min(c + radius + 1, acc.shape[1])
    k = np.argmax(acc[r0:r1, c0:c1])
    return r0 + k // (c1 - c0), c0 + k % (c1 - c0)


def condition(z: np.ndarray, sea: np.ndarray, keep: np.ndarray | None = None, max_fill_m: float | None = None):
    """Terrain for the flood model: fill the pits of a 30 m surface model (mostly artefacts between buildings and
    under bridges), except where `keep` is True (mapped lakes and reservoirs). max_fill_m caps the filling (None =
    fill completely, 0 = leave the terrain as it is). Returns the conditioned terrain and the filled depth."""
    if max_fill_m == 0:
        return np.asarray(z, np.float64), np.zeros(z.shape)
    zf = fill(z, sea, eps=1e-4)
    add = zf - z
    if keep is not None:
        add = np.where(keep, 0.0, add)
    if max_fill_m is not None:
        add = np.minimum(add, max_fill_m)
    return z + add, add
