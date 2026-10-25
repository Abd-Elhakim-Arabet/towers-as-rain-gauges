"""2D rain-on-grid flood solver: the local-inertial shallow-water equations (Bates et al. 2010; de Almeida et al. 2012).

Rain falls on every cell; the part that does not soak in (SCS curve number, cumulative) becomes water on the ground;
water moves between neighbouring cells, pushed by the slope of the water surface and held back by Manning friction:

    q_new = (q - g h_f dt dη/dx) / (1 + g dt n² |q| / h_f^(7/3))        (per cell face, unit width)
    h    += dt (Σ q_in - Σ q_out) / cell width + runoff

h_f = water depth available at the face (higher water surface minus higher ground). Explicit; the time step follows
the wave speed: dt = cfl · min(dx, dy) / sqrt(g h_max).
Two safety rules, both standard: a cell cannot send out more water than it holds (fluxes scaled down), and the speed is
held to Froude <= froude_max, because the inertial form is unstable in fast, steep flow.
Edges: where the ground runs down towards the edge, water leaves at normal depth (Manning with that slope);
flat or rising edges are walls. Sea cells absorb whatever reaches them.
Bookkeeping: every drop is counted (rain, soaked in, left through the edges, reached the sea, still on the ground).
Runs in PyTorch on the GPU if there is one, otherwise on the CPU.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .runoff import scs_runoff

G = 9.81


@dataclass
class Params:
    cfl: float = 0.7
    froude_max: float = 1.0
    h_dry: float = 1e-3           # thinner than this, no flow across a face (m)
    dt_max: float = 2.0           # larger steps make thin sheet flow flicker between steps
    dt_min: float = 0.02
    record_s: float = 60.0        # basin outflow: mean over each interval
    snap_s: float = 900.0         # depth snapshots for the website
    snap_coarsen: int = 2
    ia_ratio: float = 0.2
    device: str | None = None


class HourlyRain:
    """Rain rate fields (mm/h) on the flood grid, one per `step_s`, held constant within each step."""

    def __init__(self, fields_mmh: np.ndarray, step_s: float = 3600.0, start=None):
        self.fields = np.asarray(fields_mmh, np.float32)
        self.step_s = float(step_s)
        self.start = start

    @property
    def duration_s(self) -> float:
        return self.fields.shape[0] * self.step_s

    def index(self, t: float) -> int:
        return int(t // self.step_s)

    def field(self, i: int) -> np.ndarray | None:
        return self.fields[i] if i < self.fields.shape[0] else None


@dataclass
class Result:
    t_record: np.ndarray                       # s since start (end of each interval)
    q_gauge: dict[str, np.ndarray]             # mean outflow of each basin over the interval, m3/s
    vol_gauge: dict[str, float]                # total volume out of each basin, m3
    hmax: np.ndarray
    vmax: np.ndarray
    hrmax: np.ndarray                          # max hazard rating d(v+0.5)+DF
    t_wet: np.ndarray                          # first time depth >= 0.3 m (s), inf = never
    snaps: np.ndarray                          # (n, ny/c, nx/c) float16 depth
    t_snaps: np.ndarray
    budget: dict[str, float] = field(default_factory=dict)
    steps: int = 0
    wall_s: float = 0.0


def _dev(p: Params):
    if p.device:
        return torch.device(p.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simulate(z, n, cn, dx, dy, rain: HourlyRain, t_end_s: float, sink=None, gauges=None,
             params: Params | None = None, drain_mmh=None, h0=None, progress: bool = False) -> Result:
    p = params or Params()
    dev = _dev(p)
    f32 = dict(device=dev, dtype=torch.float32)
    Z = torch.as_tensor(np.asarray(z, np.float32), **f32)
    N = torch.as_tensor(np.asarray(n, np.float32), **f32)
    CN = torch.as_tensor(np.asarray(cn, np.float32), **f32)
    ny, nx = Z.shape
    SINK = torch.zeros((ny, nx), dtype=torch.bool, device=dev) if sink is None else \
        torch.as_tensor(np.asarray(sink, bool), device=dev)
    DRAIN = None if drain_mmh is None else torch.as_tensor(np.asarray(drain_mmh, np.float32), **f32) / 3.6e6
    H = torch.zeros((ny, nx), **f32) if h0 is None else torch.as_tensor(np.asarray(h0, np.float32), **f32)
    stored0 = float(H.double().sum() * dx * dy)
    QX = torch.zeros((ny, nx - 1), **f32)       # east-going flux across vertical faces, m2/s
    QY = torch.zeros((ny - 1, nx), **f32)       # south-going flux across horizontal faces
    NX2 = (0.5 * (N[:, 1:] + N[:, :-1])) ** 2
    NY2 = (0.5 * (N[1:, :] + N[:-1, :])) ** 2
    ZX = torch.maximum(Z[:, 1:], Z[:, :-1])
    ZY = torch.maximum(Z[1:, :], Z[:-1, :])
    # edge outflow where the ground runs down towards the edge; a flat or rising edge is a wall
    s_w = ((Z[:, 1] - Z[:, 0]) / dx).clamp(min=0)
    s_e = ((Z[:, -2] - Z[:, -1]) / dx).clamp(min=0)
    s_n = ((Z[1, :] - Z[0, :]) / dy).clamp(min=0)
    s_s = ((Z[-2, :] - Z[-1, :]) / dy).clamp(min=0)
    PC = torch.zeros((ny, nx), **f32)           # rain fallen so far, mm
    QC = torch.zeros((ny, nx), **f32)           # of which run off, mm
    HMAX = torch.zeros((ny, nx), **f32)
    VMAX = torch.zeros((ny, nx), **f32)
    HRMAX = torch.zeros((ny, nx), **f32)
    TWET = torch.full((ny, nx), float("inf"), **f32)
    gauges = gauges or {}
    gsx, gsy = {}, {}
    for k, m in gauges.items():                # +1 where the face leads out of the basin
        M = torch.as_tensor(np.asarray(m, bool), device=dev).float()
        gsx[k] = M[:, :-1] - M[:, 1:]
        gsy[k] = M[:-1, :] - M[1:, :]
    cell_a = dx * dy
    dbl = torch.float64
    b_rain = torch.zeros((), device=dev, dtype=dbl)
    b_runoff = torch.zeros((), device=dev, dtype=dbl)
    b_edge = torch.zeros((), device=dev, dtype=dbl)
    b_sea = torch.zeros((), device=dev, dtype=dbl)
    b_drain = torch.zeros((), device=dev, dtype=dbl)
    gvol = {k: torch.zeros((), device=dev, dtype=dbl) for k in gauges}
    rec_t, rec_q = [], {k: [] for k in gauges}
    vol_last, t_last_rec = {k: 0.0 for k in gauges}, 0.0
    snaps, t_snaps = [], []
    t, steps, next_rec, next_snap = 0.0, 0, 0.0, 0.0
    i_rain, R = -1, torch.zeros((ny, nx), **f32)
    wall = time.time()
    c = p.snap_coarsen

    def gauge_q():
        return {k: (gsx[k] * QX).sum() * dy + (gsy[k] * QY).sum() * dx for k in gauges}

    while t < t_end_s - 1e-6:
        i = rain.index(t)
        if i != i_rain:
            f = rain.field(i)
            R = torch.zeros((ny, nx), **f32) if f is None else torch.as_tensor(f, **f32)
            i_rain = i
        hm = float(H.max())
        dt = p.cfl * min(dx, dy) / math.sqrt(G * max(hm, 1e-3))
        dt = min(max(dt, p.dt_min), p.dt_max, t_end_s - t, (i + 1) * rain.step_s - t)
        # --- rain and losses
        dP = R * (dt / 3600.0)                                   # mm this step
        PC_new = PC + dP
        QC_new = scs_runoff(PC_new, CN, p.ia_ratio)
        exc = (QC_new - QC).clamp(min=0)                         # mm that runs off
        PC, QC = PC_new, QC_new
        b_rain += dP.double().sum() * cell_a / 1000
        b_runoff += exc.double().sum() * cell_a / 1000
        H = H + exc / 1000.0
        if DRAIN is not None:
            d = torch.minimum(H, DRAIN * dt)
            b_drain += d.double().sum() * cell_a
            H = H - d
        # --- fluxes (local inertial)
        ETA = Z + H
        hfx = torch.maximum(ETA[:, 1:], ETA[:, :-1]) - ZX
        hfy = torch.maximum(ETA[1:, :], ETA[:-1, :]) - ZY
        sx = (ETA[:, 1:] - ETA[:, :-1]) / dx
        sy = (ETA[1:, :] - ETA[:-1, :]) / dy
        hx = hfx.clamp(min=p.h_dry)
        hy = hfy.clamp(min=p.h_dry)
        QX = (QX - G * hx * dt * sx) / (1 + G * dt * NX2 * QX.abs() / hx ** (7 / 3))
        QY = (QY - G * hy * dt * sy) / (1 + G * dt * NY2 * QY.abs() / hy ** (7 / 3))
        QX = torch.where(hfx > p.h_dry, QX, torch.zeros_like(QX))
        QY = torch.where(hfy > p.h_dry, QY, torch.zeros_like(QY))
        lim_x = p.froude_max * hx * torch.sqrt(G * hx)
        lim_y = p.froude_max * hy * torch.sqrt(G * hy)
        QX = torch.maximum(torch.minimum(QX, lim_x), -lim_x)
        QY = torch.maximum(torch.minimum(QY, lim_y), -lim_y)
        # edge outflow (normal depth, outward only)
        ew = H[:, 0] ** (5 / 3) * s_w.sqrt() / N[:, 0]
        ee = H[:, -1] ** (5 / 3) * s_e.sqrt() / N[:, -1]
        en = H[0, :] ** (5 / 3) * s_n.sqrt() / N[0, :]
        es = H[-1, :] ** (5 / 3) * s_s.sqrt() / N[-1, :]
        # --- never send out more than a cell holds
        out = torch.zeros((ny, nx), **f32)
        out[:, :-1] += QX.clamp(min=0) * dt / dx
        out[:, 1:] += (-QX).clamp(min=0) * dt / dx
        out[:-1, :] += QY.clamp(min=0) * dt / dy
        out[1:, :] += (-QY).clamp(min=0) * dt / dy
        out[:, 0] += ew * dt / dx
        out[:, -1] += ee * dt / dx
        out[0, :] += en * dt / dy
        out[-1, :] += es * dt / dy
        scale = torch.where(out > H, H / out.clamp(min=1e-12), torch.ones_like(H))
        QX = torch.where(QX > 0, QX * scale[:, :-1], QX * scale[:, 1:])
        QY = torch.where(QY > 0, QY * scale[:-1, :], QY * scale[1:, :])
        ew, ee, en, es = ew * scale[:, 0], ee * scale[:, -1], en * scale[0, :], es * scale[-1, :]
        # --- update depths
        dH = torch.zeros((ny, nx), **f32)
        dH[:, :-1] -= QX * dt / dx
        dH[:, 1:] += QX * dt / dx
        dH[:-1, :] -= QY * dt / dy
        dH[1:, :] += QY * dt / dy
        dH[:, 0] -= ew * dt / dx
        dH[:, -1] -= ee * dt / dx
        dH[0, :] -= en * dt / dy
        dH[-1, :] -= es * dt / dy
        b_edge += ((ew.double().sum() + ee.double().sum()) * dy + (en.double().sum() + es.double().sum()) * dx) * dt
        H = (H + dH).clamp(min=0)
        b_sea += (H * SINK).double().sum() * cell_a
        H = torch.where(SINK, torch.zeros_like(H), H)
        t += dt
        steps += 1
        # --- records
        if gauges:
            gq = gauge_q()
            for k in gauges:
                gvol[k] += gq[k].double() * dt
        ux = torch.zeros((ny, nx), **f32)
        uy = torch.zeros((ny, nx), **f32)
        ux[:, :-1] += 0.5 * QX.abs()
        ux[:, 1:] += 0.5 * QX.abs()
        uy[:-1, :] += 0.5 * QY.abs()
        uy[1:, :] += 0.5 * QY.abs()
        wet = H > 0.01
        V = torch.where(wet, torch.sqrt(ux ** 2 + uy ** 2) / H.clamp(min=0.01), torch.zeros_like(H)).clamp(max=10)
        DF = torch.where(H >= 0.75, 1.0, torch.where(H >= 0.25, 0.5, 0.0))
        HR = H * (V + 0.5) + DF
        HMAX = torch.maximum(HMAX, H)
        VMAX = torch.maximum(VMAX, V)
        HRMAX = torch.maximum(HRMAX, HR)
        TWET = torch.where((H >= 0.3) & torch.isinf(TWET), torch.full_like(TWET, t), TWET)
        if t >= next_rec:
            span = max(t - t_last_rec, 1e-9)
            rec_t.append(t)
            for k in gauges:
                v = float(gvol[k])
                rec_q[k].append((v - vol_last[k]) / span)
                vol_last[k] = v
            t_last_rec = t
            next_rec += p.record_s
        if t >= next_snap:
            hs = torch.nn.functional.avg_pool2d(H[None, None, : ny // c * c, : nx // c * c], c)[0, 0]
            snaps.append(hs.to(torch.float16).cpu().numpy())
            t_snaps.append(t)
            next_snap += p.snap_s
        if progress and steps % 5000 == 0:
            print(f"  t={t / 3600:6.2f} h  dt={dt:5.2f} s  hmax={hm:5.2f} m  steps={steps}", flush=True)

    stored = float(H.double().sum() * cell_a)
    budget = {"rain_m3": float(b_rain), "runoff_m3": float(b_runoff), "soaked_m3": float(b_rain - b_runoff),
              "edge_m3": float(b_edge), "sea_m3": float(b_sea), "drain_m3": float(b_drain), "stored_m3": stored}
    budget["error_m3"] = stored0 + budget["runoff_m3"] - budget["edge_m3"] - budget["sea_m3"] - budget["drain_m3"] - stored
    budget["error_frac"] = budget["error_m3"] / max(budget["runoff_m3"], 1e-9)
    return Result(np.array(rec_t), {k: np.array(v) for k, v in rec_q.items()},
                  {k: float(v) for k, v in gvol.items()},
                  HMAX.cpu().numpy(), VMAX.cpu().numpy(), HRMAX.cpu().numpy(), TWET.cpu().numpy(),
                  np.stack(snaps) if snaps else np.zeros((0, 1, 1), np.float16), np.array(t_snaps),
                  budget, steps, time.time() - wall)
