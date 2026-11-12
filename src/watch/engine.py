"""One watch: every few minutes, read the links, map the rain, move the flood model on, decide the alert level.

    signals (simulated or a real feed)
      -> AI reads rain on every link                     (the Phase 2 model, looking only at the past)
      -> rain map on a ~500 m grid                       (Phase 3; weather-service rain where no link is near)
      -> flood model advanced by the same few minutes    (Phase 5 solver, carrying its water on)
      -> a 2-hour look-ahead on the expected rain        (the rain now for 30 min, then the weather-service forecast)
      -> alert level, history, event log, email

The engine keeps the last 4 days of signal per link, because the model compares the signal with its recent dry level.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from floodrisk import hazard, solver
from targ import features as feat
from targ import maps
from targ import train_torch as T
from targ.models.tcn_torch import TCN

from . import alerts
from .sources import fine_grid

ROOT = Path(__file__).resolve().parents[2]
AI_CONFIG = "mix21rel+mfb+mask+gt+west_mse"          # the model kept at the end of Phase 2
CORRECTION = ROOT / "results" / "phase7_heavy_rain_correction.json"
STEP_MIN = 5
BUF_MIN = 4 * 1440
AHEAD_S = 7200
NOWCAST_S = 1800            # the look-ahead keeps the rain the links see now for this long
DRY_RESET_H = 12            # after this many dry hours the soil counts as dry again
THRESH_DEFAULT = {1: 40.0, 3: 60.0}    # mm; used only until the city's own thresholds are computed
DRAIN_BUILT_MMH = 10.0      # storm drains: water leaves built-up land at this rate (times the built share of the cell)
DRAIN_OTHER_MMH = 2.0       # slow seepage elsewhere. Without these, puddles would never go and the alarm never clear.


class Reader:
    """The AI that reads rain from link signals (5-model ensemble), loaded once."""

    _one = None

    @classmethod
    def get(cls) -> "Reader":
        if cls._one is None:
            cls._one = cls()
        return cls._one

    def __init__(self):
        kw = json.loads((ROOT / "results" / "phase2_openmrg.json").read_text())["variants"]["physics"]["model_kw"]
        res = json.loads((ROOT / "results" / f"phase2_mixed_{AI_CONFIG}.json").read_text())
        self.thr = res["ensemble"]["threshold"]
        self.dev = T.device()
        self.models = []
        for seed in range(5):
            m = TCN(5, 3, **kw).to(self.dev)
            m.load_state_dict(torch.load(ROOT / "runs" / "phase2" / "mixed" / f"{AI_CONFIG}_seed{seed}.pt", map_location=self.dev))
            self.models.append(m.eval())
        self.corr = json.loads(CORRECTION.read_text()) if CORRECTION.exists() else None

    def rates(self, tl: np.ndarray, length_km, freq_ghz, pol, last_n: int) -> np.ndarray:
        """Rain (mm/h) on every path for the last `last_n` minutes of the buffer `tl` (n, T). NaN = no reading."""
        pol = np.where(np.asarray(pol).astype(str) == "H", "horizontal", "vertical")
        dyn = feat.dynamic_features(tl, length_km, freq_ghz, pol)
        stat, phys = feat.static_features(length_km, freq_ghz, pol), feat.physics_params(length_km, freq_ghz, pol)
        n = tl.shape[1]
        P, R = zip(*(T.predict(m, dyn, stat, phys, t_range=(n - last_n, n), link_batch=256) for m in self.models))
        R = np.where(np.mean(P, 0) > self.thr, np.mean(R, 0), 0.0)
        if self.corr:                                   # heavy rain is read too low: correction fitted on training data
            R = np.interp(R, self.corr["read_mmh"], self.corr["true_mmh"])
        return np.where(np.isfinite(tl[:, n - last_n:]), R, np.nan).astype(np.float32)


class Watch:
    def __init__(self, area, S: dict, source, out_dir: Path, drill: bool = False, thresholds: dict | None = None,
                 notify: bool = True, url: str = ""):
        self.area, self.S, self.source, self.out, self.drill = area, S, source, Path(out_dir), drill
        self.thresholds = thresholds or dict(THRESH_DEFAULT)
        self.thresholds_own = thresholds is not None
        self.notify, self.url = notify, url
        self.out.mkdir(parents=True, exist_ok=True)
        g, sl = S["g"], source.sub_links
        self.flat, self.flon = fine_grid(area.box)
        _, mlat, mlon = maps.merge_directions(np.zeros((len(sl), 1)), sl.site_0_lat.values, sl.site_0_lon.values,
                                              sl.site_1_lat.values, sl.site_1_lon.values)
        la, lo = np.meshgrid(self.flat, self.flon, indexing="ij")
        ref = (area.lat, area.lon)
        self.W = maps.idw_weights(np.column_stack(maps.to_km(mlat, mlon, *ref)),
                                  np.column_stack(maps.to_km(la.ravel(), lo.ravel(), *ref)))
        self.mid = np.column_stack([mlon, mlat])
        self.ir = np.clip(np.round((g.lat - self.flat[0]) / (self.flat[1] - self.flat[0])).astype(int), 0, self.flat.size - 1)
        self.ic = np.clip(np.round((g.lon - self.flon[0]) / (self.flon[1] - self.flon[0])).astype(int), 0, self.flon.size - 1)
        self.town_ha = np.where(S["town"], S["built_ha"], 0.0)
        self.town_w = self.town_ha / max(self.town_ha.sum(), 1e-9)
        self.tl = None
        self.t_min = None                   # last minute (UTC, since 1970) the watch has processed
        self.flood = None
        self.level, self.calm, self.peak = "normal", 0, "normal"
        self.rain_log = deque(maxlen=24 * 60 // STEP_MIN)     # (minute, town-mean mm in the step)
        self.history = deque(maxlen=7 * 24 * 60 // STEP_MIN)
        self.params = solver.Params(snap_s=1e12, record_s=1e12)
        self.drain = (DRAIN_OTHER_MMH + (DRAIN_BUILT_MMH - DRAIN_OTHER_MMH) * g.built).astype(np.float32)
        self.load()

    # ------------------------------------------------------------------ one step
    def step(self, now_min: int, coarse) -> dict:
        now_min = int(now_min) // STEP_MIN * STEP_MIN
        t0 = time.time()
        if self.t_min is not None and now_min <= self.t_min:
            return self.status
        sl = self.source.sub_links
        if self.tl is None or now_min - self.t_min > BUF_MIN // 2:       # first run, or back after a long stop
            self.tl = self.source.loss(np.arange(now_min - BUF_MIN + 1, now_min + 1), coarse)
            n_new = STEP_MIN
        else:
            new = self.source.loss(np.arange(self.t_min + 1, now_min + 1), coarse)
            self.tl = np.concatenate([self.tl[:, new.shape[1]:], new], axis=1)
            n_new = min(new.shape[1], 60)
        self.t_min = now_min
        R = Reader.get().rates(self.tl, sl.length_km.values, sl.freq_ghz.values, sl.pol.values, n_new)
        seen = np.isfinite(R)
        link_now = np.where(seen.any(1), np.where(seen, R, 0).sum(1) / np.maximum(seen.sum(1), 1), np.nan)
        Rp, _, _ = maps.merge_directions(link_now[:, None], sl.site_0_lat.values, sl.site_0_lon.values,
                                         sl.site_1_lat.values, sl.site_1_lon.values)
        shape = (self.flat.size, self.flon.size)
        m = maps.apply_idw(self.W, Rp)[:, 0].reshape(shape)
        filler = coarse.at(np.array([now_min * 60.0]), self.flat, self.flon)[0]
        rain_fine = np.where(np.isfinite(m), m, filler).astype(np.float32)        # weather-service rain where no link is near
        rain = rain_fine[self.ir][:, self.ic]                                     # on the flood grid, north -> south
        S, g = self.S, self.S["g"]
        town_mm = float((rain * self.town_w).sum()) * n_new / 60.0
        self.rain_log.append((now_min, town_mm))
        if self.flood is not None and sum(v for t, v in self.rain_log if t > now_min - DRY_RESET_H * 60) < 0.5 \
                and len(self.rain_log) >= DRY_RESET_H * 60 // STEP_MIN:
            self.flood["pc"][:] = 0
            self.flood["qc"][:] = 0
        res = solver.simulate(S["z"], S["n"], S["cn"], g.dx, g.dy, solver.HourlyRain(rain[None], step_s=n_new * 60.0),
                              n_new * 60.0, sink=S["sea"], params=self.params, state=self.flood, drain_mmh=self.drain)
        self.flood = res.state
        h = self.flood["h"]
        ha_now = float(self.town_ha[h >= hazard.WET_M].sum())
        # look ahead: the rain the links see now for 30 minutes, then the weather-service forecast
        ha_ahead, raining = ha_now, town_mm > 0.02 or bool(rain.max() > 0.5)
        if raining or h.max() > 0.05 or coarse.wet(now_min * 60.0, now_min * 60.0 + AHEAD_S):
            tt = now_min * 60.0 + 900.0 * np.arange(1, AHEAD_S // 900 + 1)
            ahead = coarse.at(tt, g.lat, g.lon)
            ahead[: NOWCAST_S // 900] = np.maximum(ahead[: NOWCAST_S // 900], rain[None])
            ra = solver.simulate(S["z"], S["n"], S["cn"], g.dx, g.dy, solver.HourlyRain(ahead, step_s=900.0), float(AHEAD_S),
                                 sink=S["sea"], params=self.params, drain_mmh=self.drain,
                                 state={k: v.copy() for k, v in self.flood.items()})
            ha_ahead = max(ha_now, float(self.town_ha[ra.hmax >= hazard.WET_M].sum()))
        sums = {T_: sum(v for t, v in self.rain_log if t > now_min - T_ * 60) for T_ in self.thresholds}
        ratio = max((sums[T_] / thr for T_, thr in self.thresholds.items() if thr), default=0.0)
        reading = alerts.level(ha_now, ha_ahead, S["alarm_ha"], ratio)
        old = self.level
        self.level, self.calm = alerts.settle(self.level, reading, self.calm)
        wet_links = int(np.nansum(Rp[:, 0] > 0.1))
        ok_links = int(np.isfinite(Rp[:, 0]).sum())
        rec = {"t": now_min * 60, "level": self.level, "rain_mmh": round(float((rain * self.town_w).sum()), 2),
               "rain_max_mmh": round(float(rain.max()), 1), "ha_now": round(ha_now, 2), "ha_ahead": round(ha_ahead, 2),
               "ratio": round(float(ratio), 3), "links_wet": wet_links}
        self.history.append(rec)
        self.status = {**rec, "why": self._why(reading, ha_now, ha_ahead, ratio), "alarm_ha": round(S["alarm_ha"], 1),
                       "links_n": int(Rp.shape[0]), "links_ok": ok_links, "rain_3h_mm": round(sums.get(3, 0.0), 1),
                       "thresholds_mm": self.thresholds, "thresholds_own": self.thresholds_own, "drill": self.drill,
                       "source": self.source.kind, "feed_problem": getattr(self.source, "last_problem", ""),
                       "step_s": round(time.time() - t0, 1), "updated": int(time.time())}
        if self.level != old:
            self._event(old, self.level)
        self._write(rain_fine, h, Rp[:, 0])
        return self.status

    status: dict = {}

    def _why(self, reading, ha_now, ha_ahead, ratio) -> str:
        a = self.S["alarm_ha"]
        if self.level == "flooding":
            return f"The model shows {ha_now:.0f} ha of built-up land under 30 cm of water (alarm at {a:.0f} ha)"
        if self.level == "warning":
            return f"The model expects {ha_ahead:.0f} ha of built-up land under water within 2 hours (alarm at {a:.0f} ha)"
        if self.level == "watch" and ratio >= 0.5:
            return f"Rain so far is at {100 * ratio:.0f}% of the rain that would flood the town"
        if self.level == "watch":
            return f"Water is standing on {max(ha_now, ha_ahead):.0f} ha of built-up land (alarm at {a:.0f} ha)"
        return "Raining, below the flash-flood threshold" if ratio > 0.02 else "No heavy rain over the town"

    # ------------------------------------------------------------------ events and email
    def _event(self, old: str, new: str) -> None:
        up = alerts.ORDER.index(new) > alerts.ORDER.index(old)
        if up:
            self.peak = max(self.peak, new, key=alerts.ORDER.index)
        mail = ""
        if self.notify and ((up and new in ("warning", "flooding")) or (new == "normal" and self.peak in ("warning", "flooding"))):
            facts = {**self.status, "time": time.strftime("%Y-%m-%d %H:%M", time.gmtime(self.t_min * 60)), "url": self.url}
            mail = alerts.send(alerts.compose(self.area.name, new, old, facts, self.drill, self.source.kind == "simulation"))
        if new == "normal":
            self.peak = "normal"
        ev = {"t": self.t_min * 60, "from": old, "to": new, "why": self.status["why"], "email": mail}
        with open(self.out / "events.jsonl", "a") as f:
            f.write(json.dumps(ev) + "\n")

    # ------------------------------------------------------------------ files the page reads
    def _write(self, rain_fine, h, link_mmh) -> None:
        from . import store
        store.write_json(self.out / "status.json", self.status)
        store.write_json(self.out / "history.json", list(self.history))
        store.write_json(self.out / "links.json", [None if not np.isfinite(v) else round(float(v), 1) for v in link_mmh])
        store.rain_png(self.out / "rain.png", rain_fine[::-1])
        store.depth_png(self.out / "depth.png", h, self.S["sea"])
        if not self.drill:
            self.save()

    def save(self) -> None:
        tmp = self.out / "state.tmp.npz"
        np.savez(tmp, tl=self.tl, t_min=self.t_min, level=self.level, calm=self.calm, peak=self.peak,
                 rain_log=np.array(self.rain_log, float).reshape(-1, 2), w=getattr(self.source, "w", np.zeros(0)),
                 **{f"flood_{k}": v for k, v in self.flood.items()})
        tmp.replace(self.out / "state.npz")

    def load(self) -> None:
        f = self.out / "state.npz"
        if self.drill or not f.exists():
            return
        z = np.load(f)
        if z["tl"].shape[0] != len(self.source.sub_links):
            return                                       # the network changed: start again
        self.tl, self.t_min = z["tl"], int(z["t_min"])
        self.level, self.calm, self.peak = str(z["level"]), int(z["calm"]), str(z["peak"])
        self.rain_log.extend((int(a), float(b)) for a, b in z["rain_log"])
        self.flood = {k[6:]: z[k] for k in z.files if k.startswith("flood_")}
        if hasattr(self.source, "w") and z["w"].size == self.source.w.size:
            self.source.w = z["w"]
        h = self.out / "history.json"
        if h.exists():
            self.history.extend(json.loads(h.read_text()))
        s = self.out / "status.json"
        if s.exists():
            self.status = json.loads(s.read_text())
