"""The background watcher: sets cities up, steps every watch every few minutes, runs drills.

One Manager lives in the web server. Heavy work (setup, model steps) is done one at a time behind a lock, so a second
city or a drill never fights the first for the graphics card. Watches are listed in data/live/<city>/watch.json and
picked up again when the server restarts.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

from . import alerts, area, geocode, sources, weather
from .engine import DRAIN_BUILT_MMH, DRAIN_OTHER_MMH, STEP_MIN

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "live_sources.yaml"
WEATHER_EVERY_S = 900          # the weather service has new data every 15 minutes at best
SETUP_STEPS = ("Terrain, land cover and soil", "Tower sites", "Links between towers", "Flash-flood rain thresholds",
               "Signal history (last 4 days)", "First rain map and flood step")


class Manager:
    def __init__(self, public_url: str = ""):
        self.public_url = public_url.rstrip("/")
        self.lock = threading.Lock()           # one heavy computation at a time
        self.watches: dict[str, object] = {}
        self.cfg: dict[str, dict] = {}
        self.net: dict[str, dict] = {}
        self.jobs: dict[str, dict] = {}        # setup progress per city
        self.drills: dict[str, dict] = {}
        self.coarse: dict[str, tuple[float, object]] = {}
        self._stop = threading.Event()

    # ------------------------------------------------------------------ life cycle
    def start(self) -> None:
        for f in sorted(area.LIVE.glob("*/watch.json")):
            cfg = json.loads(f.read_text())
            if cfg.get("active"):
                self.jobs[cfg["key"]] = self._job()
                threading.Thread(target=self._setup, args=(cfg,), daemon=True).start()
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(20):
            now_min = int(time.time()) // 60 // STEP_MIN * STEP_MIN
            for key, w in list(self.watches.items()):
                if w.t_min is not None and now_min <= w.t_min:
                    continue
                try:
                    with self.lock:
                        w.step(now_min, self._weather(key, w.area.box))
                except Exception as e:                       # one bad step must not end the watch
                    self.cfg[key]["last_error"] = f"{type(e).__name__}: {e}"
                    traceback.print_exc()

    def _weather(self, key: str, box):
        t, c = self.coarse.get(key, (0.0, None))
        if c is None or time.time() - t > WEATHER_EVERY_S:
            try:
                c = weather.fetch(box)
                self.coarse[key] = (time.time(), c)
            except Exception:
                if c is None:
                    raise                                      # no weather at all yet: the step cannot run
        return c

    # ------------------------------------------------------------------ create / remove
    @staticmethod
    def _job() -> dict:
        return {"steps": [{"name": n, "state": "waiting", "note": ""} for n in SETUP_STEPS], "done": False, "error": ""}

    def create(self, name: str, lat: float, lon: float, half_km: float, mode: str, feed: dict | None = None) -> str:
        a = area.make(name, lat, lon, half_km)
        cfg = {"key": a.key, "name": name, "lat": lat, "lon": lon, "half_km": half_km, "mode": mode, "feed": feed or {},
               "active": True, "created": int(time.time())}
        a.save()
        (a.dir / "watch.json").write_text(json.dumps(cfg, indent=1))
        self.jobs[a.key] = self._job()
        threading.Thread(target=self._setup, args=(cfg,), daemon=True).start()
        return a.key

    def remove(self, key: str) -> None:
        self.watches.pop(key, None)
        f = area.LIVE / key / "watch.json"
        if f.exists():
            cfg = json.loads(f.read_text())
            cfg["active"] = False
            f.write_text(json.dumps(cfg, indent=1))
        self.cfg.pop(key, None)
        self.jobs.pop(key, None)

    def _source(self, cfg: dict, a, links=None):
        if cfg["mode"] == "simulation":
            return sources.SimSource(a.box, links, seed=zlib.crc32(a.key.encode()) % 2 ** 31)
        feed = cfg["feed"]
        adapter = sources.load_adapter(feed["adapter"]) if feed.get("adapter") else \
            sources.TableAdapter(feed["links_url"], feed["readings_url"])
        return sources.FeedSource(adapter)

    def _setup(self, cfg: dict) -> None:
        from targ import simnet

        from . import towers
        from .engine import Watch
        key, job = cfg["key"], self.jobs[cfg["key"]]

        def run(i, fn):
            job["steps"][i]["state"] = "running"
            out = fn()
            job["steps"][i]["state"] = "done"
            return out

        try:
            a = area.Area.load(key)
            with self.lock:
                S = run(0, lambda: area.prepare(a))
                g = S["g"]
                job["steps"][0]["note"] = f"{g.z.shape[1]} x {g.z.shape[0]} cells of {g.dx:.0f} m"
                if cfg["mode"] == "simulation":
                    sites, counts = run(1, lambda: towers.build(a, S))
                    real = sum(v for k, v in counts.items() if k != "estimated")
                    job["steps"][1]["note"] = f"{real} real, {counts.get('estimated', 0)} estimated"
                    links = run(2, lambda: simnet.build(sites))
                    src = self._source(cfg, a, links)
                else:
                    job["steps"][1].update(state="done", note="from your data")
                    src = run(2, lambda: self._source(cfg, a))
                    sites, counts = None, {"your data": len(src.sub_links)}
                job["steps"][2]["note"] = f"{len(src.links)} links"
                drain = (DRAIN_OTHER_MMH + (DRAIN_BUILT_MMH - DRAIN_OTHER_MMH) * g.built).astype(np.float32)
                thr = run(3, lambda: area.thresholds(a, S, drain))
                job["steps"][3]["note"] = ", ".join(f"{v:.0f} mm in {k} h" for k, v in thr.items() if v) or "not reached in the model"
                w = Watch(a, S, src, a.dir / "live", thresholds=thr, url=f"{self.public_url}/watch/#{key}" if self.public_url else "")
                self.net[key] = self._network(a, S, src, sites, counts)
                now_min = int(time.time()) // 60 // STEP_MIN * STEP_MIN
                job["steps"][4]["state"] = "running"
                coarse = self._weather(key, a.box)
                if w.t_min is None or now_min > w.t_min:
                    w.step(now_min, coarse)
                job["steps"][4]["state"] = job["steps"][5]["state"] = "done"
            self.cfg[key] = cfg
            self.watches[key] = w
            job["done"] = True
        except Exception as e:
            traceback.print_exc()
            job["error"] = f"{type(e).__name__}: {e}"
            for s in job["steps"]:
                if s["state"] == "running":
                    s["state"] = "failed"

    @staticmethod
    def _network(a, S, src, sites, counts) -> dict:
        g, sl = S["g"], src.sub_links
        one = sl.drop_duplicates("link_id")
        elev = lambda la, lo: [round(float(g.z[g.index(y, x)]), 1) for y, x in zip(la, lo)]      # noqa: E731
        net = {"box": list(a.box), "counts": counts,
               "links": [[round(r.site_0_lon, 5), round(r.site_0_lat, 5), e0, round(r.site_1_lon, 5), round(r.site_1_lat, 5), e1]
                         for r, e0, e1 in zip(one.itertuples(), elev(one.site_0_lat, one.site_0_lon), elev(one.site_1_lat, one.site_1_lon))],
               "sites": []}
        if sites is not None:
            net["sites"] = [[round(r.lon, 5), round(r.lat, 5), e, int(bool(r.real))]
                            for r, e in zip(sites.itertuples(), elev(sites.lat, sites.lon))]
        return net

    # ------------------------------------------------------------------ drills
    def drill(self, key: str) -> None:
        if key not in self.watches or self.drills.get(key, {}).get("running"):
            return
        self.drills[key] = {"running": True, "error": "", "started": int(time.time())}
        threading.Thread(target=self._drill, args=(key,), daemon=True).start()

    def _drill(self, key: str) -> None:
        import shutil

        from .engine import Watch
        live = self.watches[key]
        out = live.out.parent / "drill"
        try:
            shutil.rmtree(out, ignore_errors=True)
            cfg = self.cfg[key]
            src = self._source(cfg, live.area, getattr(live.source, "links", None)) if cfg["mode"] == "simulation" else None
            if src is None:                                   # a drill always uses simulated links on the watched network
                src = sources.SimSource(live.area.box, _as_links(live.source.sub_links), seed=7)
            thr3 = live.thresholds.get(3) or 60.0
            now = int(time.time()) // 60 // STEP_MIN * STEP_MIN
            storm = weather.design_storm(live.area.box, now * 60.0, total_mm=2.2 * thr3, hours=3.0)
            w = Watch(live.area, live.S, src, out, drill=True, thresholds=live.thresholds, url=live.url)
            for k in range(0, 4 * 60 + 1, 10):               # 4 hours of storm clock in 10-minute hops, as fast as it computes
                if key not in self.watches:
                    break
                with self.lock:
                    w.step(now + k, storm)
        except Exception as e:
            traceback.print_exc()
            self.drills[key]["error"] = f"{type(e).__name__}: {e}"
        self.drills[key]["running"] = False

    # ------------------------------------------------------------------ what the page asks for
    def summary(self) -> list[dict]:
        out = []
        for key in sorted(set(self.jobs) | set(self.watches)):
            w, cfg = self.watches.get(key), self.cfg.get(key) or json.loads((area.LIVE / key / "watch.json").read_text())
            out.append({"key": key, "name": cfg["name"], "mode": cfg["mode"], "lat": cfg["lat"], "lon": cfg["lon"],
                        "level": w.status.get("level") if w else None, "ready": w is not None,
                        "error": self.jobs.get(key, {}).get("error", "")})
        return out

    def detail(self, key: str) -> dict:
        w = self.watches.get(key)
        cfg = self.cfg.get(key) or json.loads((area.LIVE / key / "watch.json").read_text())
        d = {"key": key, "name": cfg["name"], "mode": cfg["mode"], "lat": cfg["lat"], "lon": cfg["lon"],
             "setup": self.jobs.get(key), "ready": w is not None, "mail": alerts.mail_ready(),
             "drill": self.drills.get(key), "last_error": cfg.get("last_error", "")}
        if w:
            d.update(status=w.status, network=self.net[key], box=list(w.area.box),
                     ground=w.S["g"].dem_source, biggest_stream_km2=round(w.S["biggest_stream_km2"]))
        return d


def _as_links(sl):
    """Sub-links of a real feed, as the links table the simulator expects (for drills on a real network)."""
    one = sl.drop_duplicates("link_id")
    return one[["link_id", "site_0_lon", "site_0_lat", "site_1_lon", "site_1_lat", "length_km", "freq_ghz", "pol"]].assign(operator="feed")


def known_sources(lat: float, lon: float) -> list[dict]:
    """Live link-data sources registered for this place (configs/live_sources.yaml). None are public today."""
    import yaml
    if not REGISTRY.exists():
        return []
    out = []
    for s in (yaml.safe_load(REGISTRY.read_text()) or {}).get("sources") or []:
        lat0, lon0, lat1, lon1 = s["box"]
        if lat0 <= lat <= lat1 and lon0 <= lon <= lon1:
            out.append({k: s[k] for k in ("name", "adapter") if k in s})
    return out


def search(q: str) -> list[dict]:
    return [dict(c, known=known_sources(c["lat"], c["lon"])) for c in geocode.search(q)]
