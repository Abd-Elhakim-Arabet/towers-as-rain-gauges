"""The web API behind the Live watch page, and the static site next to it.

    GET    /api/search?q=                  city candidates (+ known live sources for each)
    GET    /api/sources                    installed adapters and the standard format
    POST   /api/check-feed                 does a feed fit the standard format? problems in plain words
    GET    /api/watches                    the watched cities
    POST   /api/watches                    start watching a city
    GET    /api/watches/{key}              setup progress, status, network
    DELETE /api/watches/{key}              stop watching
    GET    /api/watches/{key}/{file}       history.json, events.json, links.json, rain.png, depth.png  (?drill=1)
    POST   /api/watches/{key}/drill        run a made-up storm through the live chain
    POST   /api/watches/{key}/test-email   send a test email
    GET    /api/mail                       alarm email set-up: what is set (password never returned)
    POST   /api/mail                       save it (from this computer, or by anyone while nothing is set yet)
    POST   /api/mail/test                  send a test email to the saved address

There is no login: the server listens on this computer (and, if asked, the Tailscale address) only. Feed addresses
given through the page must be http(s); reading local files is only possible from an adapter file on the server.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import alerts, area, sources
from .service import Manager, search

FILES = {"history.json", "links.json", "rain.png", "depth.png", "status.json"}
FORMAT = {
    "links": [["link_id", "your name for the link"], ["lat_a, lon_a", "one end, degrees"], ["lat_b, lon_b", "the other end, degrees"],
              ["freq_ghz", "frequency in GHz"], ["pol", "polarisation: H or V"]],
    "readings": [["time", "UTC, like 2026-11-12T10:15:00Z"], ["link_id", "as in the links table"],
                 ["rsl_dbm", "received level, dBm"], ["tsl_dbm", "sent level, dBm (or give loss_db = sent minus received)"]],
    "notes": ["CSV or JSON.", "One reading per link at least every 15 minutes; every minute is best.",
              "The readings address may contain {since}: it is replaced by the UTC time we need data from."]}


class NewWatch(BaseModel):
    name: str
    lat: float
    lon: float
    half_km: float = 10.0
    mode: str = "simulation"            # simulation | feed
    adapter: str = ""
    links_url: str = ""
    readings_url: str = ""


class Mail(BaseModel):
    to: str
    user: str = ""
    password: str = ""
    host: str = ""
    port: str = "587"


class Feed(BaseModel):
    adapter: str = ""
    links_url: str = ""
    readings_url: str = ""


def _feed(f) -> dict:
    if f.adapter:
        return {"adapter": f.adapter}
    for u in (f.links_url, f.readings_url):
        if not re.match(r"^https?://", u):
            raise HTTPException(400, "Give two http(s) addresses (links and readings), or choose an installed adapter.")
    return {"links_url": f.links_url, "readings_url": f.readings_url}


def make_app(static_dir: Path | None = None, public_url: str = "") -> FastAPI:
    mgr = Manager(public_url)

    @asynccontextmanager
    async def lifespan(_app):
        mgr.start()
        yield
        mgr.stop()

    app = FastAPI(title="Towers as Rain Gauges: live watch", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.manager = mgr

    @app.get("/api/search")
    def api_search(q: str):
        if not q.strip():
            return []
        try:
            return search(q.strip())
        except Exception as e:
            raise HTTPException(502, f"The place search is not answering ({type(e).__name__}). Try again in a moment.")

    @app.get("/api/sources")
    def api_sources():
        return {"adapters": sources.installed_adapters(), "format": FORMAT, "mail": alerts.mail_ready()}

    @app.post("/api/check-feed")
    def api_check(f: Feed):
        try:
            cfg = _feed(f)
            ad = sources.load_adapter(cfg["adapter"]) if "adapter" in cfg else sources.TableAdapter(cfg["links_url"], cfg["readings_url"])
            links = ad.links()
            problems = sources.check_links(links)
            n_read = 0
            if not problems:
                import pandas as pd
                r = ad.fetch(pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=6))
                n_read = len(r)
                problems = sources.check_readings(r, links) if n_read else ["readings: nothing came back for the last 6 hours"]
            return {"ok": not problems, "problems": problems, "links": len(links), "readings": n_read}
        except HTTPException:
            raise
        except Exception as e:
            return {"ok": False, "problems": [f"could not read the feed: {type(e).__name__}: {e}"], "links": 0, "readings": 0}

    @app.get("/api/watches")
    def api_list():
        return mgr.summary()

    @app.post("/api/watches")
    def api_create(w: NewWatch):
        if w.mode not in ("simulation", "feed"):
            raise HTTPException(400, "mode must be simulation or feed")
        if not (-85 <= w.lat <= 85 and -180 <= w.lon <= 180):
            raise HTTPException(400, "That place is outside the map.")
        feed = _feed(w) if w.mode == "feed" else None
        return {"key": mgr.create(w.name[:120], w.lat, w.lon, w.half_km, w.mode, feed)}

    def _known(key: str) -> None:
        if not re.fullmatch(r"[a-z0-9_-]+", key) or not (area.LIVE / key / "watch.json").exists():
            raise HTTPException(404, "No such watch.")

    @app.get("/api/watches/{key}")
    def api_detail(key: str):
        _known(key)
        return mgr.detail(key)

    @app.delete("/api/watches/{key}")
    def api_delete(key: str):
        _known(key)
        mgr.remove(key)
        return {"ok": True}

    @app.get("/api/watches/{key}/events.json")
    def api_events(key: str, drill: int = 0):
        _known(key)
        f = area.LIVE / key / ("drill" if drill else "live") / "events.jsonl"
        return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()][-50:] if f.exists() else []

    @app.get("/api/watches/{key}/{name}")
    def api_file(key: str, name: str, drill: int = 0):
        _known(key)
        f = area.LIVE / key / ("drill" if drill else "live") / name
        if name not in FILES or not f.exists():
            raise HTTPException(404, "Not ready yet.")
        return FileResponse(f, headers={"Cache-Control": "no-store"})

    @app.post("/api/watches/{key}/drill")
    def api_drill(key: str):
        _known(key)
        if key not in mgr.watches:
            raise HTTPException(409, "The city is still being set up.")
        mgr.drill(key)
        return {"ok": True}

    @app.post("/api/watches/{key}/test-email")
    def api_test_email(key: str):
        _known(key)
        w = mgr.watches.get(key)
        if w is None:
            raise HTTPException(409, "The city is still being set up.")
        msg = alerts.compose(w.area.name, "warning", "normal", {**w.status, "time": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
                                                                 "url": w.url}, drill=True, simulated=w.source.kind == "simulation")
        msg.replace_header("Subject", f"[TEST] Flood alarm email for {w.area.name}")
        return JSONResponse({"result": alerts.send(msg)})

    def _may_edit(req: Request) -> bool:
        # The page has no login. Changing where alarms go is allowed from the computer running the server, or by
        # anyone while no address is set at all (first start, e.g. in Docker); after that only from this computer.
        # ALARM_EDIT=any lifts the rule (set in the Docker image, where every browser looks like another machine).
        host = req.client.host if req.client else ""
        return (host in ("127.0.0.1", "::1", "localhost") or not alerts.env("ALERT_TO")
                or os.environ.get("ALARM_EDIT") == "any")

    @app.get("/api/mail")
    def api_mail(req: Request):
        return {**alerts.mail_ready(), "can_edit": _may_edit(req)}

    @app.post("/api/mail")
    def api_mail_save(m: Mail, req: Request):
        if not _may_edit(req):
            raise HTTPException(403, "The alarm email can only be changed on the computer that runs the server.")
        problems = alerts.save_mail(m.to, m.user, m.password, m.host, m.port)
        if problems:
            raise HTTPException(422, " ".join(problems))
        return {**alerts.mail_ready(), "can_edit": True}

    last_test = [0.0]

    @app.post("/api/mail/test")
    def api_mail_test():
        if time.time() - last_test[0] < 20:
            raise HTTPException(429, "A test email was just sent. Wait a few seconds.")
        if not alerts.env("ALERT_TO"):
            raise HTTPException(409, "Save an address first.")
        last_test[0] = time.time()
        return {"result": alerts.send(alerts.test_message())}

    if static_dir is not None:
        app.mount("/", StaticFiles(directory=static_dir, html=True, follow_symlink=True), name="site")
    return app
