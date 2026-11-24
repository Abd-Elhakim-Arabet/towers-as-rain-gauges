"""Serve the project website on this computer: http://localhost:8000

    python scripts/serve_site.py            # builds site/_local/ and serves it
    python scripts/serve_site.py --port 8080
    python scripts/serve_site.py --tailscale   # also open it to your own devices on Tailscale
    python scripts/serve_site.py --host 0.0.0.0   # open to the whole network (servers, Docker)
    python scripts/serve_site.py --export _site   # write a static copy for any web host (no live watch)

Pages:
  /               start page with links to everything below
  /explainer.pdf  the ground-up explainer
  /replay/        Phase 4 storm replay (simulated Algerian network)
  /storm3d/       3D storm over the towers: satellite imagery, terrain, 3D buildings (needs internet)
  /flood/         Phase 5 flood risk: water depth, danger, basin flow and alarms (needs internet for imagery)
  /dashboard/     Phase 6 control room: towers, rain, flood and alarm on one 3D scene (needs internet for imagery)
  /watch/         Phase 7 live watch: pick any city, watch its rain and flood model now (needs internet)

The live watch needs the model code (torch, the trained model in runs/). Without it, or with --static, only the
pages above are served.

Listens on 127.0.0.1 only, so other machines on the lab network cannot open it. --tailscale (or --host <ip>) adds the
Tailscale address: then every device on the same tailnet can open it, the lab network still cannot.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
OUT = SITE / "_local"

NAV = """<nav style="position:sticky;top:0;z-index:50;display:flex;gap:1.2rem;flex-wrap:wrap;align-items:center;
padding:.55rem max(16px,env(safe-area-inset-left,0px));background:#102029;color:#e8eef1;
font:500 14px/1.4 'IBM Plex Sans',system-ui,sans-serif">
<strong style="letter-spacing:.04em">TOWERS AS RAIN GAUGES</strong>
<a href="../" style="color:#cde2fb">Start</a>
<a href="../dashboard/" style="color:#cde2fb">Control room (simulation)</a>
<a href="../watch/" style="color:#cde2fb">Live watch</a>
<a href="../explainer.pdf" style="color:#cde2fb">Explainer (PDF)</a>
</nav>"""

REPO = "https://github.com/Abd-Elhakim-Arabet/towers-as-rain-gauges"
START = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Towers as Rain Gauges</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{ --bg:#fbfcfd; --ink:#0b1220; --ink2:#4a5568; --muted:#8a96a8; --line:#e3e7ed; --accent:#1d5cff; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0e1116; --ink:#eef1f5; --ink2:#a7b0bd; --muted:#6f7987; --line:#232831; --accent:#7ea2ff; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.6 "IBM Plex Sans",system-ui,sans-serif; }}
main {{ max-width:760px; margin:0 auto; padding:64px 16px 72px; }}
h1 {{ font-size:32px; font-weight:600; letter-spacing:-.01em; margin:0 0 12px; }}
p.lead {{ color:var(--ink2); margin:0 0 40px; font-size:17px; }}
h2 {{ font-size:13px; font-weight:500; color:var(--muted); margin:0 0 4px; }}
ul.pages {{ list-style:none; margin:0 0 40px; padding:0; border-top:1px solid var(--line); }}
ul.pages li {{ border-bottom:1px solid var(--line); }}
ul.pages a {{ display:grid; grid-template-columns:180px 1fr; gap:4px 24px; padding:14px 0; color:inherit; text-decoration:none; }}
ul.pages a:hover b {{ color:var(--accent); }}
ul.pages b {{ font-weight:500; }}
ul.pages span {{ color:var(--ink2); }}
@media (max-width:560px) {{ ul.pages a {{ grid-template-columns:1fr; }} }}
p.real {{ color:var(--ink2); margin:0 0 40px; }}
footer {{ font-size:14px; color:var(--muted); }}
a {{ color:var(--accent); }}
</style></head><body><main>
<h1>Towers as Rain Gauges</h1>
<p class="lead">Flash-flood early warning from the microwave links that already connect mobile-phone towers.
Rain weakens the signal between two towers; AI turns that into a live rain map, and a flood model shows which streets
will flood and raises the alarm.</p>
<h2>Pages</h2>
<ul class="pages">
<li><a href="dashboard/"><b>Control room</b><span>Algiers 2001 and Ghardaïa 2008 replayed: towers, rain, 3D flood water and the alarm, with a guided tour.</span></a></li>
<li><a href="watch/"><b>Live watch</b><span>Any city, now: rain, flood risk and alert level. Needs the Python server (see the setup guide).</span></a></li>
<li><a href="replay/"><b>Storm replay</b><span>The simulated Algerian link network during the two storms.</span></a></li>
<li><a href="flood/"><b>Flood risk</b><span>Water depth, danger and alarms from our flood model.</span></a></li>
<li><a href="storm3d/"><b>3D storm</b><span>The storm over the towers, with terrain and buildings.</span></a></li>
<li><a href="explainer.pdf"><b>Explainer (PDF)</b><span>The whole project in plain language, with every result.</span></a></li>
</ul>
<h2>Real and simulated</h2>
<p class="real">Accuracy was measured on real link data from Sweden, Italy and the Netherlands.
The Algerian cities are a simulation: real tower positions and real satellite rain drive virtual links.
They show how the system would work there; they do not prove accuracy there.</p>
<footer>Code, data and setup guide: <a href="{REPO}">{REPO.removeprefix("https://")}</a></footer>
</main></body></html>
"""


def wrap(fragment: str) -> str:
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">"
            "<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>"
            f"</head><body>{NAV}\n{fragment}\n</body></html>\n")


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "replay").mkdir(parents=True)
    (OUT / "storm3d").mkdir(parents=True)
    (OUT / "index.html").write_text(START)
    replay = SITE / "phase4" / "index.html"
    if replay.exists():
        (OUT / "replay" / "index.html").write_text(wrap(replay.read_text()))
    s3d = SITE / "storm3d"
    if (s3d / "index.html").exists():          # full-screen app, its own complete page
        shutil.copy(s3d / "index.html", OUT / "storm3d" / "index.html")
        for g in s3d.glob("*.geojson"):
            shutil.copy(g, OUT / "storm3d" / g.name)
    flood = SITE / "flood" / "index.html"
    if flood.exists():                         # Phase 5: full-screen app, its own complete page
        (OUT / "flood").mkdir(parents=True)
        shutil.copy(flood, OUT / "flood" / "index.html")
    dash = SITE / "dashboard"
    if (dash / "index.html").exists():         # Phase 6: full-screen app; its data folder is linked, not copied
        (OUT / "dashboard").mkdir(parents=True)
        page = (dash / "index.html").read_text()
        cache = dash / "data" / "cities.json"
        if (dash / "template.html").exists() and cache.exists():   # template edits apply without a re-export
            data = json.dumps(json.loads(cache.read_text()), separators=(",", ":"))
            page = (dash / "template.html").read_text().replace("/*DATA*/null", data)
        (OUT / "dashboard" / "index.html").write_text(page)
        if (dash / "data").exists():
            os.symlink(dash / "data", OUT / "dashboard" / "data")
        if (dash / "logos").exists():          # small operator badges drawn on the towers
            shutil.copytree(dash / "logos", OUT / "dashboard" / "logos")
    watch = SITE / "watch" / "index.html"
    if watch.exists():                         # Phase 7: the live watch page (talks to /api/)
        (OUT / "watch").mkdir(parents=True)
        shutil.copy(watch, OUT / "watch" / "index.html")
    pdf = ROOT / "docs" / "explainer" / "explainer.pdf"
    if pdf.exists():
        shutil.copy(pdf, OUT / "explainer.pdf")


def serve_live(hosts: list[str], port: int) -> bool:
    """The pages plus the live watch API and its background watcher. False if the live parts cannot be loaded."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import uvicorn
        from watch.api import make_app
    except Exception as e:                     # no fastapi/uvicorn or no model code: fall back to the plain pages
        print(f"live watch not available ({type(e).__name__}: {e}); serving the pages only", flush=True)
        return False
    public = next((h for h in hosts if h != "127.0.0.1"), "localhost")
    app = make_app(OUT, f"http://{public}:{port}")
    servers = [uvicorn.Server(uvicorn.Config(app, host=h, port=port, log_level="warning",
                                             lifespan="on" if i == 0 else "off")) for i, h in enumerate(hosts)]
    for srv in servers[1:]:
        threading.Thread(target=srv.run, daemon=True).start()
    for h in hosts:
        print(f"serving on http://{'localhost' if h == '127.0.0.1' else h}:{port}  (live watch: /watch/)", flush=True)
    print("(Ctrl+C to stop)", flush=True)
    servers[0].run()
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", action="append", default=[], help="extra address to listen on (repeatable)")
    ap.add_argument("--tailscale", action="store_true", help="also listen on this machine's Tailscale IPv4")
    ap.add_argument("--static", action="store_true", help="pages only, without the live watch")
    ap.add_argument("--export", metavar="DIR", help="write a static copy of the pages to DIR and stop")
    args = ap.parse_args()
    if args.export:
        build()
        dest = Path(args.export).resolve()
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(OUT, dest, symlinks=False)      # follows the dashboard data link, so the copy stands alone
        (dest / ".nojekyll").write_text("")
        print(f"static copy written to {dest}")
        return
    hosts = ["0.0.0.0"] if "0.0.0.0" in args.host else ["127.0.0.1"] + args.host
    if args.tailscale:
        hosts.append(subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True,
                                    check=True).stdout.split()[0])
    build()
    hosts = list(dict.fromkeys(hosts))
    if not args.static and serve_live(hosts, args.port):
        return
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
    servers = [http.server.ThreadingHTTPServer((h, args.port), handler) for h in dict.fromkeys(hosts)]
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    for h in dict.fromkeys(hosts):
        print(f"serving on http://{'localhost' if h == '127.0.0.1' else h}:{args.port}", flush=True)
    print("(Ctrl+C to stop)", flush=True)
    try:
        servers[0].serve_forever()
    finally:
        for srv in servers:
            srv.server_close()


if __name__ == "__main__":
    main()
