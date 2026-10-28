"""Serve the project website on this computer: http://localhost:8000

    python scripts/serve_site.py            # builds site/_local/ and serves it
    python scripts/serve_site.py --port 8080
    python scripts/serve_site.py --tailscale   # also open it to your own devices on Tailscale

Pages:
  /               opens the explainer PDF (the project write-up)
  /explainer.pdf  the ground-up explainer
  /replay/        Phase 4 storm replay (simulated Algerian network)
  /storm3d/       3D storm over the towers: satellite imagery, terrain, 3D buildings (needs internet)
  /flood/         Phase 5 flood risk: water depth, danger, basin flow and alarms (needs internet for imagery)
  /dashboard/     Phase 6 control room: towers, rain, flood and alarm on one 3D scene (needs internet for imagery)

Listens on 127.0.0.1 only, so other machines on the lab network cannot open it. --tailscale (or --host <ip>) adds the
Tailscale address: then every device on the same tailnet can open it, the lab network still cannot.
"""

from __future__ import annotations

import argparse
import functools
import http.server
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
<a href="/explainer.pdf" style="color:#cde2fb">Explainer (PDF)</a>
<a href="/replay/" style="color:#cde2fb">Storm replay (simulation)</a>
<a href="/storm3d/" style="color:#cde2fb">3D storm</a>
<a href="/flood/" style="color:#cde2fb">Flood risk (simulation)</a>
<a href="/dashboard/" style="color:#cde2fb">Control room (simulation)</a>
</nav>"""


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
    (OUT / "index.html").write_text('<!doctype html><meta charset="utf-8">'
                                    '<meta http-equiv="refresh" content="0; url=/explainer.pdf">'
                                    '<a href="/explainer.pdf">Explainer (PDF)</a>\n')
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
        shutil.copy(dash / "index.html", OUT / "dashboard" / "index.html")
        if (dash / "data").exists():
            os.symlink(dash / "data", OUT / "dashboard" / "data")
    pdf = ROOT / "docs" / "explainer" / "explainer.pdf"
    if pdf.exists():
        shutil.copy(pdf, OUT / "explainer.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", action="append", default=[], help="extra address to listen on (repeatable)")
    ap.add_argument("--tailscale", action="store_true", help="also listen on this machine's Tailscale IPv4")
    args = ap.parse_args()
    hosts = ["127.0.0.1"] + args.host
    if args.tailscale:
        hosts.append(subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True,
                                    check=True).stdout.split()[0])
    build()
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
