"""Serve the project website on this computer: http://localhost:8000

    python scripts/serve_site.py            # builds site/_local/ and serves it
    python scripts/serve_site.py --port 8080

Pages:
  /               opens the explainer PDF (the project write-up)
  /explainer.pdf  the ground-up explainer
  /algeria/       Algeria tower map
  /replay/        Phase 4 storm replay (simulated Algerian network)
  /storm3d/       3D storm over the towers: satellite imagery, terrain, 3D buildings (needs internet)

Listens on 127.0.0.1 only, so other machines on the lab network cannot open it.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
OUT = SITE / "_local"

NAV = """<nav style="position:sticky;top:0;z-index:50;display:flex;gap:1.2rem;flex-wrap:wrap;align-items:center;
padding:.55rem max(16px,env(safe-area-inset-left,0px));background:#102029;color:#e8eef1;
font:500 14px/1.4 'IBM Plex Sans',system-ui,sans-serif">
<strong style="letter-spacing:.04em">TOWERS AS RAIN GAUGES</strong>
<a href="/explainer.pdf" style="color:#cde2fb">Explainer (PDF)</a>
<a href="/algeria/" style="color:#cde2fb">Algeria towers</a>
<a href="/replay/" style="color:#cde2fb">Storm replay (simulation)</a>
<a href="/storm3d/" style="color:#cde2fb">3D storm</a>
</nav>"""


def wrap(fragment: str) -> str:
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">"
            "<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>"
            f"</head><body>{NAV}\n{fragment}\n</body></html>\n")


def build() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_site.py")], check=True)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "algeria").mkdir(parents=True)
    (OUT / "replay").mkdir(parents=True)
    (OUT / "storm3d").mkdir(parents=True)
    (OUT / "index.html").write_text('<!doctype html><meta charset="utf-8">'
                                    '<meta http-equiv="refresh" content="0; url=/explainer.pdf">'
                                    '<a href="/explainer.pdf">Explainer (PDF)</a>\n')
    (OUT / "algeria" / "index.html").write_text(wrap((SITE / "algeria" / "index.html").read_text()))
    replay = SITE / "phase4" / "index.html"
    if replay.exists():
        (OUT / "replay" / "index.html").write_text(wrap(replay.read_text()))
    s3d = SITE / "storm3d"
    if (s3d / "index.html").exists():          # full-screen app, its own complete page
        shutil.copy(s3d / "index.html", OUT / "storm3d" / "index.html")
        for g in s3d.glob("*.geojson"):
            shutil.copy(g, OUT / "storm3d" / g.name)
    pdf = ROOT / "docs" / "explainer" / "explainer.pdf"
    if pdf.exists():
        shutil.copy(pdf, OUT / "explainer.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    build()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as srv:
        print(f"serving on http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
