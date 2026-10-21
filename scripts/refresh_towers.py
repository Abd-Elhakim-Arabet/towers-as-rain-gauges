"""Fetch more OpenCelliD tiles and, if any came in, rebuild everything that shows towers.

    python scripts/refresh_towers.py [--commit]

1. resume the Algiers tile sweep (scripts/fetch_opencellid_area.py; stops at the daily quota)
2. if the number of tiles grew, or earlier tiles were never built in:
   - rebuild the tower sites            (scripts/build_algeria_towers.py)
   - rerun the Algiers simulation        (scripts/phase4_simulate.py algiers: new links, signals, maps)
   - redo the PDF numbers and figures    (export_explainer_phase4.py, then pdflatex)
   - redo the replay and the 3D storm    (export_phase4_explorer.py, export_storm3d.py)
   - refresh the local site in place     (scripts/serve_site.py build(); a running server picks it up)
   - with --commit: commit and push
Prints one summary line at the end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
TILES = ROOT / "data" / "raw" / "opencellid" / "api" / "algiers"
TOTAL = 198


def n_tiles() -> int:
    return len(list(TILES.glob("tile_*.json")))


def run(*cmd, cwd=ROOT, timeout=3600):
    print("$", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(list(map(str, cmd)), cwd=cwd, timeout=timeout, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"failed: {' '.join(map(str, cmd))}")
    return r.stdout


def main(commit: bool):
    before = n_tiles()
    out = subprocess.run([PY, "scripts/fetch_opencellid_area.py", "algiers"], cwd=ROOT, timeout=900,
                         capture_output=True, text=True).stdout
    after = n_tiles()
    api = "working, daily quota used" if "Daily limit" in out else ("working" if "tile" in out or after > before else "no answer")
    # also rebuild when earlier fetches were never built in (tower file older than the newest tile)
    sites = ROOT / "data" / "processed" / "algeria_sites.csv"
    newest = max((t.stat().st_mtime for t in TILES.glob("tile_*.json")), default=0)
    stale = not sites.exists() or sites.stat().st_mtime < newest
    if after == before and not stale:
        print(f"OpenCelliD API {api}; Algiers tiles {after}/{TOTAL}; no new tiles, nothing rebuilt")
        return
    run(PY, "scripts/build_algeria_towers.py")
    run(PY, "scripts/phase4_simulate.py", "algiers")
    run(PY, "scripts/export_explainer_phase4.py")
    for _ in range(2):
        run("pdflatex", "-interaction=nonstopmode", "explainer.tex", cwd=ROOT / "docs" / "explainer")
    run(PY, "scripts/export_phase4_explorer.py")
    run(PY, "scripts/export_storm3d.py")
    sys.path.insert(0, str(ROOT / "scripts"))
    import serve_site
    serve_site.build()                      # the running local server serves the refreshed files
    msg = f"OpenCelliD API {api}; Algiers tiles {before} -> {after}/{TOTAL}; towers, simulation, PDF, replay and 3D storm rebuilt"
    if commit:
        run("git", "add", "-A")
        st = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
        if st.returncode != 0:
            run("git", "commit", "-q", "-m",
                f"Algiers towers: {after}/{TOTAL} OpenCelliD tiles; rebuild simulation, PDF, replay, 3D storm\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
            run("git", "push", "-q")
            msg += "; committed and pushed"
    print(msg)


if __name__ == "__main__":
    main("--commit" in sys.argv[1:])
