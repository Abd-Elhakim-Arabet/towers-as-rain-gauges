"""The ready-made demo data: everything the pages and the live watch need that is too big for git.

    python scripts/demo_bundle.py get            # download and unpack it into this repo (about 300 MB)
    python scripts/demo_bundle.py get --file X   # unpack a copy you already have
    python scripts/demo_bundle.py make           # (maintainers) pack it from a full working copy

What is inside (paths relative to the repo):
  runs/phase2/mixed/<model>_seed0..4.pt   the five trained AI models the live watch reads rain with
  data/interim/p4/donors_8640.npz         real dry-weather link recordings (Italy) used by the simulation
  data/processed/algeria_sites.csv        tower sites in Algeria from OpenCelliD (CC BY-SA 4.0)
  site/dashboard/index.html + data/       the control room: Algiers 2001 and Ghardaia 2008 replays
  site/flood/index.html                   the flood-risk page

Everything here can be rebuilt from the raw data with the phase scripts (see docs/SETUP.md); the bundle only saves
the hours that takes. Standard library only, so it runs before anything else is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Abd-Elhakim-Arabet/towers-as-rain-gauges"
TAG = "demo-data-v1"
NAME = "demo-bundle.tar"
URL = f"https://github.com/{REPO}/releases/download/{TAG}/{NAME}"
MODEL = "mix21rel+mfb+mask+gt+west_mse"          # same as watch.engine.AI_CONFIG
PATHS = ([f"runs/phase2/mixed/{MODEL}_seed{s}.pt" for s in range(5)]
         + ["data/interim/p4/donors_8640.npz", "data/processed/algeria_sites.csv",
            "site/dashboard/index.html", "site/dashboard/data", "site/flood/index.html"])


def make(out: Path) -> None:
    missing = [p for p in PATHS if not (ROOT / p).exists()]
    if missing:
        sys.exit("missing, build them first: " + ", ".join(missing))
    with tarfile.open(out, "w") as tar:      # the data is mostly PNG and compressed arrays; gzip would gain little
        for p in PATHS:
            tar.add(ROOT / p, arcname=p)
    print(f"{out}  {out.stat().st_size / 1e6:.0f} MB  sha256 {sha256(out)}")


def sha256(f: Path) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path) -> None:
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL) as r, open(dest, "wb") as fh:
        total, done = int(r.headers.get("content-length", 0)), 0
        for chunk in iter(lambda: r.read(1 << 20), b""):
            fh.write(chunk)
            done += len(chunk)
            if total and (done // (1 << 20)) % 25 == 0:          # a line every 25 MB, readable in logs too
                print(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
    print("  done")


def unpack(f: Path) -> None:
    with tarfile.open(f) as tar:
        names = tar.getnames()
        if any(n.startswith("/") or ".." in Path(n).parts for n in names):
            sys.exit("refusing to unpack: the file has paths outside the repo")
        tar.extractall(ROOT, filter="data") if sys.version_info >= (3, 12) else tar.extractall(ROOT)
    print(f"unpacked {len(names)} files into {ROOT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["get", "make"])
    ap.add_argument("--file", type=Path, help="get: use this bundle instead of downloading; make: where to write it")
    a = ap.parse_args()
    if a.action == "make":
        make(a.file or ROOT / NAME)
        return
    f = a.file
    if f is None:
        f = ROOT / "data" / NAME
        f.parent.mkdir(parents=True, exist_ok=True)
        download(f)
    unpack(f)
    if a.file is None:
        f.unlink()


if __name__ == "__main__":
    main()
