"""Download and verify the open datasets listed in configs/datasets.yaml.

Usage:
    python scripts/download_data.py openmrg openrainer netherlands_cml
    python scripts/download_data.py --list

Zenodo files are checked against the MD5 that the Zenodo API publishes.
Downloads resume when interrupted (curl -C -). A manifest with sizes,
checksums and dates is written to data/raw/MANIFEST.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "MANIFEST.json"
FOURTU_API = "https://data.4tu.nl/v2/articles/{uuid}/files"


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # -C - resumes a partial file; --retry survives the flaky lab network.
    subprocess.run(
        ["curl", "-sS", "-L", "--fail", "--retry", "10", "--retry-delay", "5",
         "-C", "-", "-o", str(dest), url],
        check=True,
    )


def zenodo_files(record: int) -> dict[str, dict]:
    meta = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60).json()
    return {
        f["key"]: {"url": f["links"]["self"], "size": f["size"],
                   "md5": f["checksum"].removeprefix("md5:")}
        for f in meta["files"]
    }


def fourtu_files(doi: str) -> dict[str, dict]:
    uuid = doi.split("/")[-1]
    files = requests.get(FOURTU_API.format(uuid=uuid), timeout=60).json()
    return {f["name"]: {"url": f["download_url"], "size": f["size"],
                        "md5": f.get("computed_md5") or None} for f in files}


def fetch(name: str, spec: dict, with_optional: bool) -> list[dict]:
    if spec.get("auth"):
        print(f"[{name}] needs credentials ({spec['auth']}); use the phase-specific script.")
        return []
    host = spec.get("host")
    if host == "zenodo":
        remote = zenodo_files(spec["record"])
    elif host == "4tu":
        remote = fourtu_files(spec["doi"])
    else:
        print(f"[{name}] host '{host}' is handled by a phase-specific script.")
        return []

    wanted = list(spec.get("files", []))
    if with_optional:
        wanted += spec.get("optional_files", [])

    records = []
    for fname in wanted:
        info = remote[fname]
        dest = RAW / name / fname
        if dest.exists() and dest.stat().st_size == info["size"]:
            print(f"[{name}] {fname}: already complete")
        else:
            print(f"[{name}] {fname}: downloading {info['size'] / 1e6:.1f} MB")
            curl(info["url"], dest)
        size = dest.stat().st_size
        got_md5 = md5sum(dest)
        ok = size == info["size"] and (info["md5"] is None or got_md5 == info["md5"])
        print(f"[{name}] {fname}: size {'OK' if size == info['size'] else 'MISMATCH'}, "
              f"md5 {'OK' if ok else 'MISMATCH'}")
        records.append({
            "dataset": name, "file": fname, "doi": spec.get("doi"),
            "version": spec.get("version"), "license": spec.get("license"),
            "bytes": size, "md5": got_md5, "verified": ok,
            "downloaded": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        })
    return records


def update_manifest(records: list[dict]) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for r in records:
        manifest[f"{r['dataset']}/{r['file']}"] = r
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> int:
    registry = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--optional", action="store_true", help="also fetch optional large files")
    args = ap.parse_args()

    if args.list or not args.datasets:
        for k, v in registry.items():
            print(f"{k:18s} phase {v['phase']}  {v['license']:22s} {v['title']}")
        return 0

    records = []
    for name in args.datasets:
        records += fetch(name, registry[name], args.optional)
    update_manifest(records)
    return 0 if all(r["verified"] for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
