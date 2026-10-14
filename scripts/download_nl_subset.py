"""Extract selected members of the 12.6 GB Netherlands CML zip over HTTP range requests.

The full archive would take ~10 h on our network, so we read the zip's central
directory remotely and pull only the members we need.

    python scripts/download_nl_subset.py --list            # show members
    python scripts/download_nl_subset.py --match 201106     # extract members containing "201106"
"""

from __future__ import annotations

import argparse
import io
import shutil
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "netherlands_cml"
UUID = "be252844-b672-471e-8d69-27269a862ec1"
ZIPS = {
    "RawCMLdata.zip": "ef14e92c-5847-4958-b535-215c21ca72b8",
    "IDRawCMLdata.zip": "e38c0fe2-6ad8-49fd-86ca-7f085e66e46b",
}


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file backed by HTTP Range requests."""

    READAHEAD = 1 << 20   # 1 MB: fewer requests while walking the central directory

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        self.cache_start, self.cache = 0, b""
        self.session = requests.Session()
        head = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        self.size = int(head.headers["Content-Range"].split("/")[-1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        # Serve small reads (zip headers) from a read-ahead cache.
        c0 = self.cache_start
        if c0 <= self.pos and self.pos + n <= c0 + len(self.cache):
            out = self.cache[self.pos - c0:self.pos - c0 + n]
            self.pos += len(out)
            return out
        want = max(n, self.READAHEAD)
        end = min(self.pos + want, self.size) - 1
        for attempt in range(8):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"},
                                     timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                # 4TU refuses connections when hammered; back off exponentially.
                wait = min(2 ** attempt * 5, 600)
                print(f"range read failed ({e.__class__.__name__}); retrying in {wait}s")
                time.sleep(wait)
        else:
            raise IOError(f"range read failed at {self.pos}")
        self.cache_start, self.cache = self.pos, r.content
        out = r.content[:n]
        self.pos += len(out)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="RawCMLdata.zip", choices=list(ZIPS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--match", nargs="*", default=[])
    args = ap.parse_args()

    url = f"https://data.4tu.nl/file/{UUID}/{ZIPS[args.zip]}"
    zf = zipfile.ZipFile(HTTPRangeFile(url))
    members = zf.infolist()
    if args.list:
        for m in members:
            print(f"{m.file_size / 1e6:10.1f} MB  {m.compress_size / 1e6:10.1f} MB(z)  {m.filename}")
        print(f"{len(members)} members")
        return
    for m in members:
        if args.match and not any(s in m.filename for s in args.match):
            continue
        dest = OUT / args.zip.removesuffix(".zip") / m.filename
        if dest.exists() and dest.stat().st_size == m.file_size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"extracting {m.filename} ({m.compress_size / 1e6:.1f} MB compressed)")
        with zf.open(m) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)


if __name__ == "__main__":
    main()
