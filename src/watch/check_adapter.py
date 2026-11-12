"""Check an adapter (or two table addresses) against the standard format, in plain words.

    python -m watch.check_adapter adapters/mine.py
    python -m watch.check_adapter https://example.org/links.csv "https://example.org/readings.csv?since={since}"
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from . import sources


def main(argv: list[str]) -> int:
    if len(argv) == 1 and argv[0].endswith(".py"):
        spec = importlib.util.spec_from_file_location("adapter_under_test", Path(argv[0]))
        ad = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ad)
    elif len(argv) == 2:
        ad = sources.TableAdapter(*argv)
    else:
        print(__doc__)
        return 2
    try:
        links = ad.links()
    except Exception as e:
        print(f"links() failed: {type(e).__name__}: {e}")
        return 1
    problems = sources.check_links(links)
    print(f"links: {len(links)} rows")
    if not problems:
        since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=6)
        try:
            r = ad.fetch(since)
        except Exception as e:
            print(f"fetch(since) failed: {type(e).__name__}: {e}")
            return 1
        print(f"readings in the last 6 hours: {len(r)} rows")
        problems = sources.check_readings(r, links) if len(r) else ["readings: nothing came back for the last 6 hours"]
    for p in problems:
        print("  - " + p)
    print("OK: the data fits the standard format." if not problems else f"{len(problems)} thing(s) to fix.")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
