#!/bin/sh
# Container start: fetch the demo data if any part is missing, then serve on $PORT.
set -e
cd /app
if [ ! -d site/dashboard/data ] || [ ! -f data/interim/p4/donors_8640.npz ] || [ ! -d runs/phase2/mixed ]; then
  python scripts/demo_bundle.py get
fi
exec python scripts/serve_site.py --host 0.0.0.0 --port "${PORT:-8000}"
