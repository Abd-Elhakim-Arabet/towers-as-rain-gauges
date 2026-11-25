#!/bin/sh
# Container start: fetch the demo data if any part is missing, then serve on $PORT.
set -e
cd /app
# Started as root: make sure the data volume belongs to the app user (a volume created by an older image, or
# by Docker itself, is owned by root and the download fails with "Permission denied"), then run as that user.
if [ "$(id -u)" = 0 ]; then
  mkdir -p data
  [ "$(stat -c %U data)" = app ] || chown -R app:app data
  export HOME=/home/app                      # setpriv keeps root's HOME otherwise
  exec setpriv --reuid=app --regid=app --init-groups "$0" "$@"
fi
if [ ! -d site/dashboard/data ] || [ ! -f data/interim/p4/donors_8640.npz ] || [ ! -d runs/phase2/mixed ]; then
  python scripts/demo_bundle.py get --keep     # the file stays in the volume, so a new container only unpacks it
fi
exec python scripts/serve_site.py --host 0.0.0.0 --port "${PORT:-8000}"
