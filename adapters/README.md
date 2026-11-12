# Adapters: connecting real link data to the live watch

No operator publishes microwave-link data today, so the Live watch runs on simulated links by default. To use real
data, the data owner provides it in our standard format, in one of two ways:

1. **No code.** Serve two tables (CSV or JSON) at two web addresses: the links, and the readings. Paste the two
   addresses in the "My own data" form on the Live watch page and press "Check my data". The readings address may
   contain `{since}`; it is replaced by the UTC time from which we need readings.
2. **An adapter.** If the data is stored another way, copy `template.py` to `adapters/<name>.py`, fill in
   `links()` and `fetch(since)`, and check it:

   ```
   python -m watch.check_adapter adapters/<name>.py
   ```

   It then appears in the list on the page.

The format is described at the top of `template.py`. The checker explains in plain words what is wrong: a missing
column, a frequency in MHz instead of GHz, local time instead of UTC, swapped levels, links without readings.

Adapter files run on the server with its permissions, so only the person who runs the server puts them here. The web
page cannot upload code. To have a city use an adapter automatically, register it in `configs/live_sources.yaml`.
