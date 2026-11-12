# Phase 7 plan: Live watch (any city, watched as it happens)

Status: built 12 Nov 2026 (steps 1-8 and tests); explainer Part 7 still to write.

Phases 0 to 6 replay two past storms. Phase 7 adds a second tab to the control room that does not replay anything:
you type a city, the system sets itself up there, and it keeps watching the rain and the flood model, raising an alarm
(on screen and by email) if the town starts to flood.

## 1. What the user sees

1. A "Live watch" tab next to the replay in the control room.
2. A search box. Type a city name, pick the right match from a short list (a name alone is ambiguous: "Tunis" also
   matches the whole country), and the map flies there.
3. The system looks in its list of known live data sources for that place.
   - Found: it connects and starts watching.
   - Not found (the normal case today, no operator publishes link data): the user chooses between
     **My own data** and **Simulation**.
4. Setup runs with a visible progress list (terrain, land cover, soil, towers, links, first rain map). A new city
   takes a few minutes the first time and is cached after that.
5. The watch screen: towers and links coloured by the rain they see, the rain map, flood depth, the alert level
   (normal, watch, flood warning, flooding), the last 24 hours on a timeline, and an event log.
6. An email field. When the alert level reaches flood warning or flooding, an email goes out once per event.
7. A "Run a drill" button that pushes a past or design storm through the same live chain at high speed, so the alarm
   and the email can be shown on a dry day. Drills are labelled as drills everywhere, including in the email.

## 2. The three data modes

### 2.1 Known live source
`configs/live_sources.yaml` lists adapters by area. It ships empty of real sources, because none exist publicly today.
It is the place an operator agreement would be registered.

### 2.2 My own data (adapter)
We publish one standard format, and the data owner writes a small translator to it.

Standard format:
- `links`: link_id, lat_a, lon_a, lat_b, lon_b, freq_ghz, polarisation (H or V), operator (optional)
- `readings`: time (UTC), link_id, rsl_dbm, tsl_dbm (or total loss in dB); any step from 10 s to 15 min;
  15-minute min/max logs are accepted too

Two ways to connect:
- **No code:** a URL (or file path) that already serves the standard format as CSV or JSON. We poll it.
- **With code:** a Python file in `adapters/` with two functions, `links()` and `fetch(since)`. A commented
  template (`adapters/template.py`) and a checker (`python -m watch.check_adapter adapters/mine.py`) come with it.
  The checker says in plain words what is wrong (missing column, wrong unit, times not UTC, link without readings).

Safety: adapter code is never uploaded through the web page. It is a file placed on the server by whoever runs it;
the page only lists the adapters that are installed. Uploading code through a browser would let anyone run programs
on the machine.

### 2.3 Simulation (real weather, simulated links)
Honest split, same as the rest of the project:
- **Real:** the terrain, land cover, soil, buildings, tower positions where we can get them, and the weather itself.
- **Simulated:** the links between towers and their signals.

How it works:
1. **Live rain** comes from Open-Meteo (free, no key, worldwide, 15-minute steps, recent past and next hours).
   It is weather-service rain (radar-fed where a country has radar, model-based elsewhere). Tested from this PC.
2. That rain is sharpened to about 500 m and 1 minute with the Phase 4 method (`targ.simrain`).
3. It is pushed through the simulated links with the ITU formula, wet-antenna loss and real dry-day radio noise
   (`targ.simsignal`), exactly as in Phase 4.
4. The AI model reads the rain back from those signals, and the rest of the chain runs on that. So the simulation tests the chain live. It does not measure accuracy, and the page says so.

## 3. Towers for any city

Measured from this PC: OpenStreetMap has 145 mast locations in Algiers where OpenCelliD gave us 4,092 sites, and
the OpenCelliD API is quota-limited (the Algiers sweep took days). So no single source works worldwide. Three tiers:

1. **OpenCelliD** if a sweep of the area is already cached, or as a slow background upgrade within the daily quota.
2. **OpenStreetMap** mast and antenna locations (Overpass API, free; it is sometimes busy, so we retry and cache).
3. **Estimated sites** to fill the gap up to a realistic density, placed where the built-up land is (ESA WorldCover),
   with typical urban spacing per operator.

Every tower is tagged real or estimated, the map draws them differently, and the page shows the counts.
Links are then built per operator with the Phase 4 tree builder (`targ.simnet`).

## 4. Flood model for any city

- `floodrisk.ground.build` already crops global sources (Copernicus 30 m terrain, WorldCover, HYSOGs soil). It is
  generalised from the two hard-coded cities to any box. Soil falls back to a cautious default if the soil server
  is unreachable.
- Area limit: about 30 x 30 km around the chosen centre at 30 m (60 m above that), so one GPU keeps up easily.
- The solver (`floodrisk.solver.simulate`) runs a whole storm in one call. It is split into "set up" and "advance by
  N minutes, keeping the water state", and `simulate` becomes a thin wrapper. A test checks the Phase 5 results stay
  identical, so the replay numbers in the paper do not move.
- State (water depth, flows, soil wetness) is saved to disk so a restart continues where it stopped.

## 5. Alert levels (same four as the replay)

- **Normal**
- **Watch:** rain so far reaches 50% of the flash-flood rain threshold. Thresholds are computed once per city at
  setup from design storms, as in Phase 5.
- **Flood warning:** a look-ahead run (current water + the next 2 hours of expected rain) shows the town flooding.
- **Flooding:** the model shows at least 5 ha of built-up land under 30 cm of water now (the Phase 5 rule; the
  number is a setting per watch).

Known limits, shown on the page:
- The flood model is not calibrated for a new city and has no storm drains or river defences.
- A flood that comes from far upstream, outside the watched box, is missed (the Ghardaia lesson). Setup warns when
  the rivers entering the box drain a large outside area.
- The AI reads heavy rain about 10% low; a fixed correction did not remove it (section 9), so the alarm may come
  late. The classical method is not used live: it needs 30 minutes of future signal.

## 6. Email

- Sent with the standard library (`smtplib`). Settings come from `.env`: SMTP_HOST, SMTP_PORT, SMTP_USER,
  SMTP_PASSWORD, ALERT_FROM. `.env` is never committed; `.env.example` documents the names.
- Without settings, emails are written to `data/live/outbox/` as files, so the feature can be shown offline.
- One email per event and level (no repeats every cycle), plus an all-clear. Subject, city, time, level, hectares
  flooded, worst areas, a link to the watch page, and the line "simulated links" or "DRILL" when that applies.
- A "Send test email" button.

## 7. Software layout

```
src/watch/
  geocode.py     city name -> candidates and a box (Nominatim)
  towers.py      the three tiers, cached
  weather.py     live and past rain (Open-Meteo)
  sources.py     registry, adapter loading, standard-format validation
  engine.py      one watch: ingest -> rain per link -> rain map -> flood step -> alert level
  alerts.py      levels, events, email
  store.py       state, history and event log under data/live/<city>/ (not in git)
  api.py         FastAPI: search, create/stop watch, state, frames, drill, test email
adapters/        template.py, http_csv.py, README
scripts/serve_site.py   gains the API and the background watcher (static pages keep working as today)
site/watch/      the Live watch page, same look as the control room
tests/           format checker, solver stepping equals whole-run, alert logic, email outbox, a drill end to end
```

The watcher is one background loop in the server: every 5 minutes per watched city (every few seconds in a drill).
Watches survive a restart. It only watches while the server is running, and on this lab PC that means while the
session is logged in.

Huawei switches stay as they are: storage and compute go through `targ.backends`, so OBS and ECS remain a setting.

## 8. Build order (one commit per step)

1. Solver stepping + equality test.
2. Any-city ground build + geocoding.
3. Towers (three tiers) + links.
4. Live weather + streaming signal simulation + rain map.
5. Watch engine, alert levels, state on disk.
6. Email + outbox + drill.
7. Adapter format, template, checker, no-code URL source.
8. API + Live watch page + tab in the control room.
9. Tests, README section, explainer Part 7.

## 9. Decisions (taken 12 Nov 2026)

1. **Rain for the alarm: the AI only.** The classical method looks 30 minutes ahead, so it cannot run live. A
   correction of the AI's low heavy-rain reading was fitted on the training months only and scored on the untouched
   test month (`scripts/phase7_heavy_rain_correction.py`). It improved rain overall (bias -7% -> -2%) but not the
   heavy hours that decide the alarm (-10% -> -12%), and a heavy-rain gain overshot the test month; the shortfall
   changes from month to month. So no correction is applied. The real fix is retraining with more weight on heavy
   rain.
2. **Email:** the recipient is set in `.env` (ALERT_TO), never in the repository. The sending account's SMTP settings
   are filled in by the team in `.env`; until then emails go to the outbox folder.
3. **Area:** about 30 x 30 km around the city centre.
