"""Phase 7: live watch. Pick any city, watch its rain and flood model as it happens, and raise the alarm.

Unlike Phases 4 to 6, nothing here replays a past storm. See docs/phase7/PLAN.md.

Modules
  geocode   city name -> candidates and a box
  area      terrain, soil and land cover for that box, ready for the flood model
  towers    tower sites: OpenCelliD cache, OpenStreetMap, estimated sites on built-up land
  weather   live and past rain (Open-Meteo)
  sources   where link data comes from: simulation, a standard-format feed, or an adapter
  engine    one watch: rain per link -> rain map -> flood step -> alert level
  alerts    alert levels, events and email
  store     state and history on disk (data/live/<city>/, not in git)
  api       the web API the Live watch page talks to
"""
