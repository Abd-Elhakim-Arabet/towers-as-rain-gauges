"""Phase 5: flood risk from rain maps.

A separate package from `targ` (Phases 0-4): nothing in `targ` imports it and it does not import `targ`.
The only link to Phase 4 is `floodrisk.rain.Phase4Rain`, which reads the saved Phase 4 rain fields.

Modules
  config   the two cities: domains, outlets, what was observed
  ground   terrain, land cover and soil on one grid (download + cache)
  terrain  pit filling, D8 flow directions, upstream area, basin outlines
  runoff   curve numbers, roughness, and the rain that does not soak in
  solver   2D local-inertial shallow-water solver (rain on grid), PyTorch
  rain     rain sources: Phase4Rain (the link), ImergRain, DesignStorm
  hazard   depth x speed hazard classes, exposure, alarms, scores
"""
