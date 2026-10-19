"""Figures and data files for the Phase 3 chapter of docs/explainer.

    python scripts/export_explainer_phase3.py

fig_nl_map.png       one stormy hour: AI link map vs KNMI radar (Netherlands)
fig_nowcast.png      a nowcast: now, forecast +60 min, what the links saw at +60 min
nowcast_csi.dat      CSI at 1 mm/h vs lead time, nowcast and persistence
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "explainer"
RUNS3 = ROOT / "runs" / "phase3"
LEVELS = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
# one-hue sequential (light -> dark blue), same family as the explainer's rain colour
CMAP = ListedColormap(["#dbe9f6", "#b5d2ec", "#86b4de", "#5895cc", "#2a78d6", "#1d5fa8", "#123f73"])
CMAP.set_under("white")
CMAP.set_bad("#e6e6e6")
NORM = BoundaryNorm(LEVELS, CMAP.N)
plt.rcParams.update({"font.size": 8, "font.family": "serif"})


def panel(ax, field, title, inside=None):
    f = np.array(field, float)
    if inside is not None:
        f = np.where(inside, f, np.nan)
    f = np.where(np.isfinite(f) & (f < 0.1), 0.0, f)
    ax.imshow(f, cmap=CMAP, norm=NORM, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#999999")


def colorbar(fig, axes):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04, aspect=40,
                      ticks=LEVELS, extend="min")
    cb.set_label("rain (mm/h); grey = no link within 10 km (no map)")


def crop(ds):
    ins = ds["inside"].values if "inside" in ds else np.isfinite(ds["now"].values)
    ys, xs = np.where(ins)
    return slice(max(ys.min() - 3, 0), ys.max() + 4), slice(max(xs.min() - 3, 0), xs.max() + 4)


def nl_map():
    d = xr.open_dataset(RUNS3 / "nl_storm_hour.nc").load()
    ys, xs = crop(d)
    d = d.isel(y=ys, x=xs)
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 3.9))
    t = pd.Timestamp(d.attrs["time_end"])
    panel(axs[0], d.ai_map.values, "Rain map from phone links (AI)", d.inside.values)
    panel(axs[1], d.radar.values, "KNMI gauge-adjusted radar", d.inside.values)
    fig.suptitle(f"Netherlands, hour ending {t:%d %B %Y %H:%M} UTC", fontsize=9)
    colorbar(fig, axs)
    fig.savefig(DOC / "fig_nl_map.png", dpi=220, bbox_inches="tight")


def nowcast():
    d = xr.open_dataset(RUNS3 / "nowcast_example.nc").load()
    ys, xs = crop(d)
    d = d.isel(y=ys, x=xs)
    inside = np.isfinite(d.now.values)
    t = pd.Timestamp(d.attrs["time"])
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.4))
    panel(axs[0], d.now.values, f"Link map at {t:%H:%M} (now)", inside)
    panel(axs[1], d.fc60.values, f"Forecast for {t + pd.Timedelta('60min'):%H:%M}", inside)
    panel(axs[2], d.obs60.values, f"Link map at {t + pd.Timedelta('60min'):%H:%M} (what happened)", inside)
    # motion arrows (every 12th pixel)
    step = 12
    yy, xx = np.mgrid[0:inside.shape[0]:step, 0:inside.shape[1]:step]
    u, v = d.u.values[::step, ::step], d.v.values[::step, ::step]
    m = inside[::step, ::step]
    axs[0].quiver(xx[m], yy[m], u[m], v[m], color="#eb6834", angles="xy", scale_units="xy", scale=0.25, width=0.006)
    for ax in axs:   # keep all three panels on the same extent
        ax.set_xlim(-0.5, inside.shape[1] - 0.5)
        ax.set_ylim(inside.shape[0] - 0.5, -0.5)
    fig.suptitle(f"Netherlands, {t:%d %B %Y} UTC. Orange arrows: storm motion estimated from the last 30 min",
                 fontsize=8, y=0.93)
    colorbar(fig, axs)
    fig.savefig(DOC / "fig_nowcast.png", dpi=220, bbox_inches="tight")
    return t


def csi_dat():
    r = json.loads((ROOT / "results" / "phase3_nowcast.json").read_text())["vs_link_maps"]
    rows = [{"lead": int(k[:-3]), "nowcast": r["nowcast"][k]["csi_1.0"], "persistence": r["persistence"][k]["csi_1.0"]}
            for k in r["nowcast"]]
    pd.DataFrame(rows).round(4).to_csv(DOC / "nowcast_csi.dat", sep=" ", index=False)


if __name__ == "__main__":
    nl_map()
    print("nowcast example", nowcast())
    csi_dat()
    print("ok")
