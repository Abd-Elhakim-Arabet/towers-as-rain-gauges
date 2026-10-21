"""Figures, data files and numbers for the Phase 4 chapter of docs/explainer.

    python scripts/export_explainer_phase4.py

fig_p4_network.png     towers and simulated links, Algiers and Ghardaia, over the storm total
fig_p4_steps_<city>.png one storm hour: satellite, simulated truth, AI map (all links, 3% of links)
p4_link_<city>.dat     one link through the storm: true rain, AI rain, rain loss, wet antenna, total loss
p4_city_<city>.dat     city-average hourly rain: truth, AI map, classical map, satellite
p4_density.dat         map quality vs share of links, both cities
p4_numbers.tex         every number quoted in the text
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "explainer"
P4 = ROOT / "data" / "processed" / "phase4"
RES = ROOT / "results"
LEVELS = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
CMAP = ListedColormap(["#dbe9f6", "#b5d2ec", "#86b4de", "#5895cc", "#2a78d6", "#1d5fa8", "#123f73"])
CMAP.set_under("white")
CMAP.set_bad("#e6e6e6")
NORM = BoundaryNorm(LEVELS, CMAP.N)
TOT_LEVELS = [1, 5, 10, 20, 30, 50, 80, 120]
TOT_NORM = BoundaryNorm(TOT_LEVELS, CMAP.N)
CITY = {"algiers": "Algiers, 9--10 Nov 2001", "ghardaia": "Ghardaïa, 29 Sep--1 Oct 2008"}
plt.rcParams.update({"font.size": 8, "font.family": "serif"})


def cities():
    return [c for c in ("algiers", "ghardaia") if (P4 / f"sim_{c}.npz").exists()]


def extent(z):
    lat, lon = z["lat"], z["lon"]
    d = float(lat[1] - lat[0]) / 2
    return [lon[0] - d, lon[-1] + d, lat[0] - d, lat[-1] + d]


def fig_network():
    cs = cities()
    fig, axes = plt.subplots(1, len(cs), figsize=(7.0, 3.0), squeeze=False,
                             gridspec_kw={"width_ratios": [2.0, 1.0][:len(cs)]})
    sites = pd.read_csv(ROOT / "data" / "processed" / "algeria_sites.csv")
    band_col = [(0, 20, "#c96a00", "< 20 GHz"), (20, 30, "#1baf7a", "20--30 GHz"), (30, 99, "#1d5fa8", "> 30 GHz")]
    for ax, c in zip(axes[0], cs):
        z = np.load(P4 / f"sim_{c}.npz", allow_pickle=True)
        links = pd.read_csv(RES / f"phase4_links_{c}.csv")
        tot = z["truth"].sum(0)
        im = ax.imshow(np.where(tot < 1, np.nan, tot), origin="lower", extent=extent(z), cmap=CMAP, norm=TOT_NORM,
                       alpha=0.55, interpolation="nearest")
        lat0, lat1, lon0, lon1 = z["lat"][0], z["lat"][-1], z["lon"][0], z["lon"][-1]
        s = sites[(sites.lat.between(lat0, lat1)) & (sites.lon.between(lon0, lon1))]
        ax.scatter(s.lon, s.lat, s=1.2, c="#6b7780", linewidths=0, label="tower sites (OpenCelliD)")
        for lo, hi, col, lab in band_col:
            g = links[(links.freq_ghz >= lo) & (links.freq_ghz < hi)]
            for r in g.itertuples():
                ax.plot([r.site_0_lon, r.site_1_lon], [r.site_0_lat, r.site_1_lat], color=col, lw=0.35, alpha=0.8)
            ax.plot([], [], color=col, lw=1.2, label=f"simulated link, {lab}")
        ax.set_title(f"{CITY[c].replace('--', '–')}\n{len(s)} sites, {len(links)} links", fontsize=7.5)
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(lat0, lat1)
        ax.set_aspect(1 / np.cos(np.radians((lat0 + lat1) / 2)))
        ax.tick_params(labelsize=6)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.45, -0.06),
               markerscale=4)
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.75, pad=0.02)
    cb.set_label("simulated storm total (mm)")
    fig.savefig(DOC / "fig_p4_network.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_steps(c):
    z = np.load(P4 / f"sim_{c}.npz", allow_pickle=True)
    truth = z["truth"]
    h = int(np.argmax(truth.mean((1, 2))))
    hour = pd.Timestamp(z["hours"][h])
    # satellite: mean of the two half-hours of that hour, on its own 10-km grid
    it = pd.DatetimeIndex(z["imerg_time"])
    k = np.where((it >= hour) & (it < hour + pd.Timedelta("1h")))[0]
    sat = z["imerg"][k].mean(0)
    ilat, ilon = z["imerg_lat"], z["imerg_lon"]
    ex = extent(z)
    links = pd.read_csv(RES / f"phase4_links_{c}.csv")
    sparse = links[links.link_id.isin(set(z["sparse_ids"]))]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 3.9))
    axes = axes.ravel()
    panels = [("1. what the satellite saw (10-km cells)", None),
              ("2. simulated truth (500-m cells)", truth[h]),
              (f"3. AI map from all {len(links)} links", z["ai"][h]),
              (f"4. AI map from {len(sparse)} links (3%, orange)", z["ai_sparse"][h])]
    for ax, (title, f) in zip(axes, panels):
        if f is None:
            d = 0.05
            ax.imshow(np.where(sat < 0.1, 0, sat), origin="lower", cmap=CMAP, norm=NORM, interpolation="nearest",
                      extent=[ilon[0] - d, ilon[-1] + d, ilat[0] - d, ilat[-1] + d])
        else:
            ax.imshow(np.where(np.isfinite(f) & (f < 0.1), 0, f), origin="lower", cmap=CMAP, norm=NORM,
                      interpolation="nearest", extent=ex)
        if "orange" in title:
            for r in sparse.itertuples():
                ax.plot([r.site_0_lon, r.site_1_lon], [r.site_0_lat, r.site_1_lat], color="#eb6834", lw=0.8)
        ax.set_xlim(ex[0], ex[1])
        ax.set_ylim(ex[2], ex[3])
        ax.set_aspect(1 / np.cos(np.radians((ex[2] + ex[3]) / 2)))
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    cb = fig.colorbar(sm, ax=axes.tolist(), shrink=0.75, pad=0.01, ticks=LEVELS)
    cb.set_label("mm/h")
    fig.suptitle(f"{CITY[c].replace('--', '–')}: hour starting {hour:%d %b %H:%M} UTC", fontsize=8, y=1.02)
    fig.savefig(DOC / f"fig_p4_steps_{c}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return hour


def link_dat(c):
    z = np.load(P4 / f"sim_{c}.npz", allow_pickle=True)
    i = 0                                             # wettest sub-link of the storm
    tl = z["tl_example"][i].astype(float)
    base = np.nanmedian(tl[:600])                     # dry level before the storm
    df = pd.DataFrame({"rtrue": z["rtrue_example"][i], "rai": z["rai_example"][i], "rain_loss": z["a_example"][i],
                       "wet_antenna": z["w_example"][i], "excess": tl - base})
    h = df.groupby(np.arange(len(df)) // 30).mean()          # 30-min means, readable on paper
    h.insert(0, "hour", np.arange(len(h)) * 0.5)
    h.round(3).to_csv(DOC / f"p4_link_{c}.dat", sep=" ", index=False, na_rep="nan")
    links = pd.read_csv(RES / f"phase4_links_{c}.csv").set_index("link_id")
    lid = str(z["example_ids"][i]).rsplit("_", 1)[0]
    r = links.loc[lid]
    return {"id": lid, "length_km": float(r.length_km), "freq_ghz": float(r.freq_ghz),
            "true_mm": float(np.nansum(z["rtrue_example"][i]) / 60), "ai_mm": float(np.nansum(z["rai_example"][i]) / 60)}


def city_dat(c):
    z = np.load(P4 / f"sim_{c}.npz", allow_pickle=True)
    hrs = pd.DatetimeIndex(z["hours"])
    it = pd.DatetimeIndex(z["imerg_time"])
    lat, lon = z["lat"], z["lon"]
    ilat, ilon = z["imerg_lat"], z["imerg_lon"]
    m = (ilat[:, None] >= lat[0] - 0.05) & (ilat[:, None] <= lat[-1] + 0.05) & \
        (ilon[None] >= lon[0] - 0.05) & (ilon[None] <= lon[-1] + 0.05)
    sat = pd.Series(z["imerg"][:, m].mean(1), index=it).resample("1h").mean().reindex(hrs).values
    df = pd.DataFrame({"hour": np.arange(len(hrs)), "truth": z["truth"].mean((1, 2)),
                       "ai": np.nanmean(z["ai"], (1, 2)), "classical": np.nanmean(z["classical"], (1, 2)),
                       "ai_sparse": np.nanmean(z["ai_sparse"], (1, 2)), "satellite": sat})
    df.round(3).to_csv(DOC / f"p4_city_{c}.dat", sep=" ", index=False, na_rep="nan")
    return hrs[0]


def main():
    L = []

    def cmd(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    def pct(x, sign=False):
        s = f"{x:+.0f}" if sign else f"{x:.0f}"
        return s.replace("-", "$-$").replace("+", "$+$") + r"\%"

    fig_network()
    storms = json.loads((RES / "phase4_storms.json").read_text())
    tex = json.loads((RES / "phase4_rain_texture.json").read_text())
    cmd("PfourRadarSigma", f"{tex['sigma_log_ratio']:.2f}")
    cmd("PfourRadarWet", f"{100 * tex['wet_share_mean']:.0f}\\%")
    cmd("PfourRadarCorrA", f"{tex['log_ratio_corr']['15min']:.2f}")
    cmd("PfourRadarCorrB", f"{tex['log_ratio_corr']['30min']:.2f}")
    dens = {}
    for c in cities():
        K = {"algiers": "Alg", "ghardaia": "Gha"}[c]
        r = json.loads((RES / f"phase4_{c}.json").read_text())
        hour = fig_steps(c)
        ex = link_dat(c)
        city_dat(c)
        st = storms[c]
        cmd(f"Pfour{K}Sites", str(r["n_sites"]))
        cmd(f"Pfour{K}Links", str(r["n_links"]))
        cmd(f"Pfour{K}Sub", str(r["n_sub_links"]))
        cmd(f"Pfour{K}Reliable", str(r["n_reliable"]))
        cmd(f"Pfour{K}LenMed", f"{r['link_length_km_pct'][2]:.1f}")
        cmd(f"Pfour{K}LenLo", f"{r['link_length_km_pct'][1]:.1f}")
        cmd(f"Pfour{K}LenHi", f"{r['link_length_km_pct'][3]:.1f}")
        cmd(f"Pfour{K}SatTotal", f"{st['city_total_mm']:.0f}")
        cmd(f"Pfour{K}SatPeak", f"{st['city_peak_mm_h']:.1f}")
        cmd(f"Pfour{K}SatPeakHour", pd.Timestamp(st["city_peak_hour"]).strftime("%-d %b %H:%M"))
        cmd(f"Pfour{K}SatCellMax", f"{st['box_max_cell_total_mm']:.0f}")
        days = st["city_daily_mm"]
        if c == "ghardaia":
            cmd("PfourGhaSatDayA", f"{days['2008-09-29']:.0f}")
            cmd("PfourGhaSatDayB", f"{days['2008-10-01']:.0f}")
        cmd(f"Pfour{K}Hour", hour.strftime("%-d %b %H:%M"))
        t = r["texture_check"]
        cmd(f"Pfour{K}Sigma", f"{t['sigma_log_ratio']:.2f}")
        cmd(f"Pfour{K}Wet", f"{100 * t['wet_share_mean']:.0f}\\%")
        cmd(f"Pfour{K}CorrA", f"{t['log_ratio_corr_15min']:.2f}")
        cmd(f"Pfour{K}CorrB", f"{t['log_ratio_corr_30min']:.2f}")
        cmd(f"Pfour{K}Speed", f"{np.hypot(*r['motion_km_h_median']):.0f}")
        for name, key in (("ai", "AI"), ("classical", "Cl")):
            l = r["links"][name]
            cmd(f"Pfour{K}{key}LinkR", f"{l['hourly_r']:.2f}")
            cmd(f"Pfour{K}{key}LinkBias", pct(l["bias_pct"], True))
            full = r["maps"]["1.00"][name]
            cmd(f"Pfour{K}{key}MapR", f"{full['pixel_hourly_r']:.2f}")
            cmd(f"Pfour{K}{key}MapBias", pct(full["event_total_bias_pct"], True))
            cmd(f"Pfour{K}{key}CityR", f"{full['city_mean_hourly_r']:.2f}")
            cmd(f"Pfour{K}{key}TotR", f"{full['event_total_r']:.2f}")
            cmd(f"Pfour{K}{key}CityTot", f"{full['city_total_est_mm']:.0f}")
            sp = r["maps"]["0.03"][name]
            cmd(f"Pfour{K}{key}SparseR", f"{sp['pixel_hourly_r']:.2f}")
            cmd(f"Pfour{K}{key}SparseCover", pct(100 * sp["covered_share"]))
            cmd(f"Pfour{K}{key}SparseBias", pct(sp["event_total_bias_pct"], True))
        cmd(f"Pfour{K}TruthTot", f"{r['maps']['1.00']['ai']['city_total_truth_mm']:.0f}")
        cmd(f"Pfour{K}SparseN", str(r["maps"]["0.03"]["n_links"]))
        cmd(f"Pfour{K}ExLen", f"{ex['length_km']:.1f}")
        cmd(f"Pfour{K}ExFreq", f"{ex['freq_ghz']:.0f}")
        cmd(f"Pfour{K}ExTrue", f"{ex['true_mm']:.0f}")
        cmd(f"Pfour{K}ExAI", f"{ex['ai_mm']:.0f}")
        dens[c] = pd.DataFrame({"frac": [float(f) for f in r["maps"]],
                                f"{c}_n": [m["n_links"] for m in r["maps"].values()],
                                f"{c}_ai_r": [m["ai"]["pixel_hourly_r"] for m in r["maps"].values()],
                                f"{c}_cl_r": [m["classical"]["pixel_hourly_r"] for m in r["maps"].values()],
                                f"{c}_cover": [m["ai"]["covered_share"] for m in r["maps"].values()]}).set_index("frac")
        # hourly correlation by distance to the nearest link (all links)
        for band, letter in (("0-1km", "A"), ("1-2km", "B"), ("2-5km", "C"), ("5-10km", "D")):
            v = r["maps"]["0.03"]["ai"]["by_distance"][band]
            ok = v["hourly_r"] is not None and np.isfinite(v["hourly_r"])
            cmd(f"Pfour{K}Dist{letter}", f"{v['hourly_r']:.2f}" if ok else "--")
            cmd(f"Pfour{K}DistShare{letter}", pct(100 * v["share_of_city"]))
    if dens:
        d = pd.concat(dens.values(), axis=1)
        d.reset_index().round(3).to_csv(DOC / "p4_density.dat", sep=" ", index=False, na_rep="nan")
    (DOC / "p4_numbers.tex").write_text("% generated by scripts/export_explainer_phase4.py -- do not edit\n"
                                        + "\n".join(L) + "\n")
    print("wrote", len(L), "numbers")
    summary_md()


def summary_md():
    """results/phase4.md: the numbers in plain tables."""
    tex = json.loads((RES / "phase4_rain_texture.json").read_text())
    storms = json.loads((RES / "phase4_storms.json").read_text())
    L = ["# Phase 4: simulated Algerian network (SIMULATION, not validation)", "",
         "Real tower positions (OpenCelliD) + plausible microwave tree per operator (30% fibre hubs, hop lengths from",
         "Gothenburg/NL, frequency/polarisation from a real link of similar length) + IMERG storms downscaled to",
         "500 m / 1 min (patchiness measured on Italian radar at 1 km) + real dry recordings from eastern Italian links",
         "(the AI = final recipe trained on western Italy never saw them) + wet antennas + -100 dBm floor + 1-dB rounding.",
         "", f"Real radar texture (Nov 2021): sigma {tex['sigma_log_ratio']:.2f}, wet {tex['wet_share_mean']:.2f}, "
         f"corr 15/30 min {tex['log_ratio_corr']['15min']:.2f}/{tex['log_ratio_corr']['30min']:.2f}", ""]
    for c in cities():
        r = json.loads((RES / f"phase4_{c}.json").read_text())
        st = storms[c]
        t = r["texture_check"]
        L += [f"## {r['event']} ({c})", "",
              f"IMERG city total {st['city_total_mm']:.0f} mm, peak {st['city_peak_mm_h']:.1f} mm/h at {st['city_peak_hour']}; "
              f"{r['n_sites']} sites -> {r['n_links']} links ({r['n_sub_links']} sub-links), median length "
              f"{r['link_length_km_pct'][2]:.1f} km; reliability keeps {r['n_reliable']}.",
              f"Simulated texture: sigma {t['sigma_log_ratio']:.2f}, wet {t['wet_share_mean']:.2f}, corr 15/30 min "
              f"{t['log_ratio_corr_15min']:.2f}/{t['log_ratio_corr_30min']:.2f}.", "",
              "| method | link hourly r | link total bias | map hourly r | map total bias | city-mean hourly r |",
              "|---|---|---|---|---|---|"]
        for m in ("ai", "classical"):
            l, mp = r["links"][m], r["maps"]["1.00"][m]
            L.append(f"| {m} | {l['hourly_r']:.2f} | {l['bias_pct']:+.0f}% | {mp['pixel_hourly_r']:.2f} | "
                     f"{mp['event_total_bias_pct']:+.0f}% | {mp['city_mean_hourly_r']:.2f} |")
        L += ["", "| share of links | n links | AI map r | classical map r | covered |", "|---|---|---|---|---|"]
        for f, mp in r["maps"].items():
            L.append(f"| {float(f):.0%} | {mp['n_links']} | {mp['ai']['pixel_hourly_r']:.2f} | "
                     f"{mp['classical']['pixel_hourly_r']:.2f} | {mp['ai']['covered_share']:.0%} |")
        L.append("")
    (RES / "phase4.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
