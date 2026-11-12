"""Where the link signals come from.

Every source gives the engine the same two things:
  sub_links   one row per measured path: cml_id, link_id, length_km, freq_ghz, pol (H/V), site_0_lat/lon, site_1_lat/lon
  loss(minutes, coarse)   total path loss [dB], shape (n_sub, len(minutes)); NaN where there is no reading

SimSource   real weather pushed through simulated links (Phase 4 physics and real dry-day radio noise)
FeedSource  real readings in the standard format, from a URL/file or from an adapter in adapters/

The standard format (what a data owner has to deliver):
  links     link_id, lat_a, lon_a, lat_b, lon_b, freq_ghz, pol          pol is H or V
  readings  time, link_id, rsl_dbm, tsl_dbm      time in UTC; or loss_db in place of the two levels
"""

from __future__ import annotations

import importlib.util
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.ndimage import map_coordinates

from targ import features as feat
from targ import simrain, simsignal

ROOT = Path(__file__).resolve().parents[2]
ADAPTERS = ROOT / "adapters"
FINE_DEG = 0.005                      # ~500 m rain grid, as in Phase 4
LINK_COLS = ("link_id", "lat_a", "lon_a", "lat_b", "lon_b", "freq_ghz", "pol")


def fine_grid(box):
    lat0, lon0, lat1, lon1 = box
    return (np.round(np.arange(lat0, lat1 + 1e-9, FINE_DEG), 4), np.round(np.arange(lon0, lon1 + 1e-9, FINE_DEG), 4))


# ------------------------------------------------------------------------------------------------ simulation
class SimSource:
    kind = "simulation"

    def __init__(self, box, links: pd.DataFrame, seed: int = 0):
        self.links = links
        self.sub_links = sl = simsignal.sub_links(links)
        self.flat, self.flon = fine_grid(box)
        self.p, self.tex = simsignal.SignalParams(), simrain.TextureParams()
        a, b = feat.itu_a_b(sl.freq_ghz.values, sl.pol.values)
        pr, pc, owner = [], [], []
        for i, r in enumerate(sl.itertuples()):                     # sample points along every path
            n = max(2, int(np.ceil(r.length_km / self.p.step_km)) + 1)
            f = (np.arange(n) + 0.5) / n
            pr.append(((r.site_0_lat + f * (r.site_1_lat - r.site_0_lat)) - self.flat[0]) / FINE_DEG)
            pc.append(((r.site_0_lon + f * (r.site_1_lon - r.site_0_lon)) - self.flon[0]) / FINE_DEG)
            owner.append(np.full(n, i))
        self.pr, self.pc, owner = np.concatenate(pr), np.concatenate(pc), np.concatenate(owner)
        self.starts = np.r_[0, np.flatnonzero(np.diff(owner)) + 1]
        self.counts = np.diff(np.r_[self.starts, len(owner)])
        self.ap, self.bp, self.length = a[owner], b[owner], sl.length_km.values
        rng = np.random.default_rng(seed)
        d_tsl, d_tl, _ = simsignal.donor_bank(8640)                 # real dry-day recordings (Italy 2022)
        pick = rng.integers(0, len(d_tl), size=len(sl))
        self.d_tl, self.d_tsl = d_tl[pick], d_tsl[pick]
        self.wmax = rng.uniform(*self.p.wa_max_db, size=len(sl)).astype(np.float32)
        self.w = np.zeros(len(sl), np.float32)                      # wet-antenna loss right now
        self.rng = rng
        self.noise = None                                           # small-scale rain pattern, renewed while it rains
        yy, xx = np.meshgrid(np.arange(self.flat.size) % 256, np.arange(self.flon.size) % 256, indexing="ij")
        self._tile = [yy, xx]

    def rain_field(self, coarse_frame: np.ndarray) -> np.ndarray:
        """One minute of ~500 m rain from one coarse frame already on the fine grid: adds the small-scale pattern."""
        if not coarse_frame.any():
            return coarse_frame
        new = simrain._spectral_noise(self.rng, 256, self.tex.slope, 0.1 / FINE_DEG)
        rho = np.exp(-1.0 / self.tex.tau_min)
        self.noise = new if self.noise is None else rho * self.noise + np.sqrt(1 - rho ** 2) * new
        out = coarse_frame * np.exp(self.tex.sigma * self.noise[tuple(self._tile)] - self.tex.sigma ** 2 / 2)
        out[out < self.tex.wet_threshold] = 0.0
        return out

    def loss(self, minutes: np.ndarray, coarse) -> np.ndarray:
        """minutes: epoch minutes (UTC). coarse: watch.weather.Coarse covering them."""
        minutes = np.asarray(minutes, np.int64)
        col = minutes % self.d_tl.shape[1]
        dry, tsl = self.d_tl[:, col], self.d_tsl[:, col]
        A = np.zeros(dry.shape, np.float32)
        W = np.zeros(dry.shape, np.float32)
        frames = coarse.at(minutes * 60.0, self.flat, self.flon)
        up, down = 1 - np.exp(-1 / self.p.wa_rise_min), 1 - np.exp(-1 / self.p.wa_dry_min)
        for t in range(minutes.size):
            if frames[t].any():
                R = np.clip(map_coordinates(self.rain_field(frames[t]), [self.pr, self.pc], order=1, mode="nearest"), 0, None)
                Rm = np.add.reduceat(R, self.starts) / self.counts
                A[:, t] = np.add.reduceat(self.ap * R ** self.bp, self.starts) / self.counts * self.length
                goal = self.wmax * (1 - np.exp(-Rm / self.p.wa_rate_mm_h))
            elif self.w.max() < 1e-3:
                continue
            else:
                goal = 0.0
            self.w = self.w + np.where(goal > self.w, up, down) * (goal - self.w)
            W[:, t] = self.w
        tl = np.minimum(dry + A + W, tsl + 100.0)                   # the receiver bottoms out near -100 dBm
        return np.round(tl).astype(np.float32)                      # recordings are in whole dB


# ------------------------------------------------------------------------------------------------ real feeds
def check_links(df: pd.DataFrame) -> list[str]:
    """What is wrong with a links table, in plain words. Empty list = fine."""
    bad = [f"links: column '{c}' is missing" for c in LINK_COLS if c not in df.columns]
    if bad:
        return bad
    if df.empty:
        return ["links: the table is empty"]
    if df.link_id.duplicated().any():
        bad.append(f"links: link_id is not unique (for example '{df.link_id[df.link_id.duplicated()].iloc[0]}')")
    for c in ("lat_a", "lat_b"):
        if not pd.to_numeric(df[c], errors="coerce").between(-90, 90).all():
            bad.append(f"links: '{c}' must be a latitude in degrees (-90 to 90)")
    for c in ("lon_a", "lon_b"):
        if not pd.to_numeric(df[c], errors="coerce").between(-180, 180).all():
            bad.append(f"links: '{c}' must be a longitude in degrees (-180 to 180)")
    f = pd.to_numeric(df.freq_ghz, errors="coerce")
    if not f.between(1, 100).all():
        hint = " (it looks like MHz or Hz: divide to get GHz)" if (f > 100).any() else ""
        bad.append(f"links: 'freq_ghz' must be in GHz, between 1 and 100{hint}")
    if not df.pol.astype(str).str.upper().str[0].isin(["H", "V"]).all():
        bad.append("links: 'pol' must be H or V")
    if not bad:
        km = _length_km(df)
        if (km < 0.05).any() or (km > 60).any():
            bad.append(f"links: {int(((km < 0.05) | (km > 60)).sum())} links are shorter than 50 m or longer than 60 km; "
                       "check that the two ends are not swapped or identical")
    return bad


def check_readings(df: pd.DataFrame, links: pd.DataFrame) -> list[str]:
    bad = [f"readings: column '{c}' is missing" for c in ("time", "link_id") if c not in df.columns]
    has_levels, has_loss = {"rsl_dbm", "tsl_dbm"} <= set(df.columns), "loss_db" in df.columns
    if not (has_levels or has_loss):
        bad.append("readings: give either rsl_dbm and tsl_dbm (received and sent level), or loss_db (sent minus received)")
    if bad:
        return bad
    if df.empty:
        return ["readings: the table is empty"]
    t = pd.to_datetime(df.time, utc=True, errors="coerce")
    if t.isna().any():
        bad.append("readings: 'time' could not be read as a date and time for some rows; use UTC like 2026-11-12T10:15:00Z")
    elif t.max() > pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=20):
        bad.append("readings: some times are in the future; times must be UTC, not local time")
    unknown = set(df.link_id.astype(str)) - set(links.link_id.astype(str))
    if unknown:
        bad.append(f"readings: {len(unknown)} link_id values are not in the links table (for example '{sorted(unknown)[0]}')")
    loss = _loss(df)
    if loss.notna().any() and not loss.dropna().between(10, 150).mean() > 0.95:
        bad.append("readings: sent level minus received level should be a path loss of roughly 30 to 110 dB; "
                   "check the units (dBm) and that the two columns are not swapped")
    if not t.isna().any():
        step = t.groupby(df.link_id.astype(str)).apply(lambda s: s.sort_values().diff().dt.total_seconds().median()).median()
        if step > 900:
            bad.append(f"readings: one reading every {step / 60:.0f} minutes is too slow; at most 15 minutes between readings")
    return bad


def _loss(df: pd.DataFrame) -> pd.Series:
    if "loss_db" in df.columns:
        return pd.to_numeric(df.loss_db, errors="coerce")
    return pd.to_numeric(df.tsl_dbm, errors="coerce") - pd.to_numeric(df.rsl_dbm, errors="coerce")


def _length_km(df: pd.DataFrame) -> np.ndarray:
    la, lb = np.radians(df.lat_a.astype(float)), np.radians(df.lat_b.astype(float))
    dlo = np.radians(df.lon_b.astype(float) - df.lon_a.astype(float))
    h = np.sin((lb - la) / 2) ** 2 + np.cos(la) * np.cos(lb) * np.sin(dlo / 2) ** 2
    return (2 * 6371.0088 * np.arcsin(np.sqrt(h))).values


def _read_table(where: str, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """A CSV or JSON table from an http(s) address or a file. `{since}` in the address becomes the UTC time asked for."""
    if since is not None:
        where = where.replace("{since}", since.strftime("%Y-%m-%dT%H:%M:%SZ"))
    if re.match(r"^https?://", where):
        r = requests.get(where, timeout=60)
        r.raise_for_status()
        text = r.text
    elif "://" in where:
        raise ValueError("only http(s) addresses or file paths are accepted")
    else:
        text = Path(where).read_text()
    return pd.read_json(io.StringIO(text)) if text.lstrip()[:1] in "[{" else pd.read_csv(io.StringIO(text))


class TableAdapter:
    """The no-code route: the data owner already serves the standard format."""

    def __init__(self, links_at: str, readings_at: str):
        self.links_at, self.readings_at = links_at, readings_at

    def links(self) -> pd.DataFrame:
        return _read_table(self.links_at)

    def fetch(self, since) -> pd.DataFrame:
        return _read_table(self.readings_at, pd.Timestamp(since))


def installed_adapters() -> list[str]:
    return sorted(p.stem for p in ADAPTERS.glob("*.py") if p.stem not in ("template", "__init__"))


def load_adapter(name: str):
    """An adapter is a Python file in adapters/ with links() and fetch(since). It is put there by whoever runs the
    server; the web page can only choose among installed ones."""
    if not re.fullmatch(r"[a-z0-9_]+", name) or name not in installed_adapters():
        raise ValueError(f"no adapter named '{name}' in adapters/")
    spec = importlib.util.spec_from_file_location(f"adapters.{name}", ADAPTERS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("links", "fetch"):
        if not callable(getattr(mod, fn, None)):
            raise ValueError(f"adapters/{name}.py must define {fn}()")
    return mod


class FeedSource:
    kind = "feed"

    def __init__(self, adapter):
        self.adapter = adapter
        links = adapter.links()
        problems = check_links(links)
        if problems:
            raise ValueError("; ".join(problems))
        links = links.assign(link_id=links.link_id.astype(str))
        self.links = links
        self.sub_links = pd.DataFrame({
            "cml_id": links.link_id, "link_id": links.link_id, "length_km": _length_km(links),
            "freq_ghz": links.freq_ghz.astype(float), "pol": links.pol.astype(str).str.upper().str[0],
            "site_0_lat": links.lat_a.astype(float), "site_0_lon": links.lon_a.astype(float),
            "site_1_lat": links.lat_b.astype(float), "site_1_lon": links.lon_b.astype(float)}).reset_index(drop=True)
        self._row = {k: i for i, k in enumerate(self.sub_links.cml_id)}
        self.last_problem = ""

    def loss(self, minutes: np.ndarray, coarse=None) -> np.ndarray:
        minutes = np.asarray(minutes, np.int64)
        out = np.full((len(self.sub_links), minutes.size), np.nan, np.float32)
        since = pd.Timestamp(int(minutes[0]) * 60, unit="s", tz="UTC") - pd.Timedelta(minutes=15)
        try:
            df = self.adapter.fetch(since)
            problems = check_readings(df, self.links) if len(df) else []
        except Exception as e:                                       # a feed that is down must not stop the watch
            self.last_problem = f"could not read the feed: {e}"
            return out
        self.last_problem = "; ".join(problems)
        if problems or not len(df):
            return out
        m = ((pd.to_datetime(df.time, utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)).values
        loss = _loss(df).values
        rows = df.link_id.astype(str).map(self._row).values
        grid = pd.DataFrame({"r": rows, "m": m, "v": loss}).dropna().groupby(["r", "m"]).v.mean()
        wide = grid.unstack("m").reindex(index=range(len(self.sub_links)))
        wide = wide.reindex(columns=np.arange(minutes[0] - 15, minutes[-1] + 1)).ffill(axis=1, limit=15)   # up to 15-min steps
        return wide[minutes].values.astype(np.float32)
