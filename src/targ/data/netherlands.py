"""Netherlands CML (KPN-style operator data, 2014) loader, plus KNMI radar reference.

CML: 15-min minimum and maximum received level per sub-link (no transmit level:
transmit power is constant). Processing follows the dataset's own
RawCML_to_RAINLINKFormat.R: arc-milliseconds → degrees, kHz → GHz, keep only
rows with a 15-min period and ":00" seconds, local time (CEST) → UTC, start →
end time (+15 min), sub-link ID from the rounded end-point coordinates.
Polarisation is not in the data; we assume vertical (stated in the results).

Reference: KNMI gauge-adjusted radar, hourly accumulation (RAD_NL25_RAC_MFBS_01H,
1 km grid), averaged along each link path.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "netherlands_cml" / "RawCMLdata" / "RawCMLdata" / "2014"
KNMI_ZIP = ROOT / "data" / "raw" / "knmi_radar" / \
    "RADNL_CLIM____MFBSNL25_01H_20140101T000000_20150101T000000_0002.zip"
INTERIM = ROOT / "data" / "interim"
CML_CACHE = INTERIM / "nl_2014_15min.nc"
COLS = ["LINK_ID", "SITE_ID", "YStart", "XStart", "FAR_END_SITE_ID", "YEnd", "XEnd", "Frequency",
        "DateTime", "NAME", "PERIOD_DURATION", "ES", "SES", "Pmin", "Pmax"]


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, names=COLS, dtype={"DateTime": str}).dropna()
    vendor = "NEC" if path.name.startswith("NEC") else "NOKIA"
    ok_dur = df.PERIOD_DURATION == (900 if vendor == "NEC" else 15)
    df = df[ok_dur & df.DateTime.str.endswith("00")].copy()
    for c in ("XStart", "YStart", "XEnd", "YEnd"):
        df[c] = (df[c] / 3.6e6).round(6)
    df["Frequency"] = pd.to_numeric(df.Frequency, errors="coerce") / 1e6
    df["Pmin"] = pd.to_numeric(df.Pmin, errors="coerce")
    df["Pmax"] = pd.to_numeric(df.Pmax, errors="coerce")
    local = pd.to_datetime(df.DateTime, format="%Y%m%d%H%M%S")
    df["time"] = (local.dt.tz_localize("Europe/Amsterdam", ambiguous="NaT", nonexistent="NaT")
                  .dt.tz_convert("UTC").dt.tz_localize(None) + pd.Timedelta("15min"))
    df["vendor"] = vendor
    return df.dropna(subset=["time"])


def load_cml() -> xr.Dataset:
    """pmin, pmax [dBm] per sub-link × 15-min interval (labelled at the interval end, UTC)."""
    if CML_CACHE.exists():
        return xr.open_dataset(CML_CACHE)
    import pyproj

    df = pd.concat([_read(p) for p in sorted(RAW.glob("*.csv.gz"))], ignore_index=True)
    df["cml_id"] = (df.XStart.map("{:.6f}".format) + "_" + df.YStart.map("{:.6f}".format) + "_" +
                    df.XEnd.map("{:.6f}".format) + "_" + df.YEnd.map("{:.6f}".format))
    df = df.drop_duplicates(subset=["cml_id", "time"])
    meta = df.groupby("cml_id").agg(XStart=("XStart", "first"), YStart=("YStart", "first"),
                                    XEnd=("XEnd", "first"), YEnd=("YEnd", "first"),
                                    frequency=("Frequency", "median"), vendor=("vendor", "first"))
    geod = pyproj.Geod(ellps="WGS84")
    _, _, dist = geod.inv(meta.XStart.values, meta.YStart.values, meta.XEnd.values, meta.YEnd.values)
    meta["length"] = dist / 1000.0
    times = pd.date_range(df.time.min(), df.time.max(), freq="15min")
    ii = meta.index.get_indexer(df.cml_id)
    tt = times.get_indexer(df.time)
    arr = {k: np.full((len(meta), len(times)), np.nan, np.float32) for k in ("pmin", "pmax")}
    arr["pmin"][ii, tt] = df.Pmin.values
    arr["pmax"][ii, tt] = df.Pmax.values
    ds = xr.Dataset(
        {k: (("cml_id", "time"), v) for k, v in arr.items()},
        coords={"cml_id": meta.index.values, "time": times.values,
                "site_0_lon": ("cml_id", meta.XStart.values), "site_0_lat": ("cml_id", meta.YStart.values),
                "site_1_lon": ("cml_id", meta.XEnd.values), "site_1_lat": ("cml_id", meta.YEnd.values),
                "frequency": ("cml_id", meta.frequency.values.astype(float)),
                "length": ("cml_id", meta.length.values.astype(float)),
                "vendor": ("cml_id", meta.vendor.to_numpy(dtype=str)),
                "polarization": ("cml_id", np.full(len(meta), "vertical"))})
    ds = ds.sel(cml_id=ds.length > 0.05)   # drop zero-length / co-located entries
    INTERIM.mkdir(parents=True, exist_ok=True)
    tmp = CML_CACHE.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp)
    tmp.rename(CML_CACHE)
    return ds


KNMI_PROJ = "+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=0"


def radar_along(ds_cml: xr.Dataset, start: str, end: str) -> xr.DataArray:
    """KNMI hourly gauge-adjusted radar [mm/h] along each link, labelled at the hour END (UTC)."""
    cache = INTERIM / f"nl_knmi_along_{start}_{end}.nc"
    if cache.exists():
        return xr.open_dataarray(cache)
    import h5py
    import poligrain as plg
    import pyproj

    # 1-km grid, pixel centres: x = 0.5 .. 699.5 km, y = -3650.5 .. -4414.5 km
    xg, yg = np.meshgrid(np.arange(700) + 0.5, -3650.0 - (np.arange(765) + 0.5))
    proj = pyproj.Proj(KNMI_PROJ)   # km; plain projection, no datum shift (as KNMI tools do)
    x0, y0 = proj(ds_cml.site_0_lon.values, ds_cml.site_0_lat.values)
    x1, y1 = proj(ds_cml.site_1_lon.values, ds_cml.site_1_lat.values)
    w = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=x0, y1_line=y0, x2_line=x1, y2_line=y1, cml_id=ds_cml.cml_id.values,
        x_grid=xg, y_grid=yg, grid_point_location="center")
    hours = pd.date_range(start, end, freq="1h")
    frames, ok_times = [], []
    with zipfile.ZipFile(KNMI_ZIP) as z:
        names = set(z.namelist())
        for t in hours:
            n = f"RAD_NL25_RAC_MFBS_01H/{t:%Y}/{t:%m}/RAD_NL25_RAC_MFBS_01H_{t:%Y%m%d%H%M}_NL.h5"
            if n not in names:
                continue
            with z.open(n) as f, h5py.File(f) as h:
                pv = h["image1/image_data"][()].astype(np.float32)
            pv[pv == 65535] = np.nan
            frames.append(pv * 0.01)
            ok_times.append(t)
    R = xr.DataArray(np.stack(frames), dims=("time", "y", "x"),
                     coords={"time": np.array(ok_times, dtype="datetime64[ns]")})
    along = plg.spatial.get_grid_time_series_at_intersections(R, w)
    along = along.transpose("cml_id", "time").rename("radar_rainfall_rate")
    along.to_netcdf(cache)
    return along


def check_grid_corners() -> dict:
    """Sanity check of the grid definition against the corners stored in the file."""
    import h5py
    import pyproj

    with zipfile.ZipFile(KNMI_ZIP) as z:
        n = next(n for n in z.namelist() if n.endswith(".h5"))
        with z.open(n) as f, h5py.File(f) as h:
            corners = h["geographic"].attrs["geo_product_corners"]
    lon, lat = corners[0::2], corners[1::2]
    x, y = pyproj.Proj(KNMI_PROJ)(lon, lat)
    return {"corners_lonlat": list(zip(lon.tolist(), lat.tolist())),
            "corners_xy_km": list(zip(np.round(x, 2).tolist(), np.round(y, 2).tolist()))}
