"""OpenMRG (Gothenburg, summer 2015) loaders in OpenSense format.

Raw CML data is 10-s RSL/TSL for 728 sub-links. We average it to 1 min and
cache it, because every later step (classical baseline, AI models) works at
1-min resolution.

Conventions (OpenSense naming, as used by pycomlink/poligrain):
  dims  cml_id (one per sub-link), time
  coords site_0_lat/lon, site_1_lat/lon, length [km], frequency [GHz], polarization
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "openmrg"
INTERIM = ROOT / "data" / "interim"
CML_1MIN = INTERIM / "openmrg_cml_1min.nc"


def metadata() -> pd.DataFrame:
    md = pd.read_csv(RAW / "cml" / "cml_metadata.csv")
    return md.set_index("Sublink")


def _build_cml_1min() -> xr.Dataset:
    raw = xr.open_dataset(RAW / "cml" / "cml.nc")
    md = metadata().loc[raw.sublink.values]
    out = []
    # Resample in blocks of sub-links: the raw file is ~4.6 GB.
    for i in range(0, raw.sizes["sublink"], 64):
        block = raw.isel(sublink=slice(i, i + 64)).load()
        out.append(block.resample(time="1min", label="left").mean())
    ds = xr.concat(out, dim="sublink").rename(sublink="cml_id")
    ds = ds.assign_coords(
        link_id=("cml_id", md.Link.values),
        direction=("cml_id", md.Direction.to_numpy(dtype=str)),
        site_0_lat=("cml_id", md.NearLatitude_DecDeg.values),
        site_0_lon=("cml_id", md.NearLongitude_DecDeg.values),
        site_1_lat=("cml_id", md.FarLatitude_DecDeg.values),
        site_1_lon=("cml_id", md.FarLongitude_DecDeg.values),
        length=("cml_id", md.Length_km.values),
        frequency=("cml_id", md.Frequency_GHz.values),
        polarization=("cml_id", md.Polarization.str.lower().to_numpy(dtype=str)),
    )
    ds["rsl"].attrs.update(units="dBm", long_name="received signal level, 1-min mean")
    ds["tsl"].attrs.update(units="dBm", long_name="transmitted signal level, 1-min mean")
    ds.attrs.update(raw.attrs, processing="10-s raw averaged to 1-min (left-labelled)")
    return ds


def load_cml_1min() -> xr.Dataset:
    if not CML_1MIN.exists():
        INTERIM.mkdir(parents=True, exist_ok=True)
        tmp = CML_1MIN.with_suffix(".tmp.nc")  # never leave a half-written cache
        _build_cml_1min().to_netcdf(tmp)
        tmp.rename(CML_1MIN)
    return xr.open_dataset(CML_1MIN)


def load_radar_rainrate() -> xr.DataArray:
    """SMHI radar as rain rate [mm/h] (5-min instantaneous scans, not gauge-adjusted).

    Uses the Z = a R^b parameters stored in the file itself (a=200, b=1.5).
    """
    rad = xr.open_dataset(RAW / "radar" / "radar.nc")
    dbz = rad["data"]
    a, b = float(dbz.attrs["zr_a"]), float(dbz.attrs["zr_b"])
    dbz = dbz.where(dbz > 0)  # weak echoes / clutter floor → 0 rain below
    R = ((10 ** (dbz / 10)) / a) ** (1 / b)
    R = R.fillna(0.0).where(rad["data"].notnull())
    return R.assign_coords(lat=rad.lat, lon=rad.lon).rename("rainfall_rate")


def load_city_gauges() -> xr.Dataset:
    """10 municipal gauges (7 weighing, 3 tipping-bucket) as rain rate [mm/h] at 1 min.

    Raw values are 1-min accumulations [mm] labelled at the END of the minute; we
    relabel to the start so they align with the left-labelled CML minutes.
    """
    df = pd.read_csv(RAW / "gauges" / "city" / "CityGauges-2015JJA.csv",
                     parse_dates=["Time_UTC"], index_col="Time_UTC")
    df.index = df.index.tz_convert(None)
    df.index = df.index - pd.Timedelta("1min")
    meta = pd.read_csv(RAW / "gauges" / "city" / "CityGauges-metadata.csv",
                       encoding="utf-8-sig").set_index("Name").loc[df.columns]
    return xr.Dataset(
        {"rainfall_rate": (("id", "time"), df.values.T * 60.0)},
        coords={"id": df.columns.to_numpy(dtype=str), "time": df.index.values,
                "lon": ("id", meta.Longitude_DecDeg.values),
                "lat": ("id", meta.Latitude_DecDeg.values),
                "gauge_type": ("id", meta.Type.to_numpy(dtype=str))},
    )


RADAR_ALONG_CML = INTERIM / "openmrg_radar_along_cml_5min.nc"


def radar_along_cmls(ds_cml: xr.Dataset) -> xr.DataArray:
    """Radar rain rate [mm/h] averaged along each CML path (5-min, cml_id × time).

    CML endpoints are projected into the radar's polar-stereographic grid and
    poligrain computes the length-weighted pixel intersections.
    """
    if RADAR_ALONG_CML.exists():
        return xr.open_dataarray(RADAR_ALONG_CML)
    import poligrain as plg
    import pyproj

    rad_ds = xr.open_dataset(RAW / "radar" / "radar.nc")
    to_radar = pyproj.Transformer.from_crs("EPSG:4326", rad_ds.attrs["proj_string"],
                                           always_xy=True)
    x0, y0 = to_radar.transform(ds_cml.site_0_lon.values, ds_cml.site_0_lat.values)
    x1, y1 = to_radar.transform(ds_cml.site_1_lon.values, ds_cml.site_1_lat.values)
    xg, yg = np.meshgrid(rad_ds.x.values, rad_ds.y.values)
    weights = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
        x1_line=x0, y1_line=y0, x2_line=x1, y2_line=y1,
        cml_id=ds_cml.cml_id.values, x_grid=xg, y_grid=yg, grid_point_location="center")
    R = load_radar_rainrate().transpose("time", "y", "x")
    along = plg.spatial.get_grid_time_series_at_intersections(R, weights)
    along = along.transpose("cml_id", "time").rename("radar_rainfall_rate")
    tmp = RADAR_ALONG_CML.with_suffix(".tmp.nc")
    along.to_netcdf(tmp)
    tmp.rename(RADAR_ALONG_CML)
    return along


def project_utm(lon, lat):
    """WGS84 → UTM 32N metres (Gothenburg is near the 32/33 boundary; 32N is fine for distances)."""
    import pyproj

    t = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    return t.transform(lon, lat)


def gauge_link_pairs(ds_cml: xr.Dataset, gauges: xr.Dataset, max_distance_m: float) -> pd.DataFrame:
    """CML–gauge pairs where the gauge lies within `max_distance_m` of the link path."""
    import poligrain as plg

    x0, y0 = project_utm(ds_cml.site_0_lon.values, ds_cml.site_0_lat.values)
    x1, y1 = project_utm(ds_cml.site_1_lon.values, ds_cml.site_1_lat.values)
    gx, gy = project_utm(gauges.lon.values, gauges.lat.values)
    cml = ds_cml[[]].assign_coords(site_0_x=("cml_id", x0), site_0_y=("cml_id", y0),
                                   site_1_x=("cml_id", x1), site_1_y=("cml_id", y1),
                                   length=ds_cml.length * 1000)
    g = gauges[[]].assign_coords(x=("id", gx), y=("id", gy))
    near = plg.spatial.get_closest_points_to_line(cml, g, max_distance=max_distance_m, n_closest=1)
    dist = near.distance.values[:, 0]
    ok = np.isfinite(dist)
    # poligrain returns the neighbour as a positional index into ds_gauges.id
    idx = near.neighbor_id.values[ok, 0].astype(int)
    return pd.DataFrame({"cml_id": near.cml_id.values[ok],
                         "gauge": gauges.id.values[idx],
                         "distance_m": dist[ok]})
