"""OpenRainER (Emilia-Romagna, Italy, 2021–2022) loaders, OpenSense format.

CML: 1-min RSL/TSL, ~150 links x 2 sub-links. The README says values were rounded
to 1 dB and then averaged, and recommends rounding to the nearest dB again; we do.
References: AWS gauges (15-min rain depth) and gauge-adjusted radar RADadj
(15-min depth, ~1 km grid). Both are converted to mm/h.

Time labels of the 15-min products are not documented; scripts/phase2_crossdomain.py
checks the alignment empirically (lag with the best correlation) and records it.
"""

from __future__ import annotations

import gzip
import shutil
import tarfile
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "openrainer"
INTERIM = ROOT / "data" / "interim" / "openrainer"


def _gunzip(gz: Path, out: Path) -> Path:
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with gzip.open(gz) as f, open(tmp, "wb") as g:
            shutil.copyfileobj(f, g)
        tmp.rename(out)
    return out


def cml_month(ym: str) -> xr.Dataset:
    """One month of CML data, sub-links flattened to cml_id '<link>_<channel>'."""
    gz = next((RAW / "cml").glob(f"CML_{ym}01*.nc.gz"))
    nc = _gunzip(gz, INTERIM / gz.name[:-3])
    ds = xr.open_dataset(nc).load()
    ds = ds.stack(sub=("cml_id", "sublink_id"))
    ids = [f"{c}_{s[-1]}" for c, s in ds.sub.values]
    out = xr.Dataset(
        {"rsl": (("cml_id", "time"), ds.rsl.transpose("sub", "time").values.round()),
         "tsl": (("cml_id", "time"), ds.tsl.transpose("sub", "time").values.round())},
        coords={"cml_id": ids, "time": ds.time.values,
                "length": ("cml_id", ds.length.values / 1000.0),
                "frequency": ("cml_id", ds.frequency.values / 1000.0),
                "polarization": ("cml_id", np.asarray(ds.polarization.values).astype(str)),
                "site_0_lat": ("cml_id", ds.site_0_lat.values), "site_0_lon": ("cml_id", ds.site_0_lon.values),
                "site_1_lat": ("cml_id", ds.site_1_lat.values), "site_1_lon": ("cml_id", ds.site_1_lon.values)})
    return out


def cml(months: list[str]) -> xr.Dataset:
    """Several months on a common 1-min grid and common set of sub-links."""
    parts = [cml_month(m) for m in months]
    common = sorted(set.intersection(*(set(p.cml_id.values) for p in parts)))
    parts = [p.sel(cml_id=common) for p in parts]
    # metadata must not change between months for a sub-link we keep
    keep = np.ones(len(common), bool)
    for p in parts[1:]:
        for v in ("length", "frequency", "site_0_lat", "site_1_lat"):
            keep &= np.isclose(p[v].values, parts[0][v].values)
    parts = [p.isel(cml_id=np.where(keep)[0]) for p in parts]
    coords = {k: parts[0][k] for k in parts[0].coords if k != "time"}
    ds = xr.concat([p.drop_vars(list(coords)) for p in parts], dim="time").assign_coords(coords)
    full = np.arange(ds.time.values[0], ds.time.values[-1] + np.timedelta64(1, "m"),
                     np.timedelta64(1, "m"))
    ds = ds.reindex(time=full)
    pol = np.char.lower(ds.polarization.values.astype(str))
    return ds.assign_coords(polarization=("cml_id", pol))


def aws_month(ym: str) -> xr.Dataset:
    p = RAW / "aws" / f"AWS_{ym}.nc"
    if not p.exists():
        _gunzip(RAW / "aws" / f"AWS_{ym}.nc.gz", INTERIM / f"AWS_{ym}.nc")
        p = INTERIM / f"AWS_{ym}.nc"
    return xr.open_dataset(p)[["rainfall_amount"]].load()


def gauges(months: list[str]) -> xr.Dataset:
    """AWS rain as mm/h on the 15-min grid, as in the files (label convention unchanged)."""
    parts = [aws_month(m) for m in months]
    common = sorted(set.intersection(*(set(p.id.values) for p in parts)))
    g = xr.concat([p.sel(id=common) for p in parts], dim="time")
    return xr.Dataset({"rainfall_rate": g.rainfall_amount * 4.0},
                      coords={"lon": ("id", g.longitude.values), "lat": ("id", g.latitude.values)})


def radadj_along(ds_cml: xr.Dataset, months: list[str]) -> xr.DataArray:
    """Gauge-adjusted radar rain [mm/h] averaged along each CML path, 15-min."""
    import poligrain as plg

    out = []
    for ym in months:
        cache = INTERIM / f"radadj_along_{ym}.nc"
        if not cache.exists():
            INTERIM.mkdir(parents=True, exist_ok=True)
            gz = INTERIM / f"RADadj_{ym}.nc.gz"
            if not gz.exists():
                with tarfile.open(RAW / "RADadj.tar") as tf:
                    m = tf.getmember(f"RADadj_{ym}.nc.gz")
                    m.name = gz.name
                    tf.extract(m, INTERIM)
            nc = _gunzip(gz, INTERIM / f"RADadj_{ym}.nc")
            rad = xr.open_dataset(nc)
            xg, yg = np.meshgrid(rad.lon.values, rad.lat.values)
            w = plg.spatial.calc_sparse_intersect_weights_for_several_cmls(
                x1_line=ds_cml.site_0_lon.values, y1_line=ds_cml.site_0_lat.values,
                x2_line=ds_cml.site_1_lon.values, y2_line=ds_cml.site_1_lat.values,
                cml_id=ds_cml.cml_id.values, x_grid=xg, y_grid=yg, grid_point_location="center")
            along = plg.spatial.get_grid_time_series_at_intersections(
                rad.rainfall_amount.transpose("time", "lat", "lon"), w)
            along = (along * 4.0).transpose("cml_id", "time").rename("radar_rainfall_rate")
            along.to_netcdf(cache)
            nc.unlink()
            gz.unlink()
        out.append(xr.open_dataarray(cache).sel(cml_id=ds_cml.cml_id.values).load())
    r = xr.concat(out, dim="time")
    return r.isel(time=~r.get_index("time").duplicated())
