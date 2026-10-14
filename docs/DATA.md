# Data

All data here is open. Raw files live in `data/raw/<dataset>/` and are git-ignored.
`data/raw/MANIFEST.json` records size, MD5, version, license and download date for every file.
The registry the download script reads is `configs/datasets.yaml`.

Status legend: ✅ downloaded and verified · ⏳ in progress · 🔑 needs a free account/key · 📅 later phase

## Real CML data (used for REAL validation)

| Dataset | Version | License | What we use | Size | Status |
|---|---|---|---|---|---|
| **OpenMRG**, Gothenburg, SE. DOI [10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689) | 1.1 (2022-09-23) | CC BY-SA 4.0 | full archive | 318 MB zip → 4.8 GB | ✅ MD5 matches Zenodo |
| **OpenRainER**, Emilia-Romagna, IT. Concept DOI [10.5281/zenodo.10593848](https://doi.org/10.5281/zenodo.10593848) | 2.0.3 (record 22829808, 2026-09-18) | CC BY 4.0 | `CML.tar`, `AWS.tar`, `README.txt` | 107 + 22 MB | ✅ MD5 matches Zenodo |
| ↳ gauge-adjusted radar `RADadj.tar` | same | CC BY 4.0 | radar reference for maps | 1.57 GB | 📅 Phase 2c/3 |
| **Netherlands 4-year CML**. DOI [10.4121/be252844-b672-471e-8d69-27269a862ec1](https://doi.org/10.4121/be252844-b672-471e-8d69-27269a862ec1) | 1 (2024-06-24) | CC BY 4.0 | `README.pdf`, R script; **subset** of `RawCMLdata.zip` | 12.6 GB total | ✅ docs (MD5 OK); ⚠️ subset: 1 day extracted, then 4TU blocked us |

### What is in each

**OpenMRG** (checked by loading it, 2026-10-14)
- `cml/cml.nc`: `rsl`, `tsl` [dBm], dims `time` × `sublink` = 794,887 × 728 (364 links × 2 directions), 10-s sampling, 2015-06-01 → 2015-09-01 UTC.
- `cml/cml_metadata.csv`: 7.5–38.7 GHz (mean 31.2), 0.14–15.2 km (mean 2.9 km), polarization, antenna diameters, coordinates.
- `gauges/city/`: 10 municipal tipping-bucket gauges, 1-min. `gauges/smhi/`: SMHI station Göteborg A, 15-min.
- `radar/radar.nc`: SMHI reflectivity composite, 5-min, 48 × 37 grid (dBZ = 0.4·v − 30).
- Reference paper: Andersson et al. (2022), *Earth Syst. Sci. Data*. (DOI to be added after we check it.)

**OpenRainER** (checked by loading July 2021, 2026-10-14)
- `cml/CML_YYYYMM*.nc.gz`, 24 monthly files: `rsl`, `tsl` [dBm], dims `cml_id` (151) × `sublink_id` (2) × `time` (1-min), with frequency, polarization, length and site elevation. OpenSense naming convention. The files say `version: 1.1` inside Zenodo record v2.0.3.
- `aws/AWS_YYYYMM.nc.gz`: 319 stations, 15-min `rainfall_amount`, plus temperature, humidity and wind.
- RADadj: 15-min gauge-adjusted radar accumulations, ~1 km grid.

**Netherlands** (from README.pdf)
- 15-min **min/max RSL only, no TSL**. The operator says TSL was constant (±0.2 dB, no ATPC), so attenuation = baseline − RSL.
- ~3,070 sub-links over ~1,818 paths. T-Mobile NL (now Odido). 2011-01-13 → 2015-03-15, with gaps.
- Daily gzipped CSV per vendor: Nokia (1 dB resolution) and NEC (0.1 dB; empty after 2014-08-12). Local time (CET/CEST) in raw files.
- Raw archive summary (from the zip's directory): 2011 ≈ 3.9 GB, 2012 ≈ 4.0 GB, 2013 ≈ 3.2 GB, 2014 ≈ 1.4 GB, 2015 ≈ 0.1 GB.
- Sample check: `RawCMLdata/2014/NEC_7-28-2014.csv.gz` (1.7 MB, 59,982 rows) extracted remotely and is valid. Columns are as the README describes.
- **Plan:** we do not download 12.6 GB. `scripts/download_nl_subset.py` reads the zip's directory over HTTP range requests and extracts selected days (summer 2014, including the 28 July 2014 extreme-rain day). This is the format an operator would actually export, so it tests the "15-min NMS export" story.
- Radar reference: KNMI gauge-adjusted radar from the KNMI Data Platform (🔑 free API key). Dataset name and access still to be confirmed in Phase 2c.

## Algeria scenario (SIMULATION ONLY, not validation)

| Dataset | License | Use | Status |
|---|---|---|---|
| OpenCelliD, MCC 603 (Mobilis 01, Djezzy 02, Ooredoo 03) | CC BY-SA 4.0 | tower sites (crowdsourced, approximate) | 🔑 `OPENCELLID_TOKEN` |
| GPM IMERG V07, half-hourly, 0.1° | NASA open data | real storm fields for past Algerian events | 🔑 `EARTHDATA_TOKEN` |
| CHIRPS daily, ERA5 hourly | CC BY 4.0 / Copernicus licence | auxiliary checks | 📅 (ERA5 needs a CDS key) |
| Copernicus DEM GLO-30 | Copernicus DEM licence | terrain, flow direction/accumulation | 📅 public S3, no key |
| HydroSHEDS / MERIT Hydro | HydroSHEDS licence / CC BY-NC 4.0 (MERIT) | drainage network | 📅 |
| ESA WorldCover 10 m v200 | CC BY 4.0 | impervious surface | 📅 public S3, no key |
| WorldPop 100 m | CC BY 4.0 | exposed population | 📅 |
| OpenStreetMap (Geofabrik) | ODbL 1.0 | buildings, roads, schools, hospitals | 📅 |
| Global Flood Database (Tellman et al. 2021) | CC BY-NC 4.0 (to confirm) | past flood extents for labels | 📅 via Google Earth Engine (account needed) |

Licenses marked "to confirm" are checked when the data is downloaded.

## Network notes
- Zenodo serves this lab at about 70–200 KB/s and **does not support HTTP range requests**. Downloads resume with `curl -C -`.
- 4TU supports ranges but **refuses connections** after a burst of requests (seen 2026-10-14 after the first remote listing; afterwards the whole site, data.4tu.nl, was unreachable from the lab IP). The subset script now does fewer, larger range reads (1 MB) and backs off exponentially. Retry later, or run it from another network.
