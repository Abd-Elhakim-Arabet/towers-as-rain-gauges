"""Template adapter: copy this file to adapters/<your_name>.py and fill in the two functions.

An adapter translates however YOUR system stores microwave-link data into the standard format of the live watch.
Once the file is in adapters/, it appears in the "My own data" list on the Live watch page. Check it first:

    python -m watch.check_adapter adapters/<your_name>.py

Standard format (pandas DataFrames):

  links()        one row per link, read once
      link_id    your name for the link (text)
      lat_a, lon_a, lat_b, lon_b   the two ends, degrees
      freq_ghz   frequency in GHz
      pol        polarisation, "H" or "V"

  fetch(since)   all readings taken after `since` (a UTC pandas Timestamp), called every few minutes
      time       UTC time of the reading
      link_id    as in links()
      rsl_dbm    received signal level, dBm
      tsl_dbm    transmitted signal level, dBm
                 (or one column loss_db = transmitted minus received, instead of the two levels)

One reading per link at least every 15 minutes; every minute is best. If the transmitted level is constant on your
equipment, put that constant in tsl_dbm.
"""

import pandas as pd


def links() -> pd.DataFrame:
    # Example: your inventory is a CSV with other column names and the frequency in MHz.
    # inv = pd.read_csv("/path/to/inventory.csv")
    # return pd.DataFrame({"link_id": inv["LinkName"], "lat_a": inv["SiteA_Lat"], "lon_a": inv["SiteA_Lon"],
    #                      "lat_b": inv["SiteB_Lat"], "lon_b": inv["SiteB_Lon"],
    #                      "freq_ghz": inv["Freq_MHz"] / 1000, "pol": inv["Polarisation"].str[0]})
    raise NotImplementedError("fill in links()")


def fetch(since: pd.Timestamp) -> pd.DataFrame:
    # Example: your network management system exports one CSV per 15 minutes, times in local time (UTC+1).
    # df = pd.read_csv(f"/exports/rsl_{since:%Y%m%d}.csv")
    # t = pd.to_datetime(df["Timestamp"]).dt.tz_localize("Africa/Algiers").dt.tz_convert("UTC")
    # out = pd.DataFrame({"time": t, "link_id": df["LinkName"], "rsl_dbm": df["RSL"], "tsl_dbm": df["TSL"]})
    # return out[out.time > since]
    raise NotImplementedError("fill in fetch(since)")
