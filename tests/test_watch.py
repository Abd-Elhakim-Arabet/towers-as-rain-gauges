"""Phase 7 (live watch): format checks, alert levels, email outbox, the adapter route, a streaming simulation."""

import numpy as np
import pandas as pd
import pytest

from watch import alerts, geocode, sources, weather


def links_ok():
    return pd.DataFrame({"link_id": ["a", "b"], "lat_a": [36.70, 36.72], "lon_a": [3.00, 3.02],
                         "lat_b": [36.72, 36.74], "lon_b": [3.03, 3.05], "freq_ghz": [23.0, 38.0], "pol": ["V", "H"]})


def readings_ok(n=30):
    t = pd.Timestamp.now(tz="UTC").floor("min") - pd.to_timedelta(np.arange(n)[::-1], unit="min")
    return pd.DataFrame({"time": np.tile(t, 2), "link_id": np.repeat(["a", "b"], n),
                         "rsl_dbm": -45.0, "tsl_dbm": 15.0})


def test_good_feed_passes():
    assert sources.check_links(links_ok()) == []
    assert sources.check_readings(readings_ok(), links_ok()) == []


def test_feed_mistakes_are_explained():
    bad = links_ok().assign(freq_ghz=[23000.0, 38000.0])
    assert any("MHz" in p for p in sources.check_links(bad))
    assert any("'pol'" in p for p in sources.check_links(links_ok().assign(pol=["X", "V"])))
    assert any("missing" in p for p in sources.check_links(links_ok().drop(columns="lat_b")))
    swapped = readings_ok().rename(columns={"rsl_dbm": "tsl_dbm", "tsl_dbm": "rsl_dbm"})
    assert any("swapped" in p for p in sources.check_readings(swapped, links_ok()))
    future = readings_ok().assign(time=lambda d: pd.to_datetime(d.time) + pd.Timedelta(hours=2))
    assert any("future" in p for p in sources.check_readings(future, links_ok()))
    slow = readings_ok(8).assign(time=lambda d: pd.to_datetime(d.time).dt.floor("h") - pd.to_timedelta(np.tile(np.arange(8), 2), unit="h"))
    assert any("too slow" in p for p in sources.check_readings(slow, links_ok()))


def test_feed_source_reads_loss(tmp_path):
    (tmp_path / "l.csv").write_text(links_ok().to_csv(index=False))
    (tmp_path / "r.csv").write_text(readings_ok().to_csv(index=False))
    src = sources.FeedSource(sources.TableAdapter(str(tmp_path / "l.csv"), str(tmp_path / "r.csv")))
    now = int(pd.Timestamp.now(tz="UTC").timestamp() // 60)
    tl = src.loss(np.arange(now - 9, now + 1))
    assert tl.shape == (2, 10) and np.nanmax(np.abs(tl - 60.0)) == 0      # 15 - (-45) dB


def test_adapter_names_are_safe():
    with pytest.raises(ValueError):
        sources.load_adapter("../../etc/passwd")
    with pytest.raises(ValueError):
        sources.load_adapter("template")


def test_levels_rise_fast_fall_slowly():
    assert alerts.level(0, 0, 10, 0.1) == "normal"
    assert alerts.level(0, 0, 10, 0.6) == "watch"
    assert alerts.level(2, 12, 10, 0.2) == "warning"
    assert alerts.level(11, 12, 10, 0.2) == "flooding"
    lv, calm = alerts.settle("normal", "flooding", 0)
    assert lv == "flooding"
    for _ in range(alerts.CALM_STEPS - 1):
        lv, calm = alerts.settle(lv, "normal", calm)
        assert lv == "flooding"
    assert alerts.settle(lv, "normal", calm)[0] == "normal"


def test_email_goes_to_outbox_without_account(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "OUTBOX", tmp_path)
    monkeypatch.setattr(alerts, "env", lambda k, d="": {"ALERT_TO": "someone@example.org"}.get(k, d))
    msg = alerts.compose("Testville", "flooding", "warning", {"ha_now": 12.0, "alarm_ha": 5.0}, drill=True)
    assert "outbox" in alerts.send(msg)
    text = next(tmp_path.glob("*.eml")).read_text()
    assert "[DRILL] Flooding: Testville" in text and "THIS IS A DRILL" in text and "simulated" in text


def test_box_is_capped_and_snapped():
    lat0, lon0, lat1, lon1 = geocode.box(36.75, 5.06, 99)
    assert (lat1 - lat0) * 111.2 == pytest.approx(2 * geocode.MAX_HALF_KM, rel=0.01)
    assert abs(lat0 * 3600 - round(lat0 * 3600)) < 1e-6


def test_design_storm_total():
    c = weather.design_storm((36.7, 5.0, 36.8, 5.1), 0.0, total_mm=90, hours=3)
    assert c.mmh[:, 1, 1].sum() / 4 == pytest.approx(90)
    assert c.at(np.array([-1e5]), np.array([36.75]), np.array([5.05])).max() == 0      # before the storm


def test_simulation_streams_like_one_block():
    """Asking for signals in small pieces gives the same dry signal as one block, and rain adds loss."""
    links = links_ok().rename(columns={"lat_a": "site_0_lat", "lon_a": "site_0_lon", "lat_b": "site_1_lat", "lon_b": "site_1_lon"})
    links["length_km"] = [2.9, 2.9]
    box = (36.69, 2.99, 36.75, 3.06)
    dry = weather.Coarse(np.array([0.0, 1e12]), np.array([36.69, 36.75]), np.array([2.99, 3.06]), np.zeros((2, 2, 2), np.float32))
    m = np.arange(27_000_000, 27_000_060)
    a = sources.SimSource(box, links, seed=3).loss(m, dry)
    s = sources.SimSource(box, links, seed=3)
    b = np.concatenate([s.loss(m[:20], dry), s.loss(m[20:], dry)], 1)
    assert np.array_equal(a, b, equal_nan=True)
    wet = weather.Coarse(dry.t, dry.lat, dry.lon, np.full((2, 2, 2), 30.0, np.float32))
    assert np.nanmean(sources.SimSource(box, links, seed=3).loss(m, wet) - a) > 3        # 30 mm/h on ~3 km at 23-38 GHz
