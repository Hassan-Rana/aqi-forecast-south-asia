import math

import numpy as np
import pandas as pd
import pytest

from aqi.aqi_calc import aqi, category, sub_index
from aqi.cities import CITIES
from aqi.events import event_features, load_events, season_features
from aqi.features import build_supervised, feature_columns
from aqi.forecast import forecast_city
from aqi.impact import event_impact
from aqi.model import evaluate, train_all
from aqi.synthetic import make_synthetic


@pytest.fixture(scope="module")
def data():
    return make_synthetic("2022-08-01", "2024-12-31", cities=["delhi", "lahore", "mumbai"])


def test_cpcb_breakpoints():
    assert sub_index(30, "pm2_5", "cpcb") == pytest.approx(50)
    assert sub_index(60, "pm2_5", "cpcb") == pytest.approx(100)
    assert sub_index(90, "pm2_5", "cpcb") == pytest.approx(200)
    assert sub_index(250, "pm2_5", "cpcb") == pytest.approx(400)
    assert sub_index(100, "pm10", "cpcb") == pytest.approx(100)
    assert sub_index(10_000, "pm2_5", "cpcb") == 500


def test_us_breakpoints_and_max_rule():
    assert sub_index(35.4, "pm2_5", "us") == pytest.approx(100)
    assert sub_index(55.4, "pm2_5", "us") == pytest.approx(150)
    # AQI is the larger sub-index: PM10 dominates here
    assert aqi(5, 300, "us") == round(sub_index(300, "pm10", "us"))


def test_categories():
    assert category(45, "cpcb") == "Good"
    assert category(250, "cpcb") == "Poor"
    assert category(450, "cpcb") == "Severe"
    assert category(160, "us") == "Unhealthy"
    assert category(math.nan) == "Unknown"


def test_event_calendar_and_windows():
    ev = load_events()
    assert {"Diwali", "Shab-e-Barat", "Lohri", "New Year's Eve", "Eid al-Adha"} <= set(ev["event"])
    days = pd.date_range("2024-10-29", "2024-11-05")
    f = event_features(CITIES["delhi"], days)
    assert f.loc["2024-10-31", "ev_fireworks"] == 1.0  # Diwali 2024
    assert 0 < f.loc["2024-10-30", "ev_fireworks"] < 1  # build-up the day before
    assert 0 < f.loc["2024-11-01", "ev_fireworks"] < 1  # smoke lingers after
    assert f.loc["2024-11-05", "ev_fireworks"] == 0
    assert "Diwali" in f.loc["2024-10-31", "event_names"]
    # Diwali is an Indian event in this table; Lahore gets Shab-e-Barat instead
    assert event_features(CITIES["lahore"], days)["ev_fireworks"].max() == 0
    assert event_features(CITIES["lahore"], pd.DatetimeIndex(["2024-02-25"]))["ev_fireworks"].iloc[0] > 0


def test_seasons_depend_on_city():
    nov = pd.DatetimeIndex(["2024-11-03"])
    assert season_features(CITIES["delhi"], nov)["season_crop_burning"].iloc[0] > 0.9
    assert season_features(CITIES["mumbai"], nov)["season_crop_burning"].iloc[0] == 0
    jan = pd.DatetimeIndex(["2025-01-05"])
    assert season_features(CITIES["delhi"], jan)["season_winter_inversion"].iloc[0] > \
        season_features(CITIES["karachi"], jan)["season_winter_inversion"].iloc[0]


def test_no_future_leakage(data):
    table = build_supervised(data, 2)
    row = table[(table["city"] == "delhi")].iloc[100]
    d = data[data["city"] == "delhi"].set_index("date")
    # history features come from the issue day, the target from issue + 2
    assert row["pm25_lag0"] == pytest.approx(d.loc[row["issue_date"], "pm2_5"])
    assert row["y_pm25"] == pytest.approx(d.loc[row["issue_date"] + pd.Timedelta(days=2), "pm2_5"])
    assert row["temp_mean_target"] == pytest.approx(d.loc[row["target_date"], "temp_mean"])
    assert "y_pm25" not in feature_columns(table)
    assert "ev_fireworks" not in feature_columns(table, use_occasions=False)


def test_evaluation_beats_persistence_on_synthetic(data):
    ev = evaluate(data, "2024-06-01")
    res = ev.results[1]["overall"]
    model = res["Model (history + weather + occasions)"]["mae"]
    assert model < res["Persistence (today's AQI)"]["mae"]
    assert model <= res["Model without occasion features"]["mae"] * 1.05
    assert "MAE" in ev.to_markdown()


def test_impact_finds_fireworks(data):
    table = event_impact(data)
    diwali = table[(table["event"] == "Diwali") & (table["city"] == "delhi")]
    assert not diwali.empty and diwali["median_uplift_pct"].iloc[0] > 10


def test_forecast_from_frame(data):
    models = train_all(data)
    frame = data[data["city"] == "delhi"].copy()
    last = frame["date"].max()
    future = pd.DataFrame({"date": pd.date_range(last + pd.Timedelta(days=1), periods=3), "city": "delhi"})
    for col in ("temp_mean", "temp_min", "temp_max", "precip", "wind_max", "wind_dir", "rh_mean"):
        future[col] = frame[col].iloc[-1]
    out = forecast_city("delhi", models, pd.concat([frame, future], ignore_index=True))
    assert len(out) == 3
    assert out["aqi"].between(0, 500).all()
    assert set(out.columns) >= {"date", "aqi", "category", "drivers"}
