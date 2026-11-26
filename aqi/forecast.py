"""Live forecast: today's and the next days' AQI from recent data + weather forecast."""
from __future__ import annotations

import pandas as pd

from .aqi_calc import aqi, category
from .cities import get_city
from .data import fetch_air_quality_recent, fetch_weather_recent, load_history
from .events import event_features, season_features
from .features import build_supervised
from .model import HorizonModel, load_models


def assemble_recent(city_key: str, history: pd.DataFrame | None = None,
                    recent_aq: pd.DataFrame | None = None, recent_wx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stored history (for last-year values) + the last 30 days of air quality + weather
    for recent and coming days, as one continuous daily frame for this city."""
    city = get_city(city_key)
    recent_aq = recent_aq if recent_aq is not None else fetch_air_quality_recent(city)
    recent_wx = recent_wx if recent_wx is not None else fetch_weather_recent(city)
    recent = recent_wx.join(recent_aq, how="left")
    recent.insert(0, "city", city.key)
    recent = recent.reset_index().rename(columns={"index": "date"})
    if history is not None:
        hist = history[(history["city"] == city.key) & (history["date"] < recent["date"].min())]
        recent = pd.concat([hist, recent], ignore_index=True)
    return recent


def drivers(city_key: str, day: pd.Timestamp, wx: pd.Series) -> list[str]:
    """Plain-language reasons that push AQI up or down on this day."""
    city = get_city(city_key)
    ev = event_features(city, pd.DatetimeIndex([day])).iloc[0]
    se = season_features(city, pd.DatetimeIndex([day])).iloc[0]
    out = []
    if ev["event_names"]:
        out.append(f"occasion: {ev['event_names']}")
    if se["season_crop_burning"] > 0.3:
        out.append("crop-residue burning season")
    if se["season_winter_inversion"] > 0.6:
        out.append("winter inversion season (smoke trapped near the ground)")
    if se["season_dust"] > 0.5:
        out.append("pre-monsoon dust season")
    if pd.notna(wx.get("wind_max")) and wx["wind_max"] < 10:
        out.append(f"light wind ({wx['wind_max']:.0f} km/h)")
    if pd.notna(wx.get("precip")) and wx["precip"] >= 2:
        out.append(f"rain ({wx['precip']:.0f} mm) washes particles out")
    return out


def forecast_city(city_key: str, models: dict[int, HorizonModel] | None = None,
                  frame: pd.DataFrame | None = None) -> pd.DataFrame:
    models = models or load_models()
    city = get_city(city_key)
    if frame is None:
        try:
            history = load_history([city.key])
        except FileNotFoundError:
            history = None
        frame = assemble_recent(city.key, history)
    frame = frame.sort_values("date").reset_index(drop=True)
    issue = frame.loc[frame["pm2_5"].notna(), "date"].max()  # last complete day with measurements
    rows = []
    for h, model in sorted(models.items()):
        table = build_supervised_for_issue(frame, h, issue)
        if table.empty:
            continue
        pred = model.predict(table).iloc[0]
        target = issue + pd.Timedelta(days=h)
        value = aqi(pred["pm2_5"], pred["pm10"], city.standard)
        wx = frame.set_index("date").loc[target] if target in set(frame["date"]) else pd.Series(dtype=float)
        rows.append({"city": city.name, "date": target.date(), "pm2_5": round(pred["pm2_5"], 1),
                     "pm10": round(pred["pm10"], 1), "aqi": value, "category": category(value, city.standard),
                     "scale": "India NAQI (CPCB)" if city.standard == "cpcb" else "US EPA",
                     "drivers": "; ".join(drivers(city.key, target, wx))})
    return pd.DataFrame(rows)


def build_supervised_for_issue(frame: pd.DataFrame, h: int, issue: pd.Timestamp) -> pd.DataFrame:
    """Feature row for one issue date. Target columns are unknown, so fill them to avoid dropping."""
    f = frame.copy()
    target = issue + pd.Timedelta(days=h)
    mask = f["date"] == target
    for col in ("pm2_5", "pm10"):
        f.loc[mask & f[col].isna(), col] = -1.0  # placeholder; never used as a feature for this row
    table = build_supervised(f, h)
    return table[table["issue_date"] == issue]
