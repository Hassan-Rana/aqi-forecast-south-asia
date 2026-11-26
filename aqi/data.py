"""Download and cache daily air-quality + weather data.

Sources (free, no API key):
* Open-Meteo Air Quality API – hourly PM2.5, PM10, NO2, SO2, CO, O3 and dust from the
  CAMS global reanalysis/forecast (history from Aug 2022), averaged to local days.
* Open-Meteo Historical Weather API (ERA5) – daily temperature, wind, humidity, rain.
* Open-Meteo Forecast API – recent days and the next days' weather for forecasting.

CAMS values are model estimates on a ~40 km grid, not station readings: they follow
the real day-to-day pattern well but can under-read the worst smog peaks. If you have
ground-station data (e.g. the CPCB "city_day.csv" dataset), use ``load_cpcb_csv``.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from .cities import CITIES, City, get_city

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AQ_HOURLY = ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "ozone", "dust"]
AQ_RENAME = {"nitrogen_dioxide": "no2", "sulphur_dioxide": "so2", "carbon_monoxide": "co", "ozone": "o3"}
WX_DAILY = ["temperature_2m_mean", "temperature_2m_min", "temperature_2m_max", "precipitation_sum",
            "wind_speed_10m_max", "wind_direction_10m_dominant", "relative_humidity_2m_mean"]
WX_RENAME = {"temperature_2m_mean": "temp_mean", "temperature_2m_min": "temp_min",
             "temperature_2m_max": "temp_max", "precipitation_sum": "precip",
             "wind_speed_10m_max": "wind_max", "wind_direction_10m_dominant": "wind_dir",
             "relative_humidity_2m_mean": "rh_mean"}
WEATHER_COLUMNS = list(WX_RENAME.values())
POLLUTANT_COLUMNS = ["pm2_5", "pm10", "no2", "so2", "co", "o3", "dust"]
HISTORY_START = date(2022, 8, 5)  # first full day of CAMS data in the Open-Meteo archive


def _get(url: str, params: dict, retries: int = 6) -> dict:
    """GET with retries on timeouts, connection errors, rate limits and server errors."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=120)
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            time.sleep(0.5)  # be polite to the free API
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        raise RuntimeError(f"{url} returned {r.status_code}: {r.text[:300]}")
    raise RuntimeError(f"{url} failed after {retries} attempts")


def hourly_to_daily(payload: dict) -> pd.DataFrame:
    """Average an Open-Meteo hourly air-quality payload into local calendar days."""
    h = pd.DataFrame(payload["hourly"])
    h["date"] = pd.to_datetime(h["time"]).dt.normalize()
    cols = [c for c in AQ_HOURLY if c in h]
    daily = h.groupby("date")[cols].mean()
    daily["aq_hours"] = h.groupby("date")["pm2_5"].count()
    return daily.rename(columns=AQ_RENAME)


def fetch_air_quality(city: City, start: date, end: date) -> pd.DataFrame:
    frames = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(end, chunk_start + timedelta(days=181))
        payload = _get(AQ_URL, {"latitude": city.lat, "longitude": city.lon, "hourly": ",".join(AQ_HOURLY),
                                "timezone": city.timezone, "start_date": chunk_start.isoformat(),
                                "end_date": chunk_end.isoformat()})
        frames.append(hourly_to_daily(payload))
        chunk_start = chunk_end + timedelta(days=1)
    return pd.concat(frames)


def _daily_frame(payload: dict) -> pd.DataFrame:
    d = pd.DataFrame(payload["daily"])
    d["date"] = pd.to_datetime(d.pop("time"))
    return d.set_index("date").rename(columns=WX_RENAME)


def fetch_weather_history(city: City, start: date, end: date) -> pd.DataFrame:
    payload = _get(ARCHIVE_URL, {"latitude": city.lat, "longitude": city.lon, "daily": ",".join(WX_DAILY),
                                 "timezone": city.timezone, "start_date": start.isoformat(),
                                 "end_date": end.isoformat()})
    return _daily_frame(payload)


def fetch_weather_recent(city: City, past_days: int = 30, forecast_days: int = 4) -> pd.DataFrame:
    """Recent observed-ish weather plus the forecast for the coming days."""
    payload = _get(FORECAST_URL, {"latitude": city.lat, "longitude": city.lon, "daily": ",".join(WX_DAILY),
                                  "timezone": city.timezone, "past_days": past_days,
                                  "forecast_days": forecast_days})
    return _daily_frame(payload)


def fetch_air_quality_recent(city: City, past_days: int = 30) -> pd.DataFrame:
    payload = _get(AQ_URL, {"latitude": city.lat, "longitude": city.lon, "hourly": ",".join(AQ_HOURLY),
                            "timezone": city.timezone, "past_days": past_days, "forecast_days": 1})
    daily = hourly_to_daily(payload)
    today = pd.Timestamp.now(tz=city.timezone).tz_localize(None).normalize()
    return daily[daily.index < today]  # keep complete days only


def build_city_history(city_key: str, start: date = HISTORY_START, end: date | None = None) -> pd.DataFrame:
    city = get_city(city_key)
    end = end or (date.today() - timedelta(days=6))  # ERA5 archive lags ~5 days
    aq = fetch_air_quality(city, start, end)
    wx = fetch_weather_history(city, start, end)
    df = aq.join(wx, how="inner")
    df.insert(0, "city", city.key)
    return df


def save_history(df: pd.DataFrame, city_key: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{city_key}.csv"
    df.reset_index().rename(columns={"index": "date"}).to_csv(path, index=False, float_format="%.3f")
    return path


def load_history(cities: list[str] | None = None, raw_dir: Path | None = None) -> pd.DataFrame:
    folder = raw_dir or RAW_DIR
    frames = []
    for key in cities or list(CITIES):
        path = folder / f"{key}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, parse_dates=["date"]))
    if not frames:
        raise FileNotFoundError(f"No city data in {folder}. Run: python -m aqi fetch")
    return pd.concat(frames, ignore_index=True).sort_values(["city", "date"]).reset_index(drop=True)


def load_cpcb_csv(path: str | Path, city_key: str, city_name: str | None = None) -> pd.DataFrame:
    """Load ground-station data from the CPCB "city_day.csv" layout (City, Date, PM2.5, PM10, …)
    and join ERA5 weather for the same days. The result can be saved with ``save_history``."""
    city = get_city(city_key)
    raw = pd.read_csv(path, parse_dates=["Date"])
    raw = raw[raw["City"].str.lower() == (city_name or city.name).lower()]
    df = pd.DataFrame({"date": raw["Date"].dt.normalize(), "pm2_5": raw["PM2.5"], "pm10": raw["PM10"],
                       "no2": raw.get("NO2"), "so2": raw.get("SO2"), "co": raw.get("CO"),
                       "o3": raw.get("O3")}).set_index("date")
    wx = fetch_weather_history(city, df.index.min().date(), df.index.max().date())
    df = df.join(wx, how="inner")
    df.insert(0, "city", city.key)
    return df
