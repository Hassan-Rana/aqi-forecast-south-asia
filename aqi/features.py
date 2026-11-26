"""Turn daily city data into a supervised forecasting table.

A row is issued at the end of day ``t`` (data complete up to ``t``) and predicts day
``t + h``. It combines

* **recent history** – today's and past days' PM2.5/PM10, 3/7/14-day means, trend,
  other pollutants and dust;
* **longer history** – the same week one year earlier;
* **weather** – today's weather and the weather expected on the target day
  (in training the observed ERA5 value is used, at forecast time the weather forecast);
* **occasions** – festival/fireworks/bonfire windows and seasonal pressures
  (crop burning, winter inversion, dust, monsoon) on the target day;
* calendar terms and the city.

Nothing from after day ``t`` is used except weather and the event calendar, which
are known in advance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cities import CITY_CODES, get_city
from .events import EVENT_COLUMNS, SEASON_COLUMNS, event_features, season_features

HORIZONS = (1, 2, 3)
TARGET_WX = ["temp_min", "temp_mean", "wind_max", "rh_mean", "precip"]


def _city_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex one city's data to a continuous daily index (gaps become NaN)."""
    df = df.set_index("date").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    out = df.reindex(full)
    out["city"] = df["city"].iloc[0]
    out.index.name = "date"
    return out


def history_features(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    pm = d["pm2_5"]
    for lag in (0, 1, 2, 6):
        f[f"pm25_lag{lag}"] = pm.shift(lag)
    f["pm10_lag0"] = d["pm10"]
    f["pm10_lag1"] = d["pm10"].shift(1)
    for w in (3, 7, 14):
        f[f"pm25_mean{w}"] = pm.rolling(w, min_periods=max(2, w // 2)).mean()
    f["pm25_std7"] = pm.rolling(7, min_periods=4).std()
    f["pm25_trend3"] = pm - pm.shift(3)
    f["pm_ratio"] = d["pm2_5"] / d["pm10"].replace(0, np.nan)
    for col in ("no2", "co", "so2", "o3", "dust"):
        if col in d:
            f[f"{col}_lag0"] = d[col]
    for col in ("temp_mean", "wind_max", "rh_mean", "precip"):
        f[f"{col}_today"] = d[col]
    f["precip_3d"] = d["precip"].rolling(3, min_periods=1).sum()
    return f


def target_day_features(d: pd.DataFrame, city_key: str, h: int, target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    city = get_city(city_key)
    f = pd.DataFrame(index=d.index)
    for col in TARGET_WX:
        f[f"{col}_target"] = d[col].shift(-h)
    wd = np.deg2rad(d["wind_dir"].shift(-h))
    f["wind_dir_sin_target"] = np.sin(wd)
    f["wind_dir_cos_target"] = np.cos(wd)
    # same week last year (historical pattern for this time of year)
    f["pm25_last_year"] = d["pm2_5"].rolling(7, center=True, min_periods=3).mean().shift(365 - h)
    ev = event_features(city, target_dates)
    se = season_features(city, target_dates)
    for col in EVENT_COLUMNS:
        f[col] = ev[col].values
    for col in SEASON_COLUMNS:
        f[col] = se[col].values
    doy = target_dates.dayofyear.values
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["dow"] = target_dates.dayofweek.values
    f["horizon"] = h
    f["city_code"] = CITY_CODES[city_key]
    f["event_names"] = ev["event_names"].values
    return f


def build_supervised(data: pd.DataFrame, h: int) -> pd.DataFrame:
    """Rows = (city, issue date t); targets = PM2.5 / PM10 on t + h."""
    tables = []
    for city_key, g in data.groupby("city"):
        d = _city_frame(g)
        target_dates = d.index + pd.Timedelta(days=h)
        f = history_features(d).join(target_day_features(d, city_key, h, target_dates))
        f["city"] = city_key
        f["issue_date"] = d.index
        f["target_date"] = target_dates
        f["y_pm25"] = d["pm2_5"].shift(-h)
        f["y_pm10"] = d["pm10"].shift(-h)
        tables.append(f)
    out = pd.concat(tables, ignore_index=True)
    return out.dropna(subset=["pm25_lag0", "y_pm25", "y_pm10"]).reset_index(drop=True)


NON_FEATURES = {"city", "issue_date", "target_date", "y_pm25", "y_pm10", "event_names"}
OCCASION_FEATURES = set(EVENT_COLUMNS) | set(SEASON_COLUMNS)


def feature_columns(table: pd.DataFrame, use_occasions: bool = True) -> list[str]:
    cols = [c for c in table.columns if c not in NON_FEATURES]
    if not use_occasions:
        cols = [c for c in cols if c not in OCCASION_FEATURES]
    return cols
