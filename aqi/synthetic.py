"""Synthetic daily data with realistic South-Asian patterns, for tests and offline demos.

It is NOT real data. It exists so the pipeline can be tested without internet access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cities import CITIES
from .events import event_features, season_features


def make_synthetic(start: str = "2022-08-01", end: str = "2024-12-31", seed: int = 0,
                   cities: list[str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    frames = []
    for key in cities or list(CITIES):
        city = CITIES[key]
        se = season_features(city, dates)
        ev = event_features(city, dates)
        doy = dates.dayofyear.values
        temp = 26 - 9 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 1.5, len(dates))
        wind = np.clip(12 + 6 * se["season_monsoon"].values + rng.gamma(2, 2, len(dates)), 2, None)
        precip = np.where(rng.random(len(dates)) < 0.05 + 0.5 * se["season_monsoon"].values,
                          rng.gamma(2, 6, len(dates)), 0.0)
        base = 45 if city.coastal else 70
        signal = (base * (1 + 1.4 * se["season_winter_inversion"].values + 1.2 * se["season_crop_burning"].values
                          + 0.3 * se["season_dust"].values - 0.6 * se["season_monsoon"].values)
                  * (1 + 0.8 * ev["ev_fireworks"].values + 0.3 * ev["ev_bonfire"].values)
                  * (12 / wind) ** 0.4 * np.exp(-0.03 * precip))
        noise = np.exp(rng.normal(0, 0.15, len(dates)))
        pm25 = pd.Series(signal * noise).ewm(alpha=0.6).mean().values
        pm10 = pm25 * (1.6 + 0.8 * se["season_dust"].values) * np.exp(rng.normal(0, 0.08, len(dates)))
        frames.append(pd.DataFrame({
            "date": dates, "city": key, "pm2_5": pm25, "pm10": pm10, "no2": pm25 * 0.4, "co": pm25 * 9,
            "so2": 8.0, "o3": 40 + 20 * se["season_dust"].values, "dust": 30 * se["season_dust"].values,
            "temp_mean": temp, "temp_min": temp - 6, "temp_max": temp + 6, "precip": precip,
            "wind_max": wind, "wind_dir": rng.uniform(0, 360, len(dates)), "rh_mean": 55 + 25 * se["season_monsoon"].values,
        }))
    return pd.concat(frames, ignore_index=True)
