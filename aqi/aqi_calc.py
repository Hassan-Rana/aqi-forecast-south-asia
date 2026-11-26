"""AQI from pollutant concentrations.

Two scales are supported:
* ``cpcb`` – India's National AQI (CPCB), 0–500, categories Good … Severe.
* ``us``   – US EPA AQI (2024 PM2.5 breakpoints), used by Pakistan and Bangladesh
  monitoring networks and most public dashboards in the region.

Only PM2.5 and PM10 (24-hour means, µg/m³) are used. In South Asian cities the
daily AQI is almost always set by particulate matter, so this is the quantity the
model forecasts.
"""
from __future__ import annotations

import math

import numpy as np

# (conc_low, conc_high, aqi_low, aqi_high)
_BREAKPOINTS = {
    "cpcb": {
        "pm2_5": [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200), (90, 120, 200, 300),
                  (120, 250, 300, 400), (250, 500, 400, 500)],
        "pm10": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200), (250, 350, 200, 300),
                 (350, 430, 300, 400), (430, 600, 400, 500)],
    },
    "us": {
        "pm2_5": [(0, 9.0, 0, 50), (9.0, 35.4, 50, 100), (35.4, 55.4, 100, 150), (55.4, 125.4, 150, 200),
                  (125.4, 225.4, 200, 300), (225.4, 325.4, 300, 500)],
        "pm10": [(0, 54, 0, 50), (54, 154, 50, 100), (154, 254, 100, 150), (254, 354, 150, 200),
                 (354, 424, 200, 300), (424, 604, 300, 500)],
    },
}

CATEGORIES = {
    "cpcb": [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"), (300, "Poor"),
             (400, "Very Poor"), (math.inf, "Severe")],
    "us": [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for Sensitive Groups"),
           (200, "Unhealthy"), (300, "Very Unhealthy"), (math.inf, "Hazardous")],
}


def sub_index(conc: float, pollutant: str, standard: str = "cpcb") -> float:
    """Sub-index for one pollutant, linear between breakpoints, capped at 500."""
    if conc is None or (isinstance(conc, float) and math.isnan(conc)):
        return math.nan
    conc = max(float(conc), 0.0)
    for c_lo, c_hi, a_lo, a_hi in _BREAKPOINTS[standard][pollutant]:
        if conc <= c_hi:
            return a_lo + (a_hi - a_lo) * (conc - c_lo) / (c_hi - c_lo)
    return 500.0


def aqi(pm2_5: float, pm10: float, standard: str = "cpcb") -> float:
    """AQI = the highest of the PM2.5 and PM10 sub-indices."""
    vals = [sub_index(pm2_5, "pm2_5", standard), sub_index(pm10, "pm10", standard)]
    vals = [v for v in vals if not math.isnan(v)]
    return round(max(vals)) if vals else math.nan


def aqi_array(pm2_5, pm10, standard: str = "cpcb") -> np.ndarray:
    return np.array([aqi(a, b, standard) for a, b in zip(pm2_5, pm10)], dtype=float)


def category(value: float, standard: str = "cpcb") -> str:
    if value is None or math.isnan(value):
        return "Unknown"
    for upper, name in CATEGORIES[standard]:
        if value <= upper:
            return name
    return CATEGORIES[standard][-1][1]


def category_index(value: float, standard: str = "cpcb") -> int:
    for i, (upper, _) in enumerate(CATEGORIES[standard]):
        if value <= upper:
            return i
    return len(CATEGORIES[standard]) - 1
