"""Cities covered by the forecaster.

`crop_burning_exposure` (0–1) says how strongly a city is affected by post-harvest
crop-residue (stubble) burning upwind, mainly in Indian and Pakistani Punjab in
Oct–Nov. `standard` is the national AQI scale used to report results.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    key: str
    name: str
    country: str  # "IN", "PK", "BD"
    lat: float
    lon: float
    timezone: str
    standard: str  # "cpcb" (India NAQI) or "us" (US EPA scale, used in PK and BD)
    crop_burning_exposure: float
    coastal: bool


CITIES: dict[str, City] = {
    c.key: c
    for c in [
        City("delhi", "Delhi", "IN", 28.6139, 77.2090, "Asia/Kolkata", "cpcb", 1.0, False),
        City("lahore", "Lahore", "PK", 31.5204, 74.3587, "Asia/Karachi", "us", 1.0, False),
        City("kolkata", "Kolkata", "IN", 22.5726, 88.3639, "Asia/Kolkata", "cpcb", 0.4, False),
        City("dhaka", "Dhaka", "BD", 23.8103, 90.4125, "Asia/Dhaka", "us", 0.3, False),
        City("karachi", "Karachi", "PK", 24.8607, 67.0011, "Asia/Karachi", "us", 0.1, True),
        City("mumbai", "Mumbai", "IN", 19.0760, 72.8777, "Asia/Kolkata", "cpcb", 0.0, True),
    ]
}

CITY_CODES = {key: i for i, key in enumerate(CITIES)}


def get_city(key: str) -> City:
    try:
        return CITIES[key.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown city '{key}'. Choose from: {', '.join(CITIES)}") from exc
