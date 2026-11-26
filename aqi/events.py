"""Occasions, festivals and seasons that change air quality.

Two kinds of signal are produced for every (city, date):

1. **Events** from ``data/events.csv`` – dated occasions such as Diwali, New Year's
   Eve or Shab-e-Barat fireworks, Lohri and Holika Dahan bonfires, Dussehra effigy
   burning, Eid traffic and Eid al-Adha cooking smoke. Each event has a lead/lag
   window, so the model sees the build-up and the slow clean-out afterwards.
   Lunar festival dates move every year; check the table and add rows for new years
   or local events (``data/custom_events.csv`` is read too, if present).

2. **Seasons** – smooth seasonal pressures that are not single dates:
   crop-residue burning (Oct–Nov, and a smaller wheat-stubble peak in late April),
   the winter temperature inversion that traps smoke, pre-monsoon dust and the
   monsoon wash-out.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .cities import City

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVENT_KINDS = ["fireworks", "bonfire", "effigy_burning", "festival_traffic", "sacrifice_bbq"]
EVENT_COLUMNS = [f"ev_{k}" for k in EVENT_KINDS] + ["ev_any", "ev_days_since_fireworks"]
SEASON_COLUMNS = ["season_crop_burning", "season_winter_inversion", "season_dust", "season_monsoon"]


@lru_cache(maxsize=4)
def load_events(data_dir: str | None = None) -> pd.DataFrame:
    folder = Path(data_dir) if data_dir else DATA_DIR
    frames = [pd.read_csv(folder / "events.csv")]
    custom = folder / "custom_events.csv"
    if custom.exists():
        frames.append(pd.read_csv(custom))
    ev = pd.concat(frames, ignore_index=True)
    ev["date"] = pd.to_datetime(ev["date"])
    ev["countries"] = ev["countries"].str.split("|")
    unknown = set(ev["kind"]) - set(EVENT_KINDS)
    if unknown:
        raise ValueError(f"Unknown event kind(s) in events file: {unknown}")
    return ev


def _window_weight(offset: int, lead: int, lag: int) -> float:
    """offset = day - event_date. 1.0 on the day, 0.5 the day(s) before, decaying after."""
    if offset == 0:
        return 1.0
    if -lead <= offset < 0:
        return 0.5
    if 0 < offset <= lag:
        return 1.0 / (1 + offset)
    return 0.0


def event_features(city: City, dates: pd.DatetimeIndex, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per date with event intensities (0–1) for this city's country."""
    events = load_events() if events is None else events
    ev = events[events["countries"].apply(lambda cs: city.country in cs)]
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    out = pd.DataFrame(0.0, index=dates, columns=[f"ev_{k}" for k in EVENT_KINDS])
    names: dict[pd.Timestamp, list[str]] = {}
    for row in ev.itertuples(index=False):
        for offset in range(-int(row.lead_days), int(row.lag_days) + 1):
            day = row.date + pd.Timedelta(days=offset)
            if day not in out.index:
                continue
            w = _window_weight(offset, int(row.lead_days), int(row.lag_days)) * float(row.strength)
            col = f"ev_{row.kind}"
            out.at[day, col] = max(out.at[day, col], w)
            names.setdefault(day, []).append(row.event)
    out["ev_any"] = out.max(axis=1)

    fw_dates = np.sort(ev.loc[ev["kind"] == "fireworks", "date"].values)
    since = []
    for d in dates.values:
        prev = fw_dates[fw_dates <= d]
        since.append(min((d - prev[-1]) / np.timedelta64(1, "D"), 60.0) if len(prev) else 60.0)
    out["ev_days_since_fireworks"] = since
    out["event_names"] = [", ".join(sorted(set(names.get(d, [])))) for d in out.index]
    return out


def _bump(doy: np.ndarray, center: float, sigma: float) -> np.ndarray:
    d = np.minimum(np.abs(doy - center), 365 - np.abs(doy - center))
    return np.exp(-0.5 * (d / sigma) ** 2)


def season_features(city: City, dates: pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    doy = dates.dayofyear.values.astype(float)
    crop = city.crop_burning_exposure * (_bump(doy, 307, 14) + 0.35 * _bump(doy, 115, 10))
    inversion = np.clip(np.cos(2 * np.pi * (doy - 5) / 365.25), 0, None) * (0.5 if city.coastal else 1.0)
    return pd.DataFrame(
        {
            "season_crop_burning": crop,
            "season_winter_inversion": inversion,
            "season_dust": _bump(doy, 140, 25),
            "season_monsoon": _bump(doy, 215, 30),
        },
        index=dates,
    )
