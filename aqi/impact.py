"""How much does each occasion raise pollution? (historical event-impact analysis)

For every occurrence of an event in a city, the mean PM2.5 over the event window
(the day itself plus its lag days) is compared with a baseline: the mean of days
-10…-4 before and +6…+10 after, which takes out most of the seasonal trend. The
table reports the median uplift over all occurrences, so one unusual year does not
dominate.
"""
from __future__ import annotations

import pandas as pd

from .cities import CITIES
from .events import load_events


def event_impact(data: pd.DataFrame) -> pd.DataFrame:
    events = load_events()
    rows = []
    for city_key, g in data.groupby("city"):
        city = CITIES[city_key]
        pm = g.set_index("date")["pm2_5"].sort_index()
        for ev in events[events["countries"].apply(lambda cs: city.country in cs)].itertuples(index=False):
            day = ev.date
            window = pm.loc[day: day + pd.Timedelta(days=int(ev.lag_days))]
            base = pd.concat([pm.loc[day - pd.Timedelta(days=10): day - pd.Timedelta(days=4)],
                              pm.loc[day + pd.Timedelta(days=6): day + pd.Timedelta(days=10)]])
            if len(window) == 0 or len(base) < 6:
                continue
            rows.append({"city": city_key, "event": ev.event, "kind": ev.kind, "date": day.date(),
                         "event_pm25": window.mean(), "baseline_pm25": base.mean(),
                         "uplift_pct": 100 * (window.mean() / base.mean() - 1)})
    occ = pd.DataFrame(rows)
    if occ.empty:
        return occ
    summary = (occ.groupby(["event", "city"])
               .agg(occurrences=("uplift_pct", "size"), median_uplift_pct=("uplift_pct", "median"),
                    mean_event_pm25=("event_pm25", "mean"))
               .reset_index().sort_values("median_uplift_pct", ascending=False))
    return summary.round(1)
