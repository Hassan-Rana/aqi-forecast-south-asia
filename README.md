# 🌫️ AQI forecast for South Asia

Forecasts the **Air Quality Index for today and the next two days** in six large cities of the
Indian subcontinent: **Delhi, Lahore, Kolkata, Dhaka, Karachi and Mumbai**.

It combines

* **historical data**: more than three years of daily PM2.5, PM10, NO₂, SO₂, CO, O₃ and dust,
* **current data**: the last 30 days of air quality and the weather forecast for the coming days,
* **occasions and events that add to pollution**: Diwali, New Year's Eve, Shab-e-Barat and
  Independence Day fireworks, Lohri and Holika Dahan bonfires, Dussehra effigy burning,
  Eid traffic and Eid al-Adha cooking smoke,
* **seasons**: crop-residue (stubble) burning in Punjab in Oct–Nov and late April, the winter
  temperature inversion that traps smoke near the ground, pre-monsoon dust and the monsoon wash-out.

The data, the evaluation and the forecast are refreshed **every day by GitHub Actions**.

* Latest forecast: [`reports/latest_forecast.md`](reports/latest_forecast.md)
* Model accuracy: [`reports/metrics.md`](reports/metrics.md)
* How much each occasion raises PM2.5: [`reports/event_impact.csv`](reports/event_impact.csv)

---

## How it works

```
Open-Meteo CAMS air quality (hourly → daily)  ┐
Open-Meteo ERA5 / forecast weather (daily)    ├─► features ─► gradient-boosted trees ─► PM2.5, PM10 ─► AQI
Festival & event calendar + seasonal signals  ┘     (per horizon: 1, 2, 3 days)          (national scale)
```

**Features for a forecast issued at the end of day *t* for day *t + h*:**

| Group | Examples |
|---|---|
| Recent history | PM2.5 today and 1, 2, 6 days ago, 3/7/14-day means, 7-day variability, 3-day trend, PM10, NO₂, CO, SO₂, O₃, dust |
| Long history | PM2.5 in the same week one year earlier |
| Weather | today's weather and the target day's minimum/mean temperature, maximum wind, wind direction, humidity and rain |
| Occasions | fireworks, bonfire, effigy-burning, festival-traffic and Eid al-Adha intensities on the target day, with a lead-in and a decaying after-effect, and days since the last fireworks night |
| Seasons | crop-burning pressure (scaled by how exposed each city is), winter inversion, dust, monsoon |
| Calendar | day of year (sine/cosine), weekday, city |

**Model:** scikit-learn `HistGradientBoostingRegressor` on log-concentrations, one model per pollutant
and horizon, trained on all cities together with the city as a categorical feature. The AQI is the
higher of the PM2.5 and PM10 sub-indices, on **India's National AQI (CPCB)** scale for Indian cities and
on the **US EPA** scale (2024 breakpoints) for Pakistan and Bangladesh.

**Evaluation:** a strict time split. The model is trained only on data from before the last 12 months
and tested on those 12 months, which include a full winter and the Diwali/crop-burning season. It is
compared with persistence (tomorrow = today), the 7-day mean, and the same model **without** the
occasion features, to measure what the festival and season signals add, overall and on event days.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m aqi fetch            # download history for all six cities into data/raw/
python -m aqi evaluate         # time-split evaluation → reports/metrics.md
python -m aqi impact           # occasion impact table → reports/event_impact.csv
python -m aqi train            # final models on all data → models/
python -m aqi forecast delhi   # today + next 2 days (or: forecast all)
streamlit run app.py           # dashboard: forecast, history, occasion impact, accuracy
python -m pytest -q            # tests (offline, on synthetic data)
```

Example output:

```
 city       date  pm2_5  pm10  aqi  category           scale  drivers
Delhi 2026-11-27  152.3 241.0  326  Very Poor  India NAQI (CPCB)  crop-residue burning season; winter inversion …; light wind (8 km/h)
```

## Adding occasions

Edit [`data/events.csv`](data/events.csv) or create `data/custom_events.csv` with the same columns:

```
date,event,kind,countries,lead_days,lag_days,strength
2026-12-16,Victory Day fireworks,fireworks,BD,0,1,0.4
```

`kind` is one of `fireworks`, `bonfire`, `effigy_burning`, `festival_traffic` or `sacrifice_bbq`, and
`strength` is a value from 0 to 1. Lunar festival dates (Diwali, Eid, Shab-e-Barat…) move every year,
so check the table against the official calendar for your country and add the dates for new years.

## Data and limitations

* Air quality comes from the **Copernicus CAMS** global model via the free
  [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api), from August 2022
  onwards, and weather from **ERA5** via the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).
  CAMS is a model on a ~40 km grid, not a ground station: it follows day-to-day changes well but
  usually **under-reads the worst smog peaks** and smooths out very local events such as
  one-night fireworks. `aqi.data.load_cpcb_csv` loads CPCB station data (the `city_day.csv` layout) if
  you want to train on ground measurements instead.
* In training, the target day's weather is the observed (ERA5) value. At forecast time it is the
  weather *forecast*, which adds some error, more so at 2–3 days ahead.
* The event table lists the main national occasions. Local events (rallies, sports finals, large
  weddings, industrial accidents) can be added as custom events.

## Project layout

```
aqi/
  cities.py     cities, coordinates, AQI scale, crop-burning exposure
  aqi_calc.py   CPCB and US EPA AQI from PM2.5 / PM10, categories
  events.py     festival calendar → event features; seasonal signals
  data.py       Open-Meteo download, daily aggregation, CPCB loader
  features.py   supervised table (no look-ahead), feature groups
  model.py      training, time-split evaluation with baselines and ablation
  impact.py     historical uplift of PM2.5 around each occasion
  forecast.py   live forecast with plain-language drivers
  synthetic.py  synthetic data for offline tests only
app.py          Streamlit dashboard
data/events.csv festival and occasion calendar
.github/workflows/forecast.yml  daily data refresh, evaluation and forecast
```
