"""Streamlit dashboard:  streamlit run app.py"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from aqi.aqi_calc import aqi_array
from aqi.cities import CITIES
from aqi.data import load_history

st.set_page_config(page_title="South Asia AQI Forecast", page_icon="🌫️", layout="wide")
st.title("🌫️ South Asia AQI forecast")
st.caption("History + weather forecast + festivals and seasons (Diwali, Shab-e-Barat, New Year fireworks, "
           "Lohri bonfires, crop-residue burning, winter inversion) → AQI for today and the next 2 days.")

city_key = st.sidebar.selectbox("City", list(CITIES), format_func=lambda k: CITIES[k].name)
city = CITIES[city_key]
scale = "India NAQI (CPCB)" if city.standard == "cpcb" else "US EPA"
st.sidebar.write(f"AQI scale: **{scale}**")

tab_fc, tab_hist, tab_events, tab_eval = st.tabs(["Forecast", "History", "Occasion impact", "Model accuracy"])

with tab_fc:
    if st.button("Get live forecast", type="primary"):
        from aqi.forecast import forecast_city
        try:
            with st.spinner("Downloading the last 30 days and the weather forecast…"):
                fc = forecast_city(city_key)
            cols = st.columns(len(fc))
            for col, row in zip(cols, fc.itertuples()):
                col.metric(str(row.date), f"AQI {row.aqi:.0f}", row.category, delta_color="off")
                col.caption(row.drivers or "no special occasion")
            st.dataframe(fc, hide_index=True, use_container_width=True)
        except FileNotFoundError as exc:
            st.error(f"{exc}")
        except Exception as exc:  # network errors etc.
            st.error(f"Forecast failed: {exc}")

try:
    hist = load_history([city_key])
except FileNotFoundError:
    hist = None

with tab_hist:
    if hist is None:
        st.info("No history yet – run `python -m aqi fetch`.")
    else:
        h = hist.copy()
        h["AQI"] = aqi_array(h["pm2_5"], h["pm10"], city.standard)
        st.line_chart(h.set_index("date")[["AQI"]])
        st.line_chart(h.set_index("date")[["pm2_5", "pm10"]])

with tab_events:
    if hist is not None:
        from aqi.impact import event_impact
        imp = event_impact(hist)
        st.write("Median rise in PM2.5 during each occasion compared with the surrounding days:")
        st.dataframe(imp, hide_index=True, use_container_width=True)

with tab_eval:
    report = Path(__file__).parent / "reports" / "metrics.md"
    st.markdown(report.read_text() if report.exists() else "Run `python -m aqi evaluate` to create the report.")
