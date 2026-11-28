"""Streamlit dashboard:  streamlit run app.py"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from aqi.aqi_calc import aqi_array
from aqi.cities import CITIES
from aqi.data import load_history

ASSETS = Path(__file__).parent / "assets"
ICON = ASSETS / "icon.svg"

try:
    st.set_page_config(page_title="AQI Forecast · South Asia", page_icon=str(ICON), layout="wide")
except Exception:  # very old Streamlit, or icon missing
    st.set_page_config(page_title="AQI Forecast · South Asia", page_icon="🌫️", layout="wide")
if ICON.exists():
    try:
        st.logo(str(ICON), size="large")  # top-left logo (Streamlit >= 1.39)
    except TypeError:
        st.logo(str(ICON))  # Streamlit 1.35–1.38: no size option
    except Exception:
        pass  # Streamlit < 1.35 has no st.logo; the header below still shows the logo
    icon_b64 = base64.b64encode(ICON.read_bytes()).decode()
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:18px;margin:0 0 6px">
        <img src="data:image/svg+xml;base64,{icon_b64}" width="72" height="72" alt="">
        <div><div style="font-size:2.3rem;font-weight:700;line-height:1.1">AQI<span style="color:#ff7043">·</span>Forecast</div>
        <div style="letter-spacing:.18em;font-size:.85rem;opacity:.7">SOUTH ASIA · 3-DAY OUTLOOK</div>
        <div style="height:5px;width:300px;border-radius:3px;margin-top:6px;background:linear-gradient(90deg,
        #34c759,#d4e157,#ffca28,#ff7043,#e53935,#8e24aa)"></div></div></div>""",
        unsafe_allow_html=True)
else:
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
