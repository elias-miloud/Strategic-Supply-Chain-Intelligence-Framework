"""
Strategic Supply Chain Intelligence Framework
Executive Dashboard — Streamlit App

Interactive CODIR-level dashboard for supply chain performance monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Supply Chain Intelligence | Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Synthetic Data
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range("2022-01-03", periods=104, freq="W")
    skus = [f"SKU_{i:03d}" for i in range(1, 6)]
    records = []
    for sku in skus:
        base = np.random.randint(100, 500)
        for i, date in enumerate(dates):
            trend = base + 0.5 * i
            seasonal = 0.2 * base * np.sin(2 * np.pi * i / 52)
            noise = np.random.normal(0, base * 0.05)
            demand = max(0, trend + seasonal + noise)
            forecast = demand * np.random.uniform(0.9, 1.1)
            records.append({
                "date": date, "sku": sku,
                "demand": round(demand),
                "forecast": round(forecast),
                "stock": round(demand * np.random.uniform(1.2, 2.5)),
                "service_rate": round(min(1.0, np.random.uniform(0.85, 1.0)), 3)
            })
    return pd.DataFrame(records)


df = load_data()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

st.sidebar.image("https://img.shields.io/badge/Strategic%20SC%20Intelligence-Executive%20Dashboard-darkblue", use_column_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

selected_sku = st.sidebar.selectbox("SKU", df["sku"].unique())
date_range = st.sidebar.date_input(
    "Date Range",
    [df["date"].min(), df["date"].max()]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Forecast Settings")
horizon = st.sidebar.slider("Forecast Horizon (weeks)", 4, 26, 12)
confidence = st.sidebar.slider("Confidence Interval (%)", 80, 99, 95)

# ─────────────────────────────────────────────
# Main Dashboard
# ─────────────────────────────────────────────

st.title("📊 Strategic Supply Chain Intelligence")
st.markdown("**Executive Performance Dashboard** | CODIR-Level Decision Support")
st.markdown("---")

sku_df = df[df["sku"] == selected_sku].copy()

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

avg_demand = sku_df["demand"].mean()
mape = abs(sku_df["demand"] - sku_df["forecast"]).mean() / avg_demand * 100
avg_service = sku_df["service_rate"].mean() * 100
avg_stock = sku_df["stock"].mean()

col1.metric("Avg Weekly Demand", f"{avg_demand:,.0f} units", "+2.3%")
col2.metric("Forecast MAPE", f"{mape:.1f}%", "-1.2pp", delta_color="inverse")
col3.metric("Service Rate", f"{avg_service:.1f}%", "+0.8pp")
col4.metric("Avg Stock Level", f"{avg_stock:,.0f} units", "-5.1%", delta_color="inverse")

st.markdown("---")

# Demand vs Forecast Chart
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=sku_df["date"], y=sku_df["demand"],
    name="Actual Demand", line=dict(color="#1f77b4", width=2)
))
fig1.add_trace(go.Scatter(
    x=sku_df["date"], y=sku_df["forecast"],
    name="Forecast", line=dict(color="#ff7f0e", width=2, dash="dash")
))

# Confidence band
upper = sku_df["forecast"] * (1 + (100 - confidence) / 100 * 2)
lower = sku_df["forecast"] * (1 - (100 - confidence) / 100 * 2)
fig1.add_trace(go.Scatter(
    x=pd.concat([sku_df["date"], sku_df["date"][::-1]]),
    y=pd.concat([upper, lower[::-1]]),
    fill="toself", fillcolor="rgba(255,127,14,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name=f"{confidence}% Confidence"
))

fig1.update_layout(
    title=f"Demand vs Forecast — {selected_sku}",
    xaxis_title="Date", yaxis_title="Units",
    template="plotly_white", height=400
)
st.plotly_chart(fig1, use_container_width=True)

# Bottom charts
col_a, col_b = st.columns(2)

with col_a:
    fig2 = px.bar(
        sku_df.tail(26), x="date", y="stock",
        title="Stock Level — Last 26 Weeks",
        color="stock", color_continuous_scale="Blues",
        template="plotly_white"
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    fig3 = px.line(
        sku_df.tail(52), x="date", y="service_rate",
        title="Service Rate Evolution",
        template="plotly_white"
    )
    fig3.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="Target 95%")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.caption("Strategic Supply Chain Intelligence Framework | Elias Miloud | github.com/elias-miloud")
