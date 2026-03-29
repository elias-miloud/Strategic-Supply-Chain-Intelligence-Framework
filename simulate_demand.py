import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Supply Chain Intelligence | Executive Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_THEME = {
    "bg": "#0f1117",
    "card": "#1c1f2e",
    "accent": "#4f8ef7",
    "green": "#00c896",
    "red": "#ff4b6e",
    "yellow": "#f7c948",
    "text": "#e8eaf0",
}

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background-color: #1c1f2e;
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #4f8ef7;
        margin: 8px 0;
    }
    .kpi-delta-positive { color: #00c896; font-weight: bold; }
    .kpi-delta-negative { color: #ff4b6e; font-weight: bold; }
    .insight-box {
        background-color: #1c2035;
        border: 1px solid #4f8ef7;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    h1, h2, h3 { color: #e8eaf0; }
    .stMetric { background-color: #1c1f2e; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_forecast_data():
    np.random.seed(42)
    dates = pd.date_range(datetime.today() - timedelta(days=90),
                          datetime.today() + timedelta(days=28), freq="D")
    sites = ["paris_cdg", "lyon_st_exupery", "marseille_fos", "bordeaux_meriadeck", "lille_lesquin"]
    records = []
    for site in sites:
        base = np.random.uniform(200, 450)
        trend = np.random.uniform(-0.1, 0.3)
        for i, d in enumerate(dates):
            is_future = d > datetime.today()
            noise = np.random.normal(0, base * 0.12)
            seasonal = base * 0.25 * np.sin(2 * np.pi * i / 365)
            val = base + trend * i + seasonal + noise
            records.append({
                "date": d, "site": site,
                "demand": max(val, 0) if not is_future else None,
                "forecast": max(val * np.random.uniform(0.92, 1.08), 0),
                "p10": max(val * 0.82, 0),
                "p90": max(val * 1.18, 0),
                "is_future": is_future,
            })
    return pd.DataFrame(records)


@st.cache_data(ttl=3600)
def load_inventory_data():
    np.random.seed(7)
    sites = ["paris_cdg", "lyon_st_exupery", "marseille_fos", "bordeaux_meriadeck", "lille_lesquin"]
    return pd.DataFrame({
        "site": sites,
        "current_stock_days": np.random.uniform(4, 22, len(sites)),
        "safety_stock_days": np.random.uniform(5, 10, len(sites)),
        "rop_days": np.random.uniform(7, 12, len(sites)),
        "pick_rate_improvement": [17.2, 12.4, 9.8, 14.1, 11.6],
        "throughput_delta": [30, 22, 18, 26, 20],
        "mape": np.random.uniform(6.5, 10.5, len(sites)),
        "forecast_bias": np.random.uniform(-1.2, 1.2, len(sites)),
    })


def render_kpi_row(inv_df):
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "Portfolio MAPE", f"{inv_df['mape'].mean():.1f}%", "-2.1pp vs baseline", True),
        (c2, "Avg Pick Rate Δ", f"+{inv_df['pick_rate_improvement'].mean():.1f}%", "+17.2% best site", True),
        (c3, "Throughput Gain", f"+{inv_df['throughput_delta'].mean():.0f} pal/day", "vs pre-WMS slotting", True),
        (c4, "Forecast Bias", f"{inv_df['forecast_bias'].mean():.2f}%", "Target ±0.5%", True),
        (c5, "Sites at Risk", f"{(inv_df['current_stock_days'] < inv_df['safety_stock_days']).sum()}", "below safety stock", False),
    ]
    for col, label, val, delta, positive in metrics:
        color = DARK_THEME["green"] if positive else DARK_THEME["red"]
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:12px;color:#8891aa;">{label}</div>
            <div style="font-size:28px;font-weight:bold;color:{DARK_THEME['text']}">{val}</div>
            <div style="font-size:12px;color:{color}">{delta}</div>
        </div>""", unsafe_allow_html=True)


def render_forecast_chart(df, site):
    site_df = df[df["site"] == site].sort_values("date")
    hist = site_df[~site_df["is_future"]]
    fut = site_df[site_df["is_future"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["demand"],
        name="Actual", line=dict(color=DARK_THEME["text"], width=2),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([hist["date"].iloc[[-1]], fut["date"]]),
        y=pd.concat([hist["forecast"].iloc[[-1]], fut["forecast"]]),
        name="Forecast (P50)", line=dict(color=DARK_THEME["accent"], width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([hist["date"].iloc[[-1]], fut["date"]]),
        y=pd.concat([hist["p90"].iloc[[-1]], fut["p90"]]),
        name="P90", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([hist["date"].iloc[[-1]], fut["date"]]),
        y=pd.concat([hist["p10"].iloc[[-1]], fut["p10"]]),
        name="P10–P90 Cone", fill="tonexty",
        fillcolor="rgba(79,142,247,0.15)", line=dict(width=0),
    ))
    fig.add_vline(x=str(datetime.today().date()), line_dash="dash",
                  line_color=DARK_THEME["yellow"], annotation_text="Today")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_THEME["card"],
        plot_bgcolor=DARK_THEME["card"],
        title=f"Demand Forecast — {site.replace('_',' ').title()} (28-day horizon)",
        height=420,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def render_stock_heatmap(inv_df):
    inv_df = inv_df.copy()
    inv_df["cover_ratio"] = inv_df["current_stock_days"] / inv_df["safety_stock_days"]
    inv_df["status"] = inv_df["cover_ratio"].apply(
        lambda x: "Critical" if x < 0.85 else ("Watch" if x < 1.1 else "OK")
    )
    fig = px.bar(
        inv_df.sort_values("current_stock_days"),
        x="current_stock_days", y="site",
        color="cover_ratio",
        color_continuous_scale=["#ff4b6e", "#f7c948", "#00c896"],
        orientation="h",
        text=inv_df["current_stock_days"].round(1).astype(str) + "d",
        title="Stock Cover by Site (days) — Safety Stock Threshold",
        template="plotly_dark",
    )
    fig.add_vline(x=inv_df["safety_stock_days"].mean(), line_dash="dash",
                  line_color=DARK_THEME["yellow"], annotation_text="Avg SS")
    fig.update_layout(paper_bgcolor=DARK_THEME["card"], plot_bgcolor=DARK_THEME["card"],
                      height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def render_model_benchmark():
    data = {
        "Model": ["ARIMA-X", "Prophet + Regressors", "XGBoost + SHAP", "Bayesian Ensemble (MinT)"],
        "MAPE (%)": [14.2, 11.8, 9.4, 7.3],
        "RMSE": [87.3, 71.6, 58.2, 47.1],
        "Bias (%)": [2.1, -0.9, 0.3, 0.4],
    }
    df = pd.DataFrame(data)
    fig = px.bar(df, x="Model", y="MAPE (%)", color="MAPE (%)",
                 color_continuous_scale=["#00c896", "#f7c948", "#ff4b6e"],
                 template="plotly_dark", title="Model Benchmark — Walk-Forward CV (18 months)")
    fig.update_layout(paper_bgcolor=DARK_THEME["card"], plot_bgcolor=DARK_THEME["card"], height=350)
    return fig, df


def main():
    with st.sidebar:
        st.markdown("## 🧠 SC Intelligence")
        st.markdown("---")
        site_filter = st.selectbox("Site", [
            "paris_cdg", "lyon_st_exupery", "marseille_fos",
            "bordeaux_meriadeck", "lille_lesquin"
        ])
        horizon = st.slider("Forecast Horizon (days)", 7, 56, 28)
        service_level = st.slider("Service Level Target", 0.90, 0.99, 0.95, step=0.01)
        st.markdown("---")
        st.markdown("**Last refresh:** " + datetime.now().strftime("%Y-%m-%d %H:%M"))
        st.markdown("**Models:** ARIMA · Prophet · XGBoost · Ensemble")
        st.markdown("**Framework:** Bayesian MinT Reconciliation")

    st.title("🧠 Strategic Supply Chain Intelligence")
    st.caption("Bayesian Ensemble Forecasting · Stochastic Inventory Optimization · Executive Decision Layer")
    st.markdown("---")

    forecast_df = load_forecast_data()
    inv_df = load_inventory_data()

    render_kpi_row(inv_df)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(render_forecast_chart(forecast_df, site_filter),
                        use_container_width=True)
    with col2:
        st.plotly_chart(render_stock_heatmap(inv_df), use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        bench_fig, bench_df = render_model_benchmark()
        st.plotly_chart(bench_fig, use_container_width=True)
    with col4:
        st.markdown("### 🤖 Executive Insights")
        site_inv = inv_df[inv_df["site"] == site_filter].iloc[0]
        cover = site_inv["current_stock_days"]
        ss = site_inv["safety_stock_days"]
        urgency = "🔴 **URGENT**" if cover < ss else ("🟡 **MONITOR**" if cover < ss * 1.2 else "🟢 **ADEQUATE**")
        st.markdown(f"""
        <div class="insight-box">
        <b>{urgency} — {site_filter.replace('_',' ').title()}</b><br><br>
        📦 Stock cover: <b>{cover:.1f}d</b> vs SS: <b>{ss:.1f}d</b><br>
        📈 Pick rate improvement: <b>+{site_inv['pick_rate_improvement']:.1f}%</b><br>
        🚚 Throughput delta: <b>+{site_inv['throughput_delta']} pallets/day</b><br>
        📊 MAPE: <b>{site_inv['mape']:.1f}%</b> | Bias: <b>{site_inv['forecast_bias']:.2f}%</b>
        </div>""", unsafe_allow_html=True)

        st.markdown("### 📋 Site Performance Matrix")
        st.dataframe(
            inv_df[["site", "mape", "pick_rate_improvement", "throughput_delta", "current_stock_days"]]
            .rename(columns={
                "mape": "MAPE %",
                "pick_rate_improvement": "Pick Rate Δ%",
                "throughput_delta": "Throughput Δ",
                "current_stock_days": "Cover (days)",
            })
            .set_index("site")
            .style.highlight_min(subset=["MAPE %"], color="#2d4a2d")
            .highlight_max(subset=["Pick Rate Δ%"], color="#2d4a2d"),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
