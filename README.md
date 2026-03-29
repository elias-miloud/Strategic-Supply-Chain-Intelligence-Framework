# Strategic Supply Chain Intelligence Framework

> Advanced analytics and decision-support framework for executive-level supply chain performance management.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Status](https://img.shields.io/badge/Status-Active-green) ![Domain](https://img.shields.io/badge/Domain-Supply%20Chain%20Strategy-darkblue)

---

## Overview

This repository provides a quantitative toolkit designed for strategic decision-making in complex supply chain environments. It combines time-series forecasting, scenario modelling, and performance analytics to support C-level and CODIR-level operational planning.

Built from real-world experience across large-scale logistics operations (WMS deployment, CAPEX analysis) and quantitative planning in corporate investment banking environments.

---

## Key Capabilities

| Module | Description | Methods |
|--------|-------------|---------|
| `Demand Intelligence` | Multi-model forecasting engine with ensemble voting | ARIMA, Prophet, XGBoost |
| `Scenario Modelling` | Executive-level scenario simulation & stress testing | Monte Carlo, Sensitivity Analysis |
| `Performance Analytics` | KPI tracking, variance analysis & executive reporting | SPC, Control Charts |
| `Decision Support` | Interactive dashboard for CODIR-level presentations | Plotly, Streamlit |

---

## Technical Stack

- **Language:** Python 3.11
- **Forecasting:** `statsmodels`, `prophet`, `scikit-learn`, `xgboost`
- **Visualisation:** `plotly`, `streamlit`, `matplotlib`
- **Data Engineering:** `pandas`, `numpy`
- **Simulation & Modelling:** `scipy`, `simpy`

---

## Business Context

Designed to replicate and scale analytical frameworks applied in real operational environments:

- **WMS Deployment & Slotting Optimization** — quantitative pick rate analysis leading to +17% productivity and +30 pallets/day throughput
- **Demand Planning Accuracy** — time series modelling improving forecast precision by +12% (Société Générale CIB)
- **Executive Reporting** — decision-support frameworks escalated to top management across multi-site logistics operations

---

## Repository Structure

```
strategic-supply-chain-intelligence/
│
├── data/
│   ├── simulate_demand.py          # Synthetic supply chain dataset generator
│   └── sample_dataset.csv          # Pre-generated sample data
│
├── models/
│   ├── arima_model.py              # ARIMA forecasting with auto-tuning
│   ├── prophet_model.py            # Facebook Prophet with regressors
│   ├── xgboost_model.py            # XGBoost with feature engineering
│   └── ensemble.py                 # Ensemble voting & MAPE/RMSE comparison
│
├── dashboard/
│   └── app.py                      # Streamlit executive dashboard
│
├── notebooks/
│   └── strategic_analysis.ipynb    # Full analysis walkthrough
│
└── reports/
    └── executive_summary.py        # Auto-generated PDF executive report
```

---

## Quickstart

```bash
git clone https://github.com/elias-miloud/strategic-supply-chain-intelligence
cd strategic-supply-chain-intelligence
pip install -r requirements.txt

# Run the executive dashboard
streamlit run dashboard/app.py

# Run full forecasting pipeline
python models/ensemble.py
```

---

## Results Preview

```
Model Comparison (MAPE):
  ARIMA        →  8.3%
  Prophet      →  6.7%
  XGBoost      →  5.9%
  Ensemble     →  5.1%  ← Best performer

Forecast Horizon: 12 weeks
Confidence Interval: 95%
```

---

## Author

**Elias Miloud**  
Supply Chain Project Manager | Groupe Lactalis  
M2 Supply Chain Strategy — Université Paris-Panthéon-Assas  
[LinkedIn](https://linkedin.com/in/eliasmiloud) · Paris, France
