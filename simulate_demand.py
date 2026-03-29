"""
Strategic Supply Chain Intelligence Framework
Data Simulation Module

Generates realistic supply chain demand data with:
- Seasonality patterns
- Trend components
- Noise & outliers
- External regressors (promotions, events)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_supply_chain_dataset(
    n_weeks: int = 104,
    n_skus: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic weekly demand data for supply chain forecasting.

    Args:
        n_weeks: Number of weeks to simulate
        n_skus: Number of SKUs to simulate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: date, sku_id, demand, price, promotion, stock_level
    """
    np.random.seed(seed)
    records = []

    start_date = datetime(2022, 1, 3)

    for sku in range(1, n_skus + 1):
        base_demand = np.random.randint(100, 500)
        trend = np.random.uniform(-0.5, 1.5)
        seasonality_amplitude = np.random.uniform(0.1, 0.4)

        for week in range(n_weeks):
            date = start_date + timedelta(weeks=week)

            # Trend component
            trend_component = base_demand + trend * week

            # Seasonal component (annual cycle)
            seasonal_component = seasonality_amplitude * base_demand * np.sin(
                2 * np.pi * week / 52
            )

            # Noise
            noise = np.random.normal(0, base_demand * 0.05)

            # Promotion effect (random weeks)
            promotion = int(np.random.random() < 0.1)
            promo_effect = promotion * base_demand * np.random.uniform(0.2, 0.5)

            demand = max(0, trend_component + seasonal_component + noise + promo_effect)

            records.append({
                "date": date,
                "sku_id": f"SKU_{sku:03d}",
                "demand": round(demand),
                "price": round(np.random.uniform(5, 50), 2),
                "promotion": promotion,
                "stock_level": round(demand * np.random.uniform(1.0, 2.5)),
                "lead_time_days": np.random.randint(3, 15)
            })

    df = pd.DataFrame(records)
    df.to_csv("data/sample_dataset.csv", index=False)
    print(f"Dataset generated: {len(df)} records | {n_skus} SKUs | {n_weeks} weeks")
    return df


if __name__ == "__main__":
    df = generate_supply_chain_dataset()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
