import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson, norm
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SITE_PROFILES = {
    "paris_cdg": {"base": 420, "amp": 0.28, "trend": 0.0012, "promo_rate": 0.08, "cv": 0.18},
    "lyon_st_exupery": {"base": 310, "amp": 0.34, "trend": 0.0008, "promo_rate": 0.06, "cv": 0.22},
    "marseille_fos": {"base": 275, "amp": 0.41, "trend": 0.0005, "promo_rate": 0.07, "cv": 0.31},
    "bordeaux_meriadeck": {"base": 190, "amp": 0.22, "trend": 0.0015, "promo_rate": 0.05, "cv": 0.15},
    "lille_lesquin": {"base": 230, "amp": 0.30, "trend": 0.0009, "promo_rate": 0.09, "cv": 0.20},
}

SKU_CLASSES = {
    "A": {"velocity": "fast", "weight": 0.20, "demand_multiplier": 3.2, "dispatch": nbinom},
    "B": {"velocity": "medium", "weight": 0.35, "demand_multiplier": 1.0, "dispatch": poisson},
    "C": {"velocity": "slow", "weight": 0.45, "demand_multiplier": 0.3, "dispatch": poisson},
}

FRENCH_HOLIDAYS = [
    "2023-01-01", "2023-04-10", "2023-05-01", "2023-05-08",
    "2023-05-18", "2023-05-29", "2023-07-14", "2023-08-15",
    "2023-11-01", "2023-11-11", "2023-12-25",
    "2024-01-01", "2024-04-01", "2024-05-01", "2024-05-08",
    "2024-05-09", "2024-05-20", "2024-07-14", "2024-08-15",
    "2024-11-01", "2024-11-11", "2024-12-25",
]


def _build_fourier_terms(t: np.ndarray, period: float, n_harmonics: int) -> np.ndarray:
    terms = []
    for k in range(1, n_harmonics + 1):
        terms.append(np.sin(2 * np.pi * k * t / period))
        terms.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(terms)


def _generate_sku_demand(dates, site_cfg, sku_class, sku_id, rng):
    n = len(dates)
    t = np.arange(n)

    fourier = _build_fourier_terms(t, period=365.25, n_harmonics=3)
    seasonal_weights = rng.normal(0, site_cfg["amp"] / 3, fourier.shape[1])
    seasonality = fourier @ seasonal_weights

    trend = site_cfg["trend"] * t
    noise_level = site_cfg["cv"] * site_cfg["base"]
    noise = rng.normal(0, noise_level, n)

    mu = site_cfg["base"] * SKU_CLASSES[sku_class]["demand_multiplier"]
    holiday_dates = pd.to_datetime(FRENCH_HOLIDAYS)
    holiday_mask = pd.Series(dates).isin(holiday_dates).values
    holiday_effect = holiday_mask * rng.uniform(0.15, 0.45) * mu

    promo_mask = rng.binomial(1, site_cfg["promo_rate"], n).astype(bool)
    promo_lift = promo_mask * rng.uniform(0.20, 0.60) * mu

    dow_effect = np.array([0.05, 0.03, 0.0, 0.02, 0.08, -0.12, -0.18])
    dow = np.array([pd.Timestamp(d).dayofweek for d in dates])
    day_of_week_effect = mu * dow_effect[dow]

    raw_demand = mu + trend + seasonality * mu + noise + holiday_effect + promo_lift + day_of_week_effect
    raw_demand = np.clip(raw_demand, 0, None)

    if SKU_CLASSES[sku_class]["dispatch"] == nbinom:
        r = mu / (site_cfg["cv"] ** 2 * mu - 1 + 1e-6)
        r = max(r, 0.5)
        p = r / (r + raw_demand + 1e-6)
        demand = rng.negative_binomial(max(int(r), 1), np.clip(p, 0.01, 0.99), n)
    else:
        demand = rng.poisson(np.clip(raw_demand, 0.1, None))

    return pd.DataFrame({
        "date": dates,
        "site": [sku_id.split("_")[0]] * n,
        "sku_id": [sku_id] * n,
        "sku_class": [sku_class] * n,
        "demand": demand,
        "promo_flag": promo_mask.astype(int),
        "holiday_flag": holiday_mask.astype(int),
    })


def generate_dataset(n_sites: int = 5, n_skus: int = 500, n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    logger.info(f"Generating dataset: {n_sites} sites × {n_skus} SKUs × {n_days} days")

    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    site_names = list(SITE_PROFILES.keys())[:n_sites]

    sku_classes = rng.choice(["A", "B", "C"], size=n_skus, p=[0.20, 0.35, 0.45])
    sku_ids_base = [f"SKU_{str(i).zfill(4)}" for i in range(n_skus)]

    all_frames = []
    total = n_sites * n_skus
    count = 0

    for site in site_names:
        cfg = SITE_PROFILES[site]
        for sku_cls, sku_base in zip(sku_classes, sku_ids_base):
            sku_id = f"{site}_{sku_base}"
            df = _generate_sku_demand(dates, cfg, sku_cls, sku_id, rng)
            all_frames.append(df)
            count += 1
            if count % 500 == 0:
                logger.info(f"  Progress: {count}/{total}")

    result = pd.concat(all_frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["site", "sku_id", "date"]).reset_index(drop=True)
    logger.info(f"Dataset shape: {result.shape}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Synthetic supply chain demand generator")
    parser.add_argument("--sites", type=int, default=5)
    parser.add_argument("--skus", type=int, default=500)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/sample_dataset.csv")
    args = parser.parse_args()

    df = generate_dataset(n_sites=args.sites, n_skus=args.skus, n_days=args.days, seed=args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
