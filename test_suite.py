import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from scipy.optimize import minimize_scalar
import logging
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StochasticSafetyStockOptimizer:

    def __init__(self, service_level: float = 0.95, n_scenarios: int = 10_000,
                 holding_cost_rate: float = 0.25, stockout_cost_multiplier: float = 3.0):
        self.service_level = service_level
        self.n_scenarios = n_scenarios
        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_multiplier = stockout_cost_multiplier
        self.rng = np.random.default_rng(42)

    def _simulate_lead_time_demand(self, mu_demand: float, sigma_demand: float,
                                    mu_lead_time: float, sigma_lead_time: float) -> np.ndarray:
        lead_times = self.rng.lognormal(
            mean=np.log(mu_lead_time) - 0.5 * (sigma_lead_time / mu_lead_time) ** 2,
            sigma=sigma_lead_time / mu_lead_time,
            size=self.n_scenarios,
        )
        lead_times = np.clip(np.round(lead_times).astype(int), 1, 30)

        ltd_samples = np.array([
            self.rng.negative_binomial(
                max(int(mu_demand * lt / max((sigma_demand / mu_demand) ** 2 * mu_demand - 1, 0.5)), 1),
                np.clip(mu_demand * lt / (mu_demand * lt + max((sigma_demand / mu_demand) ** 2 * mu_demand - 1, 0.5) * mu_demand * lt), 0.01, 0.99)
            )
            for lt in lead_times
        ])
        return ltd_samples

    def analytical_safety_stock(self, mu_demand: float, sigma_demand: float,
                                  mu_lead_time: float, sigma_lead_time: float) -> Dict:
        z = norm.ppf(self.service_level)
        ss_demand_var = z * sigma_demand * np.sqrt(mu_lead_time)
        ss_lt_var = z * mu_demand * sigma_lead_time
        ss_combined = z * np.sqrt(mu_lead_time * sigma_demand ** 2 + mu_demand ** 2 * sigma_lead_time ** 2)

        return {
            "z_score": z,
            "ss_demand_uncertainty_only": ss_demand_var,
            "ss_leadtime_uncertainty_only": ss_lt_var,
            "ss_combined": ss_combined,
            "rop": mu_demand * mu_lead_time + ss_combined,
        }

    def saa_safety_stock(self, mu_demand: float, sigma_demand: float,
                          mu_lead_time: float, sigma_lead_time: float,
                          unit_cost: float = 10.0) -> Dict:
        ltd_samples = self._simulate_lead_time_demand(
            mu_demand, sigma_demand, mu_lead_time, sigma_lead_time
        )

        holding_cost = self.holding_cost_rate * unit_cost
        stockout_cost = self.stockout_cost_multiplier * unit_cost

        def total_cost(ss):
            ss = max(ss, 0)
            holding = holding_cost * ss
            shortfalls = np.maximum(ltd_samples - ss - mu_demand * mu_lead_time, 0)
            expected_stockout = stockout_cost * shortfalls.mean()
            return holding + expected_stockout

        result = minimize_scalar(total_cost, bounds=(0, ltd_samples.max()), method="bounded")
        optimal_ss = max(result.x, 0)

        empirical_service_level = float(np.mean(ltd_samples <= mu_demand * mu_lead_time + optimal_ss))
        p10, p50, p90 = np.percentile(ltd_samples, [10, 50, 90])

        return {
            "optimal_safety_stock": optimal_ss,
            "reorder_point": mu_demand * mu_lead_time + optimal_ss,
            "empirical_service_level": empirical_service_level,
            "expected_total_cost": result.fun,
            "ltd_p10": p10,
            "ltd_p50": p50,
            "ltd_p90": p90,
            "ltd_mean": ltd_samples.mean(),
            "ltd_std": ltd_samples.std(),
        }

    def compute_portfolio(self, sku_params: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"sku_id", "mu_demand", "sigma_demand", "mu_lead_time",
                         "sigma_lead_time", "unit_cost"}
        missing = required_cols - set(sku_params.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        results = []
        for _, row in sku_params.iterrows():
            analytical = self.analytical_safety_stock(
                row["mu_demand"], row["sigma_demand"],
                row["mu_lead_time"], row["sigma_lead_time"],
            )
            saa = self.saa_safety_stock(
                row["mu_demand"], row["sigma_demand"],
                row["mu_lead_time"], row["sigma_lead_time"],
                unit_cost=row["unit_cost"],
            )
            results.append({
                "sku_id": row["sku_id"],
                "analytical_ss": analytical["ss_combined"],
                "optimal_ss": saa["optimal_safety_stock"],
                "rop": saa["reorder_point"],
                "service_level_achieved": saa["empirical_service_level"],
                "total_cost": saa["expected_total_cost"],
                "capital_tied_up": saa["optimal_safety_stock"] * row["unit_cost"],
            })

        df = pd.DataFrame(results)
        total_capital = df["capital_tied_up"].sum()
        logger.info(f"Portfolio SS computed: {len(df)} SKUs, total capital={total_capital:,.0f}€")
        return df
