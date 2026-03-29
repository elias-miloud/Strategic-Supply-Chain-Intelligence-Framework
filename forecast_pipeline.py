import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ScenarioEngine:

    SCENARIO_LABELS = {
        "bear": {"quantile": 0.10, "label": "Pessimistic (P10)", "color": "#e74c3c"},
        "base": {"quantile": 0.50, "label": "Base Case (P50)", "color": "#3498db"},
        "bull": {"quantile": 0.90, "label": "Optimistic (P90)", "color": "#2ecc71"},
    }

    def __init__(self, n_simulations: int = 5_000, horizon: int = 28, seed: int = 42):
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

    def _fit_residual_distribution(self, residuals: np.ndarray) -> Dict:
        mu = float(np.mean(residuals))
        sigma = float(np.std(residuals))
        df, loc, scale = student_t.fit(residuals)
        ks_stat = float(np.max(np.abs(
            np.sort(residuals) / (sigma + 1e-8) - norm.cdf(np.sort(residuals), mu / sigma)
        )))
        return {"mu": mu, "sigma": sigma, "t_df": df, "t_loc": loc, "t_scale": scale, "ks_stat": ks_stat}

    def generate_demand_scenarios(self, point_forecast: np.ndarray,
                                   historical_residuals: np.ndarray) -> Dict:
        dist = self._fit_residual_distribution(historical_residuals)
        sim_residuals = student_t.rvs(
            df=dist["t_df"], loc=dist["t_loc"], scale=dist["t_scale"],
            size=(self.n_simulations, self.horizon),
            random_state=self.rng.integers(0, 2**31),
        )
        simulated_paths = np.clip(point_forecast[np.newaxis, :] + sim_residuals, 0, None)

        scenarios = {}
        for name, cfg in self.SCENARIO_LABELS.items():
            q = cfg["quantile"]
            path = np.percentile(simulated_paths, q * 100, axis=0)
            cumulative = np.cumsum(path)
            scenarios[name] = {
                "label": cfg["label"],
                "color": cfg["color"],
                "daily": path,
                "cumulative": cumulative,
                "total_28d": float(cumulative[-1]),
            }

        scenarios["fan"] = {
            "p5": np.percentile(simulated_paths, 5, axis=0),
            "p25": np.percentile(simulated_paths, 25, axis=0),
            "p75": np.percentile(simulated_paths, 75, axis=0),
            "p95": np.percentile(simulated_paths, 95, axis=0),
        }
        return scenarios

    def compute_pnl_impact(self, scenarios: Dict, unit_margin: float,
                            unit_cost: float, current_stock: float) -> Dict:
        results = {}
        for name in ["bear", "base", "bull"]:
            total_demand = scenarios[name]["total_28d"]
            revenue_demand = min(total_demand, current_stock)
            lost_sales = max(total_demand - current_stock, 0)
            surplus = max(current_stock - total_demand, 0)
            results[name] = {
                "scenario": scenarios[name]["label"],
                "total_demand_units": total_demand,
                "units_sold": revenue_demand,
                "lost_sales_units": lost_sales,
                "surplus_units": surplus,
                "gross_margin": revenue_demand * unit_margin,
                "lost_sales_cost": lost_sales * unit_margin,
                "holding_cost": surplus * unit_cost * 0.25 / 365 * 28,
                "net_impact": revenue_demand * unit_margin - lost_sales * unit_margin * 0.5,
            }

        base_net = results["base"]["net_impact"]
        for name in results:
            results[name]["delta_vs_base"] = results[name]["net_impact"] - base_net

        return results

    def generate_executive_summary(self, sku_id: str, site: str,
                                    scenarios: Dict, pnl: Dict,
                                    current_stock: float) -> str:
        base = scenarios["base"]
        bear = scenarios["bear"]
        bull = scenarios["bull"]

        demand_range = f"{bear['total_28d']:,.0f}–{bull['total_28d']:,.0f} units"
        stock_cover = current_stock / (base["total_28d"] / 28 + 1e-6)
        urgency = "🔴 URGENT" if stock_cover < 7 else ("🟡 MONITOR" if stock_cover < 14 else "🟢 ADEQUATE")

        pnl_risk = pnl["bear"]["lost_sales_cost"]
        pnl_upside = pnl["bull"]["gross_margin"] - pnl["base"]["gross_margin"]

        summary = (
            f"[{urgency}] {site} / {sku_id}\n"
            f"28-day demand range: {demand_range} (P10–P90)\n"
            f"Current stock cover: {stock_cover:.1f} days\n"
            f"Downside risk (P10): −€{pnl_risk:,.0f} in lost sales\n"
            f"Upside opportunity (P90): +€{pnl_upside:,.0f} gross margin\n"
            f"Recommended action: {'Expedite replenishment' if stock_cover < 7 else 'Standard replenishment cycle'}"
        )
        return summary
