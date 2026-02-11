"""Demand model for computing true demand and AI forecasts."""

import numpy as np
from scipy import stats
from typing import Dict, Optional
from .products import ProductConfig


class DemandModel:
    """Model for computing demand, forecasts, and optimal orders."""

    def __init__(self, product: ProductConfig, seed: Optional[int] = None):
        self.product = product
        self.rng = np.random.default_rng(seed)

    def compute_true_demand(
        self,
        visible_features: Dict[str, float],
        hidden_features: Dict[str, float],
        add_noise: bool = True,
    ) -> float:
        """
        Compute true demand from all features.

        Args:
            visible_features: Features the AI can see (e.g., temperature)
            hidden_features: Features the AI cannot see (e.g., events)
            add_noise: Whether to add random noise

        Returns:
            True demand value
        """
        base = self.product.base_level
        visible_effect = self.product.compute_visible_effect(visible_features)
        hidden_effect = self.product.compute_hidden_effect(hidden_features)

        expected_demand = base + visible_effect + hidden_effect

        if add_noise:
            noise = self.rng.normal(0, self.product.noise_std)
            return max(0, expected_demand + noise)

        return max(0, expected_demand)

    def compute_ai_forecast(self, visible_features: Dict[str, float]) -> float:
        """
        Compute AI forecast from visible features only.

        The AI doesn't know about hidden features, so its forecast
        is based only on what it can observe.

        Args:
            visible_features: Features the AI can see

        Returns:
            AI's demand forecast
        """
        base = self.product.base_level
        visible_effect = self.product.compute_visible_effect(visible_features)

        return max(0, base + visible_effect)

    def compute_optimal_order(
        self, forecast: float, demand_std: Optional[float] = None
    ) -> float:
        """
        Compute newsvendor optimal order quantity.

        Uses the critical ratio formula:
        Optimal = forecast + z * std

        where z is the z-score for the critical ratio percentile.

        Args:
            forecast: Expected demand
            demand_std: Standard deviation (defaults to product noise_std)

        Returns:
            Optimal order quantity
        """
        if demand_std is None:
            demand_std = self.product.noise_std

        critical_ratio = self.product.critical_ratio
        z_score = stats.norm.ppf(critical_ratio)

        optimal = forecast + z_score * demand_std
        return max(0, round(optimal))

    def compute_profit(self, order: float, actual_demand: float) -> float:
        """
        Compute profit for an order given actual demand.

        Args:
            order: Order quantity
            actual_demand: Realized demand

        Returns:
            Profit in currency units
        """
        sold = min(order, actual_demand)
        unsold = max(0, order - actual_demand)
        lost_sales = max(0, actual_demand - order)

        revenue = sold * self.product.price
        cost = order * self.product.cost
        salvage = unsold * self.product.salvage

        profit = revenue - cost + salvage

        return profit

    def compute_optimal_profit(self, actual_demand: float) -> float:
        """
        Compute profit if optimal order was placed (with hindsight).

        This is the theoretical maximum profit given perfect information.

        Args:
            actual_demand: Realized demand

        Returns:
            Maximum possible profit
        """
        # With perfect information, order exactly the demand
        return self.compute_profit(actual_demand, actual_demand)
