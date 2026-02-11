"""Scenario loading and management."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import random


@dataclass
class Scenario:
    """A single experiment scenario."""

    id: str
    product: str
    product_display_name: str
    date: str
    scenario_type: str
    difficulty: str
    narrative: str
    demand_history: List[float]

    # Features
    visible_features: Dict[str, Any]
    hidden_features: Dict[str, Any]
    noise_items: List[str]

    # Pre-computed values
    ai_forecast: float
    ai_recommendation: float
    true_expected_demand: float
    optimal_order: float
    actual_demand: float

    # Cost structure
    price: float
    cost: float
    salvage: float
    profit_per_unit: float
    loss_per_unit: float

    # AI responses
    ai_responses: Dict[str, str] = field(default_factory=dict)

    # Full config for agent responder
    full_config: Dict = field(default_factory=dict)


class ScenarioLoader:
    """Loads and manages scenarios from YAML configuration."""

    def __init__(self, yaml_path: Path):
        """
        Initialize loader with path to scenario backlog.

        Args:
            yaml_path: Path to scenario_backlog.yaml
        """
        self.yaml_path = yaml_path
        self._data = None
        self._scenarios = None
        self._products = None

    @property
    def data(self) -> Dict:
        """Lazy load YAML data."""
        if self._data is None:
            with open(self.yaml_path, "r") as f:
                self._data = yaml.safe_load(f)
        return self._data

    @property
    def products(self) -> Dict:
        """Get product configurations."""
        if self._products is None:
            self._products = self.data.get("products", {})
        return self._products

    def load_scenario(self, scenario_id: str) -> Scenario:
        """
        Load a single scenario by ID.

        Args:
            scenario_id: Scenario ID (e.g., "S01")

        Returns:
            Scenario object
        """
        scenario_data = self.data.get("scenarios", {}).get(scenario_id)
        if scenario_data is None:
            raise ValueError(f"Scenario {scenario_id} not found")

        # Get product config
        product_name = scenario_data["product"]
        product_config = self.products.get(product_name, {})

        return Scenario(
            id=scenario_data["id"],
            product=product_name,
            product_display_name=product_config.get("display_name", product_name),
            date=scenario_data["date"],
            scenario_type=scenario_data["scenario_type"],
            difficulty=scenario_data["difficulty"],
            narrative=scenario_data["narrative"].strip(),
            demand_history=scenario_data["demand_history"],
            visible_features=scenario_data["features"]["visible"],
            hidden_features=scenario_data["features"]["hidden"],
            noise_items=scenario_data["features"].get("noise", []),
            ai_forecast=scenario_data["computed"]["ai_forecast"],
            ai_recommendation=scenario_data["computed"]["ai_recommendation"],
            true_expected_demand=scenario_data["computed"]["true_expected_demand"],
            optimal_order=scenario_data["computed"]["optimal_order"],
            actual_demand=scenario_data["computed"]["actual_demand"],
            price=product_config.get("price", 0),
            cost=product_config.get("cost", 0),
            salvage=product_config.get("salvage", 0),
            profit_per_unit=product_config.get("profit_per_unit", 0),
            loss_per_unit=product_config.get("loss_per_unit", 0),
            ai_responses=scenario_data.get("ai_responses", {}),
            full_config=scenario_data,
        )

    def load_all_scenarios(self) -> List[Scenario]:
        """Load all scenarios."""
        if self._scenarios is None:
            self._scenarios = []
            for scenario_id in self.data.get("scenarios", {}).keys():
                self._scenarios.append(self.load_scenario(scenario_id))
        return self._scenarios

    def get_scenario_ids(self) -> List[str]:
        """Get list of all scenario IDs."""
        return list(self.data.get("scenarios", {}).keys())

    def get_randomized_order(self, seed: Optional[int] = None) -> List[str]:
        """
        Get randomized scenario order.

        Args:
            seed: Random seed for reproducibility

        Returns:
            List of scenario IDs in random order
        """
        ids = self.get_scenario_ids()
        if seed is not None:
            random.seed(seed)
        random.shuffle(ids)
        return ids

    def get_scenarios_by_type(self, scenario_type: str) -> List[Scenario]:
        """
        Get scenarios of a specific type.

        Args:
            scenario_type: "trust_ai" or "override_up" or "override_down"

        Returns:
            List of matching scenarios
        """
        return [s for s in self.load_all_scenarios() if s.scenario_type == scenario_type]
