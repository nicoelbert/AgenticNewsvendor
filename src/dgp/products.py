"""Product configurations for the newsvendor experiment."""

from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path
import yaml


@dataclass
class ProductConfig:
    """Configuration for a single product category."""

    name: str
    display_name: str
    base_level: float
    price: float
    cost: float
    salvage: float
    profit_per_unit: float
    loss_per_unit: float
    noise_std: float
    visible_betas: Dict[str, float] = field(default_factory=dict)
    hidden_betas: Dict[str, float] = field(default_factory=dict)

    @property
    def critical_ratio(self) -> float:
        """Compute newsvendor critical ratio Cu / (Cu + Co)."""
        cu = self.profit_per_unit  # Underage cost (lost profit)
        co = self.loss_per_unit  # Overage cost (loss on unsold)
        return cu / (cu + co)

    def compute_visible_effect(self, features: Dict[str, float]) -> float:
        """Compute demand contribution from visible features."""
        effect = 0.0
        for feature, value in features.items():
            if feature in self.visible_betas:
                effect += self.visible_betas[feature] * value
        return effect

    def compute_hidden_effect(self, features: Dict[str, float]) -> float:
        """Compute demand contribution from hidden features."""
        effect = 0.0
        for feature, value in features.items():
            if feature in self.hidden_betas:
                effect += self.hidden_betas[feature] * value
        return effect


def load_product_configs(yaml_path: Path) -> Dict[str, ProductConfig]:
    """Load product configurations from scenario backlog YAML."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    products = {}
    for name, config in data.get("products", {}).items():
        products[name] = ProductConfig(
            name=name,
            display_name=config["display_name"],
            base_level=config["base_level"],
            price=config["price"],
            cost=config["cost"],
            salvage=config["salvage"],
            profit_per_unit=config["profit_per_unit"],
            loss_per_unit=config["loss_per_unit"],
            noise_std=config["noise_std"],
            visible_betas=config.get("visible_betas", {}),
            hidden_betas=config.get("hidden_betas", {}),
        )

    return products
