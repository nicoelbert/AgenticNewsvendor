"""
Experiment scaffolding for controlled product/model simulations.

This module generates synthetic demand, trains baseline models, and
produces evaluation windows ready for the Streamlit dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

from . import features as feat
from . import simulator


@dataclass
class HorizonConfig:
    training_window: int
    history_days: int
    forecast_days: int

    @property
    def total_days(self) -> int:
        return self.training_window + self.history_days + self.forecast_days


@dataclass
class ProductConfig:
    visible_features: List[str]
    hidden_features: List[str]
    noise_features: List[str]
    betas: Mapping[str, float]
    base_level: float = 50.0
    trend_slope: float = 0.0
    season_strength: float = 8.0
    epsilon_std: float = 4.0
    purchase_price: float = 3.0
    selling_price: float = 6.0
    salvage_value: float = 0.5


@dataclass
class ModelConfig:
    features: List[str]


@dataclass
class ExperimentConfig:
    horizon: HorizonConfig
    products: Dict[str, ProductConfig]
    models: Dict[str, ModelConfig]
    assignments: List[Tuple[str, str]]
    start_date: str
    seed: int
    output_dir: Path


def _build_feature_pool(total_days: int, start_date: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10_000, size=10)

    dates = pd.date_range(start=start_date, periods=total_days, freq="D")
    df = pd.DataFrame(index=dates)
    df["price"] = feat.price_series(total_days, seed=seeds[0]).to_numpy()
    df["price_step"] = feat.price_step_function(total_days, seed=seeds[1]).to_numpy()
    df["promo"] = feat.promo_series(total_days, seed=seeds[2]).to_numpy()
    df["promo_step"] = feat.promotion_step_function(total_days, seed=seeds[3]).to_numpy()
    df["temperature"] = feat.temperature_series(total_days, seed=seeds[4]).to_numpy()
    df["rain"] = feat.rain_series(total_days, seed=seeds[5]).to_numpy()
    df["holiday"] = feat.binary_holiday(total_days, seed=seeds[6]).to_numpy()
    df["market_index"] = feat.market_index_series(total_days, seed=seeds[7]).to_numpy()
    df["odd_cycle"] = feat.odd_cycle_feature(total_days, seed=seeds[8]).to_numpy()
    df["weekday"] = feat.weekday_indicator(total_days).to_numpy()
    df["sports_event"] = feat.sports_event_flag(total_days, seed=seeds[9]).to_numpy()
    return df


def _simulate_product_demand(
    feature_pool: pd.DataFrame,
    product_name: str,
    cfg: ProductConfig,
) -> pd.DataFrame:
    explanatory = list(cfg.visible_features) + list(cfg.hidden_features)
    demand = simulator.generate_demand(
        feature_pool,
        explanatory_vars=explanatory,
        betas=cfg.betas,
        base_level=cfg.base_level,
        trend_slope=cfg.trend_slope,
        season_strength=cfg.season_strength,
        epsilon_std=cfg.epsilon_std,
    )
    df = feature_pool.copy()
    df["demand"] = demand.values
    df["product"] = product_name
    return df


def _train_and_predict(
    df: pd.DataFrame,
    model_cfg: ModelConfig,
    horizon: HorizonConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    total_days = horizon.total_days
    train_end = horizon.training_window
    eval_start = train_end

    features = model_cfg.features
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features for model: {missing}")

    X_train = df.iloc[:train_end][features]
    y_train = df.iloc[:train_end]["demand"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    eval_slice = slice(eval_start, total_days)
    X_eval = df.iloc[eval_slice][features]
    forecasts = model.predict(X_eval)

    result = df.iloc[eval_slice].copy()
    result["forecast"] = forecasts
    result["order_suggestion"] = forecasts

    history_len = horizon.history_days
    history_actual = result.iloc[:history_len]["demand"].values
    history_forecast = result.iloc[:history_len]["forecast"].values
    rmse = float(np.sqrt(np.mean((history_actual - history_forecast) ** 2)))

    metrics = {"rmse_history": rmse}

    phases = np.array(["history"] * history_len + ["forecast"] * horizon.forecast_days)
    if len(result) != len(phases):
        raise ValueError("Phase labelling mismatch with evaluation window.")
    result["phase"] = phases
    result["has_actual"] = result["phase"] == "history"
    return result, metrics


def run_experiment(config: ExperimentConfig) -> Dict[Tuple[str, str], pd.DataFrame]:
    feature_pool = _build_feature_pool(
        total_days=config.horizon.total_days,
        start_date=config.start_date,
        seed=config.seed,
    )

    outputs: Dict[Tuple[str, str], pd.DataFrame] = {}
    for product_name, model_name in config.assignments:
        product_cfg = config.products[product_name]
        model_cfg = config.models[model_name]

        product_df = _simulate_product_demand(feature_pool, product_name, product_cfg)
        forecast_df, metrics = _train_and_predict(
            product_df, model_cfg, config.horizon
        )
        forecast_df["model"] = model_name
        forecast_df["product"] = product_name
        forecast_df["model_label"] = model_name
        forecast_df["rmse_history"] = metrics["rmse_history"]
        outputs[(product_name, model_name)] = forecast_df.reset_index().rename(
            columns={"index": "date"}
        )

    return outputs


def save_experiment_outputs(
    outputs: Mapping[Tuple[str, str], pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for (product, model), df in outputs.items():
        filename = f"{product}_{model}.parquet"
        df.to_parquet(output_dir / filename, index=False)


def _parse_product_configs(raw: Mapping[str, object]) -> Dict[str, ProductConfig]:
    products: Dict[str, ProductConfig] = {}
    for name, params in raw.items():
        products[name] = ProductConfig(
            visible_features=params["visible_features"],
            hidden_features=params.get("hidden_features", []),
            noise_features=params.get("noise_features", []),
            betas=params["betas"],
            base_level=params.get("base_level", 50.0),
            trend_slope=params.get("trend_slope", 0.0),
            season_strength=params.get("season_strength", 8.0),
            epsilon_std=params.get("epsilon_std", 4.0),
            purchase_price=params.get("purchase_price", 3.0),
            selling_price=params.get("selling_price", 6.0),
            salvage_value=params.get("salvage_value", 0.5),
        )
    return products


def _parse_model_configs(raw: Mapping[str, object]) -> Dict[str, ModelConfig]:
    return {name: ModelConfig(features=params["features"]) for name, params in raw.items()}


def load_experiment(config_path: Path) -> Dict[Tuple[str, str], pd.DataFrame]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    exp_cfg = cfg_dict["experiment"]
    horizon = HorizonConfig(
        training_window=exp_cfg["horizon"]["training_window"],
        history_days=exp_cfg["horizon"]["history_days"],
        forecast_days=exp_cfg["horizon"]["forecast_days"],
    )

    products = _parse_product_configs(exp_cfg["products"])
    models = _parse_model_configs(exp_cfg["models"])
    assignments = [
        (item["product"], item["model"]) for item in exp_cfg["assignments"]
    ]

    config = ExperimentConfig(
        horizon=horizon,
        products=products,
        models=models,
        assignments=assignments,
        start_date=exp_cfg.get("start_date", "2024-01-01"),
        seed=exp_cfg.get("seed", 42),
        output_dir=Path(exp_cfg["output_dir"]),
    )

    outputs = run_experiment(config)
    save_experiment_outputs(outputs, config.output_dir)
    return outputs


__all__ = [
    "ExperimentConfig",
    "HorizonConfig",
    "ModelConfig",
    "ProductConfig",
    "run_experiment",
    "save_experiment_outputs",
    "load_experiment",
]
