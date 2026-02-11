# simulator/core.py
import numpy as np
import pandas as pd


def weekly_seasonality(T, strength=10.0):
    base = np.array([0, -1, 1, 3, 5, 8, 2])  # effect Mon..Sun
    days = np.arange(T)
    return strength * (base[days % 7] / np.max(np.abs(base)))


def generate_demand(
    features: pd.DataFrame,
    explanatory_vars: list,
    betas: dict,
    base_level=50,
    trend_slope=0.0,
    season_strength=0.0,
    epsilon_std=5.0,
    seed=None,
):
    """
    Demand = base + trend + weekly season + sum_{i in explanatory} beta_i * X_i + noise
    """
    rng = np.random.default_rng(seed)
    T = len(features)
    trend = trend_slope * np.arange(T)
    season = weekly_seasonality(T, season_strength)
    mu = base_level + trend + season

    # add feature contributions
    for v in explanatory_vars:
        mu += betas.get(v, 1.0) * features[v].values

    eps = rng.normal(0, epsilon_std, size=T)
    demand = np.maximum(0, mu + eps)
    return pd.Series(np.round(demand, 0), name="demand")
