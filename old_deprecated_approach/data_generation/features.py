# simulator/features.py
import numpy as np
import pandas as pd


def ar1_feature(T, phi=0.5, sigma=1.0, seed=None, name="ar1"):
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    eps = rng.normal(0, sigma, size=T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + eps[t]
    return pd.Series(x, name=name)


def binary_holiday(T, prob=0.05, cluster_prob=0.3, seed=None, name="holiday"):
    rng = np.random.default_rng(seed)
    x = (rng.random(T) < prob).astype(int)
    for t in range(1, T):
        if x[t - 1] == 1 and rng.random() < cluster_prob:
            x[t] = 1
    return pd.Series(x, name=name)


# promotion stepfunction
def promotion_step_function(T, prob=0.1, cluster_prob=0.4, seed=None, name="promotion"):
    rng = np.random.default_rng(seed)
    promo = (rng.random(T) < prob).astype(int)
    for t in range(1, T):
        if promo[t - 1] == 1 and rng.random() < cluster_prob:
            promo[t] = 1
    return pd.Series(promo, name=name)


def price_step_function(
    T,
    weekday_high=10.0,
    weekend_low=8.0,
    noise=0.2,
    seed=None,
    name="price_step",
):
    """
    Weekly price profile switching between high (Mon-Thu) and low (Fri-Sun) levels.
    """
    rng = np.random.default_rng(seed)
    weekdays = np.arange(T) % 7
    base = np.where(weekdays >= 4, weekend_low, weekday_high)
    price = base + rng.normal(0, noise, size=T)
    return pd.Series(price, name=name)


def price_series(
    T,
    weekday_high=10.5,
    weekend_low=8.2,
    promo_drop=1.0,
    noise=0.25,
    seed=None,
    name="price",
):
    """
    Weekly price series with distinct high/low levels and occasional weekend promotions.
    """
    rng = np.random.default_rng(seed)
    weekdays = np.arange(T) % 7
    base = np.where(weekdays >= 4, weekend_low, weekday_high)  # Fri-Sun cheaper

    # add small cyclical variation to mimic promotional cycles
    promo_flag = ((weekdays == 5) | (weekdays == 6)).astype(float)
    promo_effect = promo_flag * promo_drop

    price = base - promo_effect + rng.normal(0, noise, size=T)
    return pd.Series(price, name=name)


def promo_series(T, prob=0.1, cluster_prob=0.4, seed=None, name="promo"):
    rng = np.random.default_rng(seed)
    promo = (rng.random(T) < prob).astype(int)
    for t in range(1, T):
        if promo[t - 1] == 1 and rng.random() < cluster_prob:
            promo[t] = 1
    return pd.Series(promo, name=name)


def temperature_series(
    T, base=15, amp=10, phi=0.7, sigma=1.0, seed=None, name="temperature"
):
    """seasonal sinusoid + AR noise"""
    rng = np.random.default_rng(seed)
    days = np.arange(T)
    seasonal = base + amp * np.sin(2 * np.pi * days / 365.0)
    noise = np.zeros(T)
    eps = rng.normal(0, sigma, size=T)
    for t in range(1, T):
        noise[t] = phi * noise[t - 1] + eps[t]
    return pd.Series(seasonal + noise, name=name)


def weekday_indicator(T, start_day=0, name="weekday"):
    """0=Mon,...,6=Sun"""
    days = np.arange(T)
    weekday = (start_day + days) % 7
    weight_map = {
        0: 1.5,  # Monday
        1: 1.0,  # Tuesday
        2: 1.0,  # Wednesday
        3: 1.5,  # Thursday
        4: 2.0,  # Friday
        5: 3.0,  # Saturday
        6: 1.0,  # Sunday
    }
    weighted = np.array([weight_map[idx] for idx in weekday])
    return pd.Series(weighted, name=name)


def rain_series(
    T,
    base_prob=0.2,
    seasonal_amplitude=0.15,
    seed=None,
    name="rain",
):
    """Binary rain indicator with mild seasonality and clustering."""
    rng = np.random.default_rng(seed)
    days = np.arange(T)
    seasonal = base_prob + seasonal_amplitude * np.sin(
        2 * np.pi * days / 365.0 + np.pi / 3
    )
    seasonal = np.clip(seasonal, 0.05, 0.9)
    rain = (rng.random(T) < seasonal).astype(int)
    for t in range(1, T):
        if rain[t - 1] == 1 and rng.random() < 0.35:
            rain[t] = 1
    return pd.Series(rain, name=name)


def market_index_series(T, drift=0.0005, sigma=0.01, seed=None, name="market_index"):
    """Slow-moving market index as a geometric random walk."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, sigma, size=T)
    levels = 100 * np.exp(np.cumsum(returns))
    return pd.Series(levels, name=name)


def odd_cycle_feature(T, period=9, amplitude=3.0, seed=None, name="odd_cycle"):
    """Quasi-periodic signal capturing idiosyncratic fluctuations."""
    rng = np.random.default_rng(seed)
    days = np.arange(T)
    signal = amplitude * np.sin(2 * np.pi * days / period)
    noise = rng.normal(0, amplitude * 0.2, size=T)
    return pd.Series(signal + noise, name=name)


def sports_event_flag(T, prob=0.05, seed=None, name="sports_event"):
    """Occasional event flag that triggers multi-day clusters."""
    rng = np.random.default_rng(seed)
    events = (rng.random(T) < prob).astype(int)
    for t in range(1, T):
        if events[t - 1] == 1 and rng.random() < 0.5:
            events[t] = 1
    return pd.Series(events, name=name)
