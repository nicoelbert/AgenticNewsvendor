"""
Data Generation Module

This module handles data generation for the research project.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataGenerator:
    """Main data generation class for creating synthetic datasets."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the data generator.

        Args:
            seed: Random seed for reproducible results
        """
        if seed is not None:
            np.random.seed(seed)

        logger.info("DataGenerator initialized")

    def generate_time_series(
        self,
        start_date: str,
        end_date: str,
        frequency: str = "D",
        noise_level: float = 0.1,
    ) -> pd.DataFrame:
        """
        Generate synthetic time series data.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Frequency of data points ('D' for daily, 'H' for hourly)
            noise_level: Level of random noise to add

        Returns:
            DataFrame with time series data
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)

        # Generate base trend
        base_trend = np.linspace(100, 200, len(date_range))

        # Add seasonal component
        seasonal = 20 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365.25)

        # Add noise
        noise = np.random.normal(0, noise_level * base_trend.std(), len(date_range))

        # Combine components
        values = base_trend + seasonal + noise

        df = pd.DataFrame(
            {
                "timestamp": date_range,
                "value": values,
                "trend": base_trend,
                "seasonal": seasonal,
                "noise": noise,
            }
        )

        logger.info(f"Generated time series with {len(df)} data points")
        return df

    def generate_demand_data(
        self, n_products: int = 10, n_periods: int = 100
    ) -> pd.DataFrame:
        """
        Generate synthetic demand data for newsvendor problem.

        Args:
            n_products: Number of products
            n_periods: Number of time periods

        Returns:
            DataFrame with demand data
        """
        products = [f"Product_{i:03d}" for i in range(1, n_products + 1)]
        periods = range(1, n_periods + 1)

        data = []
        for product in products:
            # Different demand patterns for each product
            base_demand = np.random.uniform(50, 200)
            volatility = np.random.uniform(0.1, 0.3)

            for period in periods:
                # Add some trend and seasonality
                trend_factor = 1 + (period - 1) * np.random.uniform(-0.001, 0.002)
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * period / 12)

                demand = max(
                    0,
                    np.random.normal(
                        base_demand * trend_factor * seasonal_factor,
                        base_demand * volatility,
                    ),
                )

                data.append(
                    {
                        "product": product,
                        "period": period,
                        "demand": round(demand, 2),
                        "base_demand": round(base_demand, 2),
                        "volatility": round(volatility, 3),
                    }
                )

        df = pd.DataFrame(data)
        logger.info(
            f"Generated demand data for {n_products} products over {n_periods} periods"
        )
        return df

    def export_data(self, data: pd.DataFrame, filepath: str, format: str = "csv"):
        """
        Export generated data to file.

        Args:
            data: DataFrame to export
            filepath: Output file path
            format: Export format ('csv', 'json', 'parquet')
        """
        if format == "csv":
            data.to_csv(filepath, index=False)
        elif format == "json":
            data.to_json(filepath, orient="records", date_format="iso")
        elif format == "parquet":
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Data exported to {filepath} in {format} format")


# Convenience re-exports for experiment pipeline
from .experiment import (  # noqa: E402
    ExperimentConfig,
    HorizonConfig,
    ModelConfig,
    ProductConfig,
    load_experiment,
    run_experiment,
    save_experiment_outputs,
)

__all__ = [
    "DataGenerator",
    "ExperimentConfig",
    "HorizonConfig",
    "ModelConfig",
    "ProductConfig",
    "load_experiment",
    "run_experiment",
    "save_experiment_outputs",
]
