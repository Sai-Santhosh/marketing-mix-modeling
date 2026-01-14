"""
Synthetic data generation for Marketing Mix Modeling.

This module provides realistic synthetic data generation for testing
and demonstrating MMM capabilities without requiring real marketing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class ChannelParams:
    """Parameters for a single media channel.

    Attributes:
        name: Channel identifier.
        base_spend: Average weekly spend.
        spend_volatility: Standard deviation of spend (as fraction of base).
        true_coefficient: True effect size for generating response.
        adstock_decay: True adstock decay for data generation.
        saturation_alpha: True saturation alpha.
        saturation_k: True saturation k (half-saturation point).
    """

    name: str
    base_spend: float = 1000.0
    spend_volatility: float = 0.3
    true_coefficient: float = 100.0
    adstock_decay: float = 0.5
    saturation_alpha: float = 2.0
    saturation_k: float = 500.0


@dataclass
class SimulationConfig:
    """Configuration for synthetic data generation.

    Attributes:
        n_weeks: Number of weeks to simulate.
        start_date: Start date for the time series.
        channels: List of channel configurations.
        base_response: Baseline response level (intercept).
        trend_coefficient: Weekly trend growth rate.
        seasonality_amplitude: Amplitude of seasonal variation.
        seasonality_period: Period of seasonality in weeks.
        noise_std: Standard deviation of random noise.
        random_seed: Random seed for reproducibility.
    """

    n_weeks: int = 104
    start_date: str = "2023-01-01"
    channels: list[ChannelParams] = field(default_factory=list)
    base_response: float = 200.0
    trend_coefficient: float = 0.5
    seasonality_amplitude: float = 20.0
    seasonality_period: int = 52
    noise_std: float = 15.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Set default channels if not provided."""
        if not self.channels:
            self.channels = [
                ChannelParams(
                    name="search",
                    base_spend=1200,
                    true_coefficient=120.0,
                    adstock_decay=0.4,
                    saturation_k=600,
                ),
                ChannelParams(
                    name="social",
                    base_spend=800,
                    true_coefficient=80.0,
                    adstock_decay=0.3,
                    saturation_k=400,
                ),
                ChannelParams(
                    name="display",
                    base_spend=600,
                    true_coefficient=40.0,
                    adstock_decay=0.6,
                    saturation_k=500,
                ),
                ChannelParams(
                    name="audio",
                    base_spend=400,
                    true_coefficient=50.0,
                    adstock_decay=0.5,
                    saturation_k=300,
                ),
            ]


class DataSimulator:
    """Generate synthetic marketing mix data.

    This class creates realistic synthetic data with known ground truth
    for testing and demonstrating MMM capabilities. The generated data
    includes:
    - Weekly time series
    - Channel spend with realistic patterns
    - Response variable with known effects
    - Trend and seasonality components

    Example:
        >>> config = SimulationConfig(n_weeks=104)
        >>> simulator = DataSimulator(config)
        >>> data = simulator.generate()
        >>> print(data.head())

    Attributes:
        config: SimulationConfig with generation parameters.
        ground_truth_: Dictionary of true parameters after generation.
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        n_weeks: int | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the DataSimulator.

        Args:
            config: Full configuration object.
            n_weeks: Shortcut for number of weeks (ignored if config provided).
            seed: Random seed (ignored if config provided).
        """
        if config is not None:
            self.config = config
        else:
            self.config = SimulationConfig(
                n_weeks=n_weeks or 104,
                random_seed=seed or 42,
            )

        self.ground_truth_: dict[str, Any] = {}
        self._rng = np.random.default_rng(self.config.random_seed)

    def generate(self) -> pd.DataFrame:
        """Generate synthetic marketing mix data.

        Returns:
            DataFrame with date, spend columns, control variables, and KPI.
        """
        n = self.config.n_weeks

        # Generate time index
        start = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        dates = [start + timedelta(weeks=i) for i in range(n)]
        weeks = np.arange(n)

        # Initialize DataFrame
        data: dict[str, NDArray[Any] | list[datetime]] = {
            "date": dates,
            "week": weeks,
        }

        # Generate baseline components
        trend = self.config.trend_coefficient * weeks
        seasonality = self.config.seasonality_amplitude * np.sin(
            2 * np.pi * weeks / self.config.seasonality_period
        )

        data["trend"] = trend
        data["seasonality"] = seasonality

        # Generate channel spend and calculate response
        total_response = self.config.base_response + trend + seasonality
        channel_contributions: dict[str, NDArray[np.float64]] = {}

        for channel in self.config.channels:
            # Generate spend with gamma distribution (realistic right skew)
            shape = 1 / (channel.spend_volatility**2)
            scale = channel.base_spend * channel.spend_volatility**2
            spend = self._rng.gamma(shape=shape, scale=scale, size=n)

            # Add weekly patterns (higher spend on certain weeks)
            weekly_pattern = 1 + 0.1 * np.sin(2 * np.pi * weeks / 4)
            spend = spend * weekly_pattern

            data[f"spend_{channel.name}"] = spend

            # Calculate true response (adstock + saturation + coefficient)
            adstocked = self._apply_adstock(spend, channel.adstock_decay)
            saturated = self._apply_saturation(
                adstocked, channel.saturation_alpha, channel.saturation_k
            )
            contribution = channel.true_coefficient * saturated

            channel_contributions[channel.name] = contribution
            total_response = total_response + contribution

        # Add noise
        noise = self._rng.normal(0, self.config.noise_std, size=n)
        data["noise"] = noise
        total_response = total_response + noise

        # Ensure positive response
        data["kpi"] = np.maximum(total_response, 0)

        # Store ground truth
        self.ground_truth_ = {
            "base_response": self.config.base_response,
            "trend_coefficient": self.config.trend_coefficient,
            "seasonality_amplitude": self.config.seasonality_amplitude,
            "noise_std": self.config.noise_std,
            "channel_params": {ch.name: ch for ch in self.config.channels},
            "channel_contributions": channel_contributions,
        }

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _apply_adstock(
        self, x: NDArray[np.float64], decay: float
    ) -> NDArray[np.float64]:
        """Apply geometric adstock transformation."""
        n = len(x)
        result = np.zeros(n, dtype=np.float64)
        carry = 0.0

        for i in range(n):
            carry = x[i] + decay * carry
            result[i] = carry

        return result

    def _apply_saturation(
        self, x: NDArray[np.float64], alpha: float, k: float
    ) -> NDArray[np.float64]:
        """Apply Hill saturation transformation."""
        x_alpha = np.power(x, alpha)
        k_alpha = k**alpha
        return x_alpha / (x_alpha + k_alpha + 1e-10)

    def get_true_coefficients(self) -> pd.Series:
        """Get the true coefficients used for data generation.

        Returns:
            Series with channel names and true coefficients.
        """
        if not self.ground_truth_:
            raise RuntimeError("Must call generate() first")

        return pd.Series(
            {ch.name: ch.true_coefficient for ch in self.config.channels}
        )

    def get_true_contribution_shares(self) -> pd.Series:
        """Get the true contribution share by channel.

        Returns:
            Series with normalized contribution shares.
        """
        if not self.ground_truth_:
            raise RuntimeError("Must call generate() first")

        contributions = self.ground_truth_["channel_contributions"]
        mean_contrib = pd.Series({k: float(v.mean()) for k, v in contributions.items()})
        return mean_contrib / mean_contrib.sum()

    def __repr__(self) -> str:
        """Return string representation."""
        n_channels = len(self.config.channels)
        return f"DataSimulator(n_weeks={self.config.n_weeks}, channels={n_channels})"


def generate_sample_data(
    n_weeks: int = 104,
    seed: int = 42,
) -> pd.DataFrame:
    """Convenience function to generate sample data.

    Args:
        n_weeks: Number of weeks to simulate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic marketing mix data.
    """
    simulator = DataSimulator(n_weeks=n_weeks, seed=seed)
    return simulator.generate()
