"""
Tests for data simulation module.

Tests cover:
- Synthetic data generation
- Ground truth tracking
- Configuration handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_analytics.data.simulator import (
    ChannelParams,
    DataSimulator,
    SimulationConfig,
    generate_sample_data,
)


class TestChannelParams:
    """Tests for ChannelParams dataclass."""

    def test_default_values(self) -> None:
        """Test default parameter values."""
        params = ChannelParams(name="search")
        assert params.base_spend == 1000.0
        assert params.spend_volatility == 0.3
        assert params.true_coefficient == 100.0

    def test_custom_values(self) -> None:
        """Test custom parameter values."""
        params = ChannelParams(
            name="social",
            base_spend=500.0,
            true_coefficient=80.0,
            adstock_decay=0.4,
        )
        assert params.base_spend == 500.0
        assert params.true_coefficient == 80.0
        assert params.adstock_decay == 0.4


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_default_channels(self) -> None:
        """Test that default channels are created."""
        config = SimulationConfig()
        assert len(config.channels) == 4
        channel_names = [ch.name for ch in config.channels]
        assert "search" in channel_names
        assert "social" in channel_names

    def test_custom_channels(self) -> None:
        """Test custom channel configuration."""
        custom_channels = [
            ChannelParams(name="paid_search"),
            ChannelParams(name="meta"),
        ]
        config = SimulationConfig(channels=custom_channels)
        assert len(config.channels) == 2

    def test_default_params(self) -> None:
        """Test default simulation parameters."""
        config = SimulationConfig()
        assert config.n_weeks == 104
        assert config.base_response == 200.0
        assert config.noise_std == 15.0


class TestDataSimulator:
    """Tests for DataSimulator."""

    def test_generate_basic(self) -> None:
        """Test basic data generation."""
        simulator = DataSimulator(n_weeks=52, seed=42)
        data = simulator.generate()

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 52

    def test_generate_columns(self) -> None:
        """Test generated columns."""
        simulator = DataSimulator(n_weeks=52, seed=42)
        data = simulator.generate()

        # Check essential columns
        assert "date" in data.columns
        assert "week" in data.columns
        assert "kpi" in data.columns
        assert "trend" in data.columns
        assert "seasonality" in data.columns

        # Check spend columns
        assert "spend_search" in data.columns
        assert "spend_social" in data.columns
        assert "spend_display" in data.columns
        assert "spend_audio" in data.columns

    def test_reproducibility(self) -> None:
        """Test that same seed produces same data."""
        sim1 = DataSimulator(n_weeks=52, seed=42)
        sim2 = DataSimulator(n_weeks=52, seed=42)

        data1 = sim1.generate()
        data2 = sim2.generate()

        pd.testing.assert_frame_equal(data1, data2)

    def test_different_seeds(self) -> None:
        """Test that different seeds produce different data."""
        sim1 = DataSimulator(n_weeks=52, seed=42)
        sim2 = DataSimulator(n_weeks=52, seed=123)

        data1 = sim1.generate()
        data2 = sim2.generate()

        # KPIs should differ
        assert not np.allclose(data1["kpi"].values, data2["kpi"].values)

    def test_positive_kpi(self) -> None:
        """Test that KPI values are non-negative."""
        simulator = DataSimulator(n_weeks=104, seed=42)
        data = simulator.generate()

        assert (data["kpi"] >= 0).all()

    def test_positive_spend(self) -> None:
        """Test that spend values are positive."""
        simulator = DataSimulator(n_weeks=104, seed=42)
        data = simulator.generate()

        spend_cols = [c for c in data.columns if c.startswith("spend_")]
        for col in spend_cols:
            assert (data[col] > 0).all()

    def test_date_column(self) -> None:
        """Test date column is properly formatted."""
        simulator = DataSimulator(n_weeks=52, seed=42)
        data = simulator.generate()

        assert pd.api.types.is_datetime64_any_dtype(data["date"])
        # Dates should be weekly (7 days apart)
        date_diffs = data["date"].diff().dropna()
        assert (date_diffs == pd.Timedelta(days=7)).all()

    def test_ground_truth_stored(self) -> None:
        """Test that ground truth is stored after generation."""
        simulator = DataSimulator(n_weeks=52, seed=42)
        simulator.generate()

        assert simulator.ground_truth_ is not None
        assert "base_response" in simulator.ground_truth_
        assert "channel_params" in simulator.ground_truth_

    def test_get_true_coefficients(self) -> None:
        """Test getting true coefficients."""
        simulator = DataSimulator(n_weeks=52, seed=42)
        simulator.generate()

        coefs = simulator.get_true_coefficients()
        assert isinstance(coefs, pd.Series)
        assert len(coefs) == 4  # 4 default channels

    def test_get_true_coefficients_before_generate(self) -> None:
        """Test error when getting coefficients before generate."""
        simulator = DataSimulator(n_weeks=52, seed=42)

        with pytest.raises(RuntimeError, match="generate"):
            simulator.get_true_coefficients()

    def test_get_true_contribution_shares(self) -> None:
        """Test getting true contribution shares."""
        simulator = DataSimulator(n_weeks=52, seed=42)
        simulator.generate()

        shares = simulator.get_true_contribution_shares()
        assert isinstance(shares, pd.Series)
        assert shares.sum() == pytest.approx(1.0)
        assert (shares >= 0).all()

    def test_custom_config(self) -> None:
        """Test with custom configuration."""
        config = SimulationConfig(
            n_weeks=30,
            base_response=500.0,
            noise_std=10.0,
            random_seed=123,
        )
        simulator = DataSimulator(config=config)
        data = simulator.generate()

        assert len(data) == 30

    def test_shortcut_parameters(self) -> None:
        """Test shortcut n_weeks and seed parameters."""
        simulator = DataSimulator(n_weeks=26, seed=99)
        data = simulator.generate()

        assert len(data) == 26

    def test_trend_component(self) -> None:
        """Test trend component is generated."""
        config = SimulationConfig(n_weeks=52, trend_coefficient=2.0)
        simulator = DataSimulator(config=config)
        data = simulator.generate()

        # Trend should increase
        assert data["trend"].iloc[-1] > data["trend"].iloc[0]

    def test_seasonality_component(self) -> None:
        """Test seasonality component is generated."""
        config = SimulationConfig(
            n_weeks=104,  # 2 years to see full cycle
            seasonality_amplitude=50.0,
        )
        simulator = DataSimulator(config=config)
        data = simulator.generate()

        # Seasonality should oscillate
        season = data["seasonality"]
        assert season.max() > 0
        assert season.min() < 0

    def test_repr(self) -> None:
        """Test string representation."""
        simulator = DataSimulator(n_weeks=52, seed=42)
        repr_str = repr(simulator)

        assert "52" in repr_str
        assert "4" in repr_str  # 4 channels


class TestGenerateSampleData:
    """Tests for convenience function."""

    def test_generate_sample_data(self) -> None:
        """Test convenience function."""
        data = generate_sample_data(n_weeks=30, seed=42)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 30

    def test_generate_sample_data_defaults(self) -> None:
        """Test convenience function with defaults."""
        data = generate_sample_data()

        assert len(data) == 104  # Default weeks
