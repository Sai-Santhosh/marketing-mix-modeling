"""
Tests for feature engineering module.

Tests cover:
- FeatureEngineer initialization and configuration
- Feature matrix construction
- Transform application
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_analytics.core.features import ChannelConfig, FeatureConfig, FeatureEngineer


class TestChannelConfig:
    """Tests for ChannelConfig dataclass."""

    def test_default_spend_column(self) -> None:
        """Test that spend column defaults to spend_{name}."""
        config = ChannelConfig(name="search")
        assert config.spend_column == "spend_search"

    def test_explicit_spend_column(self) -> None:
        """Test explicit spend column setting."""
        config = ChannelConfig(name="search", spend_column="my_search_spend")
        assert config.spend_column == "my_search_spend"

    def test_default_params(self) -> None:
        """Test default transformation parameters."""
        config = ChannelConfig(name="social")
        assert config.adstock_decay == 0.5
        assert config.saturation_alpha == 2.0
        assert config.saturation_k == 500.0


class TestFeatureConfig:
    """Tests for FeatureConfig dataclass."""

    def test_empty_channels(self) -> None:
        """Test default empty channels list."""
        config = FeatureConfig()
        assert config.channels == []

    def test_seasonality_defaults(self) -> None:
        """Test seasonality default settings."""
        config = FeatureConfig()
        assert config.include_seasonality is True
        assert config.seasonality_period == 52
        assert config.n_harmonics == 2


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""

    def test_from_channels_factory(self) -> None:
        """Test factory method for creating from channel names."""
        engineer = FeatureEngineer.from_channels(
            ["search", "social"],
            default_decay=0.6,
        )
        assert len(engineer.config.channels) == 2
        assert engineer.config.channels[0].adstock_decay == 0.6

    def test_fit_transform(self, sample_data: pd.DataFrame) -> None:
        """Test fit_transform creates correct feature matrix."""
        config = FeatureConfig(
            channels=[
                ChannelConfig(name="search"),
                ChannelConfig(name="social"),
            ]
        )
        engineer = FeatureEngineer(config)
        X = engineer.fit_transform(sample_data)

        # Check expected columns exist
        assert "search_transformed" in X.columns
        assert "social_transformed" in X.columns
        assert "trend" in X.columns

        # Check shape
        assert len(X) == len(sample_data)

    def test_fit_then_transform(self, sample_data: pd.DataFrame) -> None:
        """Test separate fit and transform."""
        config = FeatureConfig(
            channels=[ChannelConfig(name="search")]
        )
        engineer = FeatureEngineer(config)

        engineer.fit(sample_data)
        assert engineer._is_fitted

        X = engineer.transform(sample_data)
        assert "search_transformed" in X.columns

    def test_transform_before_fit_raises(self, sample_data: pd.DataFrame) -> None:
        """Test that transform before fit raises error."""
        engineer = FeatureEngineer()
        with pytest.raises(RuntimeError, match="must be fitted"):
            engineer.transform(sample_data)

    def test_missing_column_raises(self, sample_data: pd.DataFrame) -> None:
        """Test that missing spend column raises error."""
        config = FeatureConfig(
            channels=[ChannelConfig(name="nonexistent")]
        )
        engineer = FeatureEngineer(config)

        with pytest.raises(ValueError, match="Missing required columns"):
            engineer.fit(sample_data)

    def test_trend_feature(self, sample_data: pd.DataFrame) -> None:
        """Test trend feature is sequential."""
        config = FeatureConfig(
            channels=[],
            include_trend=True,
            include_seasonality=False,
        )
        engineer = FeatureEngineer(config)
        X = engineer.fit_transform(sample_data)

        # Trend should be 0, 1, 2, ...
        expected_trend = np.arange(len(sample_data), dtype=np.float64)
        np.testing.assert_array_equal(X["trend"].values, expected_trend)

    def test_seasonality_features(self, sample_data: pd.DataFrame) -> None:
        """Test seasonality features are created."""
        config = FeatureConfig(
            channels=[],
            include_trend=False,
            include_seasonality=True,
            n_harmonics=2,
        )
        engineer = FeatureEngineer(config)
        X = engineer.fit_transform(sample_data)

        # Should have sin_1, cos_1, sin_2, cos_2
        assert "sin_1" in X.columns
        assert "cos_1" in X.columns
        assert "sin_2" in X.columns
        assert "cos_2" in X.columns

    def test_no_seasonality(self, sample_data: pd.DataFrame) -> None:
        """Test disabling seasonality features."""
        config = FeatureConfig(
            channels=[],
            include_trend=True,
            include_seasonality=False,
        )
        engineer = FeatureEngineer(config)
        X = engineer.fit_transform(sample_data)

        assert "sin_1" not in X.columns
        assert "cos_1" not in X.columns

    def test_transformed_features_bounded(self, sample_data: pd.DataFrame) -> None:
        """Test that transformed channel features are in [0, 1]."""
        config = FeatureConfig(
            channels=[
                ChannelConfig(name="search"),
                ChannelConfig(name="social"),
            ]
        )
        engineer = FeatureEngineer(config)
        X = engineer.fit_transform(sample_data)

        for col in engineer.get_channel_features():
            assert X[col].min() >= 0
            assert X[col].max() <= 1

    def test_get_channel_features(self, sample_data: pd.DataFrame) -> None:
        """Test get_channel_features returns correct names."""
        config = FeatureConfig(
            channels=[
                ChannelConfig(name="search"),
                ChannelConfig(name="social"),
            ]
        )
        engineer = FeatureEngineer(config)
        engineer.fit(sample_data)

        channel_feats = engineer.get_channel_features()
        assert "search_transformed" in channel_feats
        assert "social_transformed" in channel_feats

    def test_get_control_features(self, sample_data: pd.DataFrame) -> None:
        """Test get_control_features returns non-channel features."""
        config = FeatureConfig(
            channels=[ChannelConfig(name="search")],
            include_trend=True,
            include_seasonality=True,
            n_harmonics=1,
        )
        engineer = FeatureEngineer(config)
        engineer.fit(sample_data)

        control_feats = engineer.get_control_features()
        assert "trend" in control_feats
        assert "sin_1" in control_feats
        assert "search_transformed" not in control_feats

    def test_control_columns(self, sample_data: pd.DataFrame) -> None:
        """Test including additional control columns."""
        config = FeatureConfig(
            channels=[],
            include_trend=False,
            include_seasonality=False,
            control_columns=["seasonality"],  # Use existing column
        )
        engineer = FeatureEngineer(config)
        X = engineer.fit_transform(sample_data)

        assert "seasonality" in X.columns
        np.testing.assert_array_equal(
            X["seasonality"].values, sample_data["seasonality"].values
        )

    def test_repr(self) -> None:
        """Test string representation."""
        config = FeatureConfig(
            channels=[ChannelConfig(name="search"), ChannelConfig(name="social")]
        )
        engineer = FeatureEngineer(config)

        assert "2" in repr(engineer)
        assert "False" in repr(engineer)  # not fitted
