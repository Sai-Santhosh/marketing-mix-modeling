"""
Tests for the Marketing Mix Model.

Tests cover:
- Model fitting and prediction
- Coefficient estimation
- Attribution analysis
- Model diagnostics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_analytics.core.features import ChannelConfig, FeatureConfig, FeatureEngineer
from mmm_analytics.core.model import (
    MarketingMixModel,
    ModelConfig,
    ModelMetrics,
    ModelResults,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ModelConfig()
        assert len(config.alphas) == 5
        assert config.cv_folds == 5
        assert config.fit_intercept is True
        assert config.scale_features is True


class TestMarketingMixModel:
    """Tests for MarketingMixModel."""

    def test_fit_basic(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test basic model fitting."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        model = MarketingMixModel()
        results = model.fit(X, y)

        assert isinstance(results, ModelResults)
        assert model._is_fitted

    def test_fit_returns_metrics(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test that fit returns proper metrics."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        model = MarketingMixModel()
        results = model.fit(X, y)

        metrics = results.metrics
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.r2 <= 1
        assert metrics.rmse > 0
        assert metrics.mae > 0
        assert metrics.mape >= 0

    def test_fit_returns_coefficients(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test that fit returns coefficients."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        model = MarketingMixModel()
        results = model.fit(X, y)

        coef = results.attribution.coefficients
        assert len(coef) == X.shape[1]

    def test_fit_returns_attribution(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test that fit returns attribution analysis."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]
        channel_features = engineer.get_channel_features()

        model = MarketingMixModel()
        results = model.fit(X, y, channel_features=channel_features)

        attribution = results.attribution
        assert len(attribution.contribution_share) == len(channel_features)
        # Shares should sum to approximately 1 (or 0 if all contributions are negative)
        total_share = attribution.contribution_share.sum()
        assert total_share == pytest.approx(1.0, rel=0.01) or total_share == pytest.approx(0.0, abs=0.01)

    def test_predict(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test prediction on new data."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        model = MarketingMixModel()
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert predictions.dtype == np.float64

    def test_predict_before_fit_raises(self) -> None:
        """Test that predict before fit raises error."""
        model = MarketingMixModel()
        X = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(X)

    def test_mismatched_lengths_raises(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test that mismatched X and y lengths raise error."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"].iloc[:-5]  # Wrong length

        model = MarketingMixModel()
        with pytest.raises(ValueError, match="rows but y has"):
            model.fit(X, y)

    def test_residuals_properties(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test residual properties."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        model = MarketingMixModel()
        results = model.fit(X, y)

        # Residuals should have same length as data
        assert len(results.residuals) == len(y)
        # Residuals should have mean close to 0 for good fit
        assert np.abs(np.mean(results.residuals)) < np.std(results.residuals)

    def test_predictions_match_residuals(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test that y = predictions + residuals."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"].values

        model = MarketingMixModel()
        results = model.fit(X, y)

        reconstructed = results.predictions + results.residuals
        np.testing.assert_array_almost_equal(reconstructed, y, decimal=10)

    def test_vif_calculation(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test VIF calculation for multicollinearity."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        model = MarketingMixModel()
        results = model.fit(X, y)

        vif = results.vif
        assert len(vif) == X.shape[1]
        # VIF should be >= 1
        assert all(v >= 1.0 for v in vif if np.isfinite(v))

    def test_confidence_intervals(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test confidence interval calculation."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        model = MarketingMixModel()
        results = model.fit(X, y)

        ci = results.attribution.confidence_intervals
        assert "lower" in ci.columns
        assert "upper" in ci.columns
        assert len(ci) == X.shape[1]
        # Lower should be <= upper
        assert all(ci["lower"] <= ci["upper"])

    def test_feature_importance(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test feature importance ranking."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        model = MarketingMixModel()
        results = model.fit(X, y)

        importance = results.feature_importance
        assert len(importance) == X.shape[1]
        # Should be sorted descending
        assert list(importance.values) == sorted(importance.values, reverse=True)
        # Should be non-negative (absolute values)
        assert all(v >= 0 for v in importance.values)

    def test_summary_string(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test summary string generation."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        model = MarketingMixModel()
        results = model.fit(X, y)

        summary = results.summary()
        assert "R-squared" in summary
        assert "RMSE" in summary
        assert "CHANNEL CONTRIBUTION" in summary

    def test_custom_alphas(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test with custom regularization strengths."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        config = ModelConfig(alphas=(1.0, 10.0))
        model = MarketingMixModel(config)
        results = model.fit(X, y)

        # Selected alpha should be one of the provided values
        assert results.selected_alpha in [1.0, 10.0]

    def test_no_scaling(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test model without feature scaling."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        config = ModelConfig(scale_features=False)
        model = MarketingMixModel(config)
        results = model.fit(X, y)

        assert model.scaler_ is None
        assert results is not None

    def test_durbin_watson_bounds(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test Durbin-Watson statistic is in valid range."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        model = MarketingMixModel()
        results = model.fit(X, y)

        dw = results.metrics.durbin_watson
        assert 0 <= dw <= 4

    def test_coef_property(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test coef_ property access."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        model = MarketingMixModel()
        assert model.coef_ is None

        model.fit(X, y)
        assert model.coef_ is not None
        assert len(model.coef_) == X.shape[1]

    def test_intercept_property(
        self, sample_data: pd.DataFrame, feature_config: FeatureConfig
    ) -> None:
        """Test intercept_ property access."""
        engineer = FeatureEngineer(feature_config)
        X = engineer.fit_transform(sample_data)
        y = sample_data["kpi"]

        model = MarketingMixModel()
        assert model.intercept_ is None

        model.fit(X, y)
        assert model.intercept_ is not None

    def test_repr(self) -> None:
        """Test string representation."""
        model = MarketingMixModel()
        repr_str = repr(model)
        assert "not fitted" in repr_str

        # After fitting it should show fitted
        # (would need data to test this)
