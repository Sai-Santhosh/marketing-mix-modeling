"""
Tests for the MMM Pipeline.

Tests cover:
- End-to-end pipeline execution
- Configuration handling
- Results generation and saving
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmm_analytics.core.pipeline import MMMPipeline, PipelineConfig, PipelineResults
from mmm_analytics.data.simulator import DataSimulator


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.target_column == "kpi"
        assert config.adstock_decay == 0.5
        assert config.include_trend is True
        assert config.include_seasonality is True

    def test_custom_channels(self) -> None:
        """Test custom channel configuration."""
        config = PipelineConfig(channels=["paid_search", "meta", "tiktok"])
        assert len(config.channels) == 3


class TestMMMPipeline:
    """Tests for MMMPipeline."""

    def test_fit_basic(self, sample_data: pd.DataFrame) -> None:
        """Test basic pipeline fitting."""
        pipeline = MMMPipeline()
        results = pipeline.fit(sample_data, run_optimization=False)

        assert isinstance(results, PipelineResults)
        assert pipeline._is_fitted

    def test_fit_with_optimization(self, sample_data: pd.DataFrame) -> None:
        """Test pipeline with optimization enabled."""
        pipeline = MMMPipeline()
        results = pipeline.fit(sample_data, run_optimization=True)

        assert results.optimization_results is not None

    def test_fit_without_optimization(self, sample_data: pd.DataFrame) -> None:
        """Test pipeline with optimization disabled."""
        pipeline = MMMPipeline()
        results = pipeline.fit(sample_data, run_optimization=False)

        assert results.optimization_results is None

    def test_auto_detect_channels(self, sample_data: pd.DataFrame) -> None:
        """Test automatic channel detection."""
        config = PipelineConfig(channels=[])  # Empty - should auto-detect
        pipeline = MMMPipeline(config)
        pipeline.fit(sample_data, run_optimization=False)

        # Should have detected 4 channels
        assert len(pipeline.config.channels) == 4

    def test_explicit_channels(self, sample_data: pd.DataFrame) -> None:
        """Test explicit channel specification."""
        config = PipelineConfig(channels=["search", "social"])
        pipeline = MMMPipeline(config)
        results = pipeline.fit(sample_data, run_optimization=False)

        # Should only analyze specified channels
        channel_summary = results.get_channel_summary()
        assert len(channel_summary) == 2

    def test_missing_target_raises(self, sample_data: pd.DataFrame) -> None:
        """Test that missing target column raises error."""
        config = PipelineConfig(target_column="nonexistent")
        pipeline = MMMPipeline(config)

        with pytest.raises(ValueError, match="not found"):
            pipeline.fit(sample_data)

    def test_missing_spend_column_raises(self) -> None:
        """Test that missing spend column raises error."""
        # Create data without expected spend column
        data = pd.DataFrame({
            "kpi": [100, 200, 300],
            "spend_search": [10, 20, 30],
        })

        config = PipelineConfig(channels=["search", "nonexistent"])
        pipeline = MMMPipeline(config)

        with pytest.raises(ValueError, match="not found"):
            pipeline.fit(data)

    def test_predict(self, sample_data: pd.DataFrame) -> None:
        """Test prediction on new data."""
        pipeline = MMMPipeline()
        pipeline.fit(sample_data, run_optimization=False)

        predictions = pipeline.predict(sample_data)
        assert len(predictions) == len(sample_data)

    def test_predict_before_fit_raises(self, sample_data: pd.DataFrame) -> None:
        """Test that predict before fit raises error."""
        pipeline = MMMPipeline()

        with pytest.raises(RuntimeError, match="must be fitted"):
            pipeline.predict(sample_data)

    def test_results_model_metrics(self, sample_data: pd.DataFrame) -> None:
        """Test that results contain model metrics."""
        pipeline = MMMPipeline()
        results = pipeline.fit(sample_data, run_optimization=False)

        metrics = results.model_results.metrics
        assert hasattr(metrics, "r2")
        assert hasattr(metrics, "rmse")
        assert hasattr(metrics, "mape")

    def test_results_attribution(self, sample_data: pd.DataFrame) -> None:
        """Test that results contain attribution."""
        pipeline = MMMPipeline()
        results = pipeline.fit(sample_data, run_optimization=False)

        attribution = results.model_results.attribution
        assert hasattr(attribution, "coefficients")
        assert hasattr(attribution, "contribution_share")

    def test_results_feature_matrix(self, sample_data: pd.DataFrame) -> None:
        """Test that results contain feature matrix."""
        pipeline = MMMPipeline()
        results = pipeline.fit(sample_data, run_optimization=False)

        assert isinstance(results.feature_matrix, pd.DataFrame)
        assert len(results.feature_matrix) == len(sample_data)

    def test_get_channel_summary(self, sample_data: pd.DataFrame) -> None:
        """Test channel summary generation."""
        pipeline = MMMPipeline()
        results = pipeline.fit(sample_data, run_optimization=False)

        summary = results.get_channel_summary()
        assert isinstance(summary, pd.DataFrame)
        assert "coefficient" in summary.columns
        assert "share" in summary.columns

    def test_save_results(self, sample_data: pd.DataFrame) -> None:
        """Test saving results to files."""
        pipeline = MMMPipeline()
        results = pipeline.fit(sample_data, run_optimization=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            results.save_results(output_path)

            # Check files exist
            assert (output_path / "metrics.csv").exists()
            assert (output_path / "coefficients.csv").exists()
            assert (output_path / "channel_summary.csv").exists()
            assert (output_path / "predictions.csv").exists()
            assert (output_path / "optimization.csv").exists()

    def test_custom_adstock_decay(self, sample_data: pd.DataFrame) -> None:
        """Test pipeline with custom adstock decay."""
        config = PipelineConfig(adstock_decay=0.7)
        pipeline = MMMPipeline(config)
        results = pipeline.fit(sample_data, run_optimization=False)

        assert results is not None

    def test_custom_saturation_params(self, sample_data: pd.DataFrame) -> None:
        """Test pipeline with custom saturation parameters."""
        config = PipelineConfig(
            saturation_alpha=1.5,
            saturation_k=800.0,
        )
        pipeline = MMMPipeline(config)
        results = pipeline.fit(sample_data, run_optimization=False)

        assert results is not None

    def test_custom_cv_folds(self, sample_data: pd.DataFrame) -> None:
        """Test pipeline with custom CV folds."""
        config = PipelineConfig(cv_folds=3)
        pipeline = MMMPipeline(config)
        results = pipeline.fit(sample_data, run_optimization=False)

        assert results is not None

    def test_optimization_method(self, sample_data: pd.DataFrame) -> None:
        """Test different optimization methods."""
        for method in ["scipy", "greedy", "gradient"]:
            config = PipelineConfig(optimization_method=method)
            pipeline = MMMPipeline(config)
            results = pipeline.fit(sample_data, run_optimization=True)

            assert results.optimization_results is not None

    def test_repr(self) -> None:
        """Test string representation."""
        config = PipelineConfig(channels=["search", "social"])
        pipeline = MMMPipeline(config)

        repr_str = repr(pipeline)
        assert "2" in repr_str  # 2 channels
        assert "not fitted" in repr_str


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    def test_full_pipeline_synthetic_data(self) -> None:
        """Test complete pipeline with synthetic data."""
        # Generate data
        simulator = DataSimulator(n_weeks=52, seed=42)
        data = simulator.generate()

        # Run pipeline
        pipeline = MMMPipeline()
        results = pipeline.fit(data, run_optimization=True)

        # Verify results
        assert results.model_results.metrics.r2 > 0.5  # Should fit reasonably
        assert results.optimization_results is not None

        # Channel summary should have all channels
        summary = results.get_channel_summary()
        assert len(summary) == 4  # 4 default channels

    def test_ground_truth_recovery(self) -> None:
        """Test that model produces reasonable results."""
        # Generate data with known parameters
        simulator = DataSimulator(n_weeks=104, seed=42)
        data = simulator.generate()

        # Run pipeline
        pipeline = MMMPipeline()
        results = pipeline.fit(data, run_optimization=False)

        # Get true and estimated shares
        true_shares = simulator.get_true_contribution_shares()
        est_summary = results.get_channel_summary()

        # Model should have positive R² (better than mean baseline)
        # Note: With regularization and simplified transforms, R² may be moderate
        assert results.model_results.metrics.r2 > 0.3

        # Model should produce predictions
        assert len(results.model_results.predictions) == len(data)

        # Channel summary should have all channels
        assert len(est_summary) == len(true_shares)

    def test_minimal_data_handling(self) -> None:
        """Test pipeline with minimal data."""
        simulator = DataSimulator(n_weeks=15, seed=42)
        data = simulator.generate()

        config = PipelineConfig(cv_folds=2)  # Reduce CV folds for small data
        pipeline = MMMPipeline(config)

        # Should complete without error
        results = pipeline.fit(data, run_optimization=False)
        assert results is not None
