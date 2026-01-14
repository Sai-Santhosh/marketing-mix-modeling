"""
Pytest configuration and fixtures for MMM Analytics tests.

This module provides shared fixtures for testing across all test modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_analytics.core.features import ChannelConfig, FeatureConfig, FeatureEngineer
from mmm_analytics.core.model import MarketingMixModel, ModelConfig
from mmm_analytics.core.pipeline import MMMPipeline, PipelineConfig
from mmm_analytics.data.simulator import DataSimulator, SimulationConfig


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate sample marketing data for testing."""
    simulator = DataSimulator(n_weeks=52, seed=42)
    return simulator.generate()


@pytest.fixture
def large_sample_data() -> pd.DataFrame:
    """Generate larger sample data for integration tests."""
    simulator = DataSimulator(n_weeks=104, seed=42)
    return simulator.generate()


@pytest.fixture
def minimal_data() -> pd.DataFrame:
    """Generate minimal data for edge case testing."""
    simulator = DataSimulator(n_weeks=20, seed=42)
    return simulator.generate()


@pytest.fixture
def sample_channels() -> list[str]:
    """Default channel list."""
    return ["search", "social", "display", "audio"]


@pytest.fixture
def sample_spend_array() -> np.ndarray:
    """Sample spend array for transform testing."""
    rng = np.random.default_rng(42)
    return rng.gamma(shape=5, scale=200, size=52)


@pytest.fixture
def feature_config(sample_channels: list[str]) -> FeatureConfig:
    """Sample feature configuration."""
    return FeatureConfig(
        channels=[
            ChannelConfig(name=ch, adstock_decay=0.5, saturation_alpha=2.0, saturation_k=500.0)
            for ch in sample_channels
        ],
        include_trend=True,
        include_seasonality=True,
    )


@pytest.fixture
def model_config() -> ModelConfig:
    """Sample model configuration."""
    return ModelConfig(
        alphas=(0.1, 1.0, 10.0),
        cv_folds=3,
        fit_intercept=True,
        scale_features=True,
    )


@pytest.fixture
def pipeline_config(sample_channels: list[str]) -> PipelineConfig:
    """Sample pipeline configuration."""
    return PipelineConfig(
        channels=sample_channels,
        target_column="kpi",
        adstock_decay=0.5,
        saturation_alpha=2.0,
        saturation_k=500.0,
        cv_folds=3,
    )


@pytest.fixture
def fitted_pipeline(sample_data: pd.DataFrame, pipeline_config: PipelineConfig) -> MMMPipeline:
    """Return a fitted pipeline for testing."""
    pipeline = MMMPipeline(pipeline_config)
    pipeline.fit(sample_data, run_optimization=False)
    return pipeline


@pytest.fixture
def feature_matrix(
    sample_data: pd.DataFrame, feature_config: FeatureConfig
) -> pd.DataFrame:
    """Generate feature matrix for testing."""
    engineer = FeatureEngineer(feature_config)
    return engineer.fit_transform(sample_data)


# Parametrized fixtures for property-based testing
@pytest.fixture(params=[0.3, 0.5, 0.7])
def decay_rate(request: pytest.FixtureRequest) -> float:
    """Parametrized decay rates for testing."""
    return request.param


@pytest.fixture(params=[1.0, 2.0, 3.0])
def saturation_alpha(request: pytest.FixtureRequest) -> float:
    """Parametrized saturation alpha values."""
    return request.param


@pytest.fixture(params=[250.0, 500.0, 1000.0])
def saturation_k(request: pytest.FixtureRequest) -> float:
    """Parametrized saturation k values."""
    return request.param
