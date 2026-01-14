"""
End-to-end MMM Pipeline.

This module provides a high-level pipeline that orchestrates all MMM components:
feature engineering, model fitting, attribution analysis, and budget optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog

from mmm_analytics.core.features import ChannelConfig, FeatureConfig, FeatureEngineer
from mmm_analytics.core.model import MarketingMixModel, ModelConfig, ModelResults
from mmm_analytics.core.optimizer import (
    BudgetOptimizer,
    OptimizationConstraints,
    OptimizationMethod,
    OptimizationResult,
)


if TYPE_CHECKING:
    pass


logger = structlog.get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the MMM Pipeline.

    Attributes:
        channels: List of channel names to analyze.
        target_column: Name of the target variable column.
        date_column: Name of the date/time column (optional).
        adstock_decay: Default adstock decay rate.
        saturation_alpha: Default saturation alpha parameter.
        saturation_k: Default saturation k parameter.
        model_alphas: Ridge regularization strengths to try.
        cv_folds: Number of cross-validation folds.
        include_trend: Whether to include trend features.
        include_seasonality: Whether to include seasonality features.
        optimization_method: Budget optimization method.
    """

    channels: list[str] = field(default_factory=list)
    target_column: str = "kpi"
    date_column: str | None = None
    adstock_decay: float = 0.5
    saturation_alpha: float = 2.0
    saturation_k: float = 500.0
    model_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    cv_folds: int = 5
    include_trend: bool = True
    include_seasonality: bool = True
    optimization_method: str = "scipy"


@dataclass
class PipelineResults:
    """Complete pipeline results.

    Attributes:
        model_results: Model fitting and attribution results.
        optimization_results: Budget optimization results (if run).
        feature_matrix: Transformed feature matrix.
        raw_data: Original input data.
        config: Pipeline configuration used.
    """

    model_results: ModelResults
    optimization_results: OptimizationResult | None
    feature_matrix: pd.DataFrame
    raw_data: pd.DataFrame
    config: PipelineConfig

    def get_channel_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of channel performance."""
        attribution = self.model_results.attribution

        summary_data = {
            "coefficient": attribution.coefficients,
            "contribution": attribution.channel_contributions,
            "share": attribution.contribution_share,
            "roas_index": attribution.roas_index,
        }

        # Filter to only channel features
        channel_features = [f"{ch}_transformed" for ch in self.config.channels]
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.loc[summary_df.index.isin(channel_features)]

        # Rename index to channel names
        summary_df.index = [idx.replace("_transformed", "") for idx in summary_df.index]

        return summary_df

    def save_results(self, output_dir: str | Path) -> None:
        """Save results to files.

        Args:
            output_dir: Directory to save results.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_df = pd.DataFrame([self.model_results.metrics.to_dict()])
        metrics_df.to_csv(output_path / "metrics.csv", index=False)

        # Save coefficients
        self.model_results.attribution.coefficients.to_csv(
            output_path / "coefficients.csv", header=["coefficient"]
        )

        # Save channel summary
        self.get_channel_summary().to_csv(output_path / "channel_summary.csv")

        # Save predictions
        predictions_df = pd.DataFrame(
            {
                "actual": self.raw_data[self.config.target_column],
                "predicted": self.model_results.predictions,
                "residual": self.model_results.residuals,
            }
        )
        predictions_df.to_csv(output_path / "predictions.csv", index=False)

        # Save optimization results if available
        if self.optimization_results:
            opt_df = pd.DataFrame(
                {
                    "current": self.optimization_results.current_allocation,
                    "optimized": self.optimization_results.optimized_allocation,
                    "delta": self.optimization_results.reallocation_delta,
                }
            )
            opt_df.to_csv(output_path / "optimization.csv")

        logger.info("Results saved", output_dir=str(output_path))


class MMMPipeline:
    """End-to-end Marketing Mix Modeling pipeline.

    This pipeline orchestrates:
    1. Feature engineering (adstock + saturation transforms)
    2. Model fitting (Ridge regression with CV)
    3. Attribution analysis (contribution, ROAS)
    4. Budget optimization (optional)

    Example:
        >>> config = PipelineConfig(
        ...     channels=["search", "social", "display", "audio"],
        ...     target_column="conversions",
        ... )
        >>> pipeline = MMMPipeline(config)
        >>> results = pipeline.fit(data)
        >>> print(results.model_results.summary())

    Attributes:
        config: PipelineConfig with all settings.
        feature_engineer: FeatureEngineer instance.
        model: MarketingMixModel instance.
        optimizer: BudgetOptimizer instance.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the MMM Pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or PipelineConfig()
        self._is_fitted = False
        self.feature_engineer: FeatureEngineer | None = None
        self.model: MarketingMixModel | None = None
        self.optimizer: BudgetOptimizer | None = None
        self.results_: PipelineResults | None = None

    def fit(
        self,
        data: pd.DataFrame,
        run_optimization: bool = True,
    ) -> PipelineResults:
        """Fit the complete MMM pipeline.

        Args:
            data: Input DataFrame with spend columns and target.
            run_optimization: Whether to run budget optimization.

        Returns:
            PipelineResults with all analysis outputs.

        Raises:
            ValueError: If required columns are missing.
        """
        logger.info(
            "Starting MMM pipeline",
            n_rows=len(data),
            channels=self.config.channels,
        )

        # Validate input
        self._validate_input(data)

        # Auto-detect channels if not specified
        if not self.config.channels:
            self.config.channels = self._detect_channels(data)
            logger.info("Auto-detected channels", channels=self.config.channels)

        # Step 1: Feature Engineering
        logger.info("Step 1: Feature engineering")
        feature_config = self._build_feature_config()
        self.feature_engineer = FeatureEngineer(feature_config)
        X = self.feature_engineer.fit_transform(data)

        # Step 2: Model Fitting
        logger.info("Step 2: Model fitting")
        model_config = ModelConfig(
            alphas=self.config.model_alphas,
            cv_folds=self.config.cv_folds,
        )
        self.model = MarketingMixModel(model_config)

        y = data[self.config.target_column]
        channel_features = self.feature_engineer.get_channel_features()
        model_results = self.model.fit(X, y, channel_features=channel_features)

        logger.info(
            "Model fitted",
            r2=f"{model_results.metrics.r2:.4f}",
            alpha=model_results.selected_alpha,
        )

        # Step 3: Budget Optimization (optional)
        optimization_results = None
        if run_optimization:
            logger.info("Step 3: Budget optimization")
            optimization_results = self._run_optimization(data, model_results)

        self.results_ = PipelineResults(
            model_results=model_results,
            optimization_results=optimization_results,
            feature_matrix=X,
            raw_data=data.copy(),
            config=self.config,
        )

        self._is_fitted = True
        logger.info("Pipeline complete")

        return self.results_

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data.

        Args:
            data: Input DataFrame with same structure as training data.

        Returns:
            Predicted values.

        Raises:
            RuntimeError: If pipeline has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before predict")

        if self.feature_engineer is None or self.model is None:
            raise RuntimeError("Pipeline components not initialized")

        X = self.feature_engineer.transform(data)
        return self.model.predict(X)

    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data has required columns."""
        if self.config.target_column not in data.columns:
            raise ValueError(
                f"Target column '{self.config.target_column}' not found in data"
            )

        for channel in self.config.channels:
            spend_col = f"spend_{channel}"
            if spend_col not in data.columns:
                raise ValueError(f"Spend column '{spend_col}' not found for channel '{channel}'")

    def _detect_channels(self, data: pd.DataFrame) -> list[str]:
        """Auto-detect channel names from spend columns."""
        channels = []
        for col in data.columns:
            if col.startswith("spend_"):
                channel = col.replace("spend_", "")
                channels.append(channel)
        return channels

    def _build_feature_config(self) -> FeatureConfig:
        """Build FeatureConfig from pipeline config."""
        channel_configs = [
            ChannelConfig(
                name=ch,
                adstock_decay=self.config.adstock_decay,
                saturation_alpha=self.config.saturation_alpha,
                saturation_k=self.config.saturation_k,
            )
            for ch in self.config.channels
        ]

        return FeatureConfig(
            channels=channel_configs,
            include_trend=self.config.include_trend,
            include_seasonality=self.config.include_seasonality,
        )

    def _run_optimization(
        self,
        data: pd.DataFrame,
        model_results: ModelResults,
    ) -> OptimizationResult:
        """Run budget optimization."""
        # Get current spend (use last period or average)
        spend_cols = [f"spend_{ch}" for ch in self.config.channels]
        current_spend = data[spend_cols].mean()
        current_spend.index = [col.replace("spend_", "") for col in current_spend.index]

        # Get channel coefficients (map from transformed names)
        coefficients = pd.Series(dtype=float)
        for ch in self.config.channels:
            feat_name = f"{ch}_transformed"
            if feat_name in model_results.attribution.coefficients.index:
                coefficients[ch] = model_results.attribution.coefficients[feat_name]

        # Saturation parameters for response curves
        sat_params = {
            ch: (self.config.saturation_alpha, self.config.saturation_k)
            for ch in self.config.channels
        }

        self.optimizer = BudgetOptimizer(
            method=OptimizationMethod(self.config.optimization_method)
        )

        constraints = OptimizationConstraints(
            total_budget=current_spend.sum(),
        )

        return self.optimizer.optimize(
            current_spend=current_spend,
            coefficients=coefficients,
            saturation_params=sat_params,
            constraints=constraints,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        status = "fitted" if self._is_fitted else "not fitted"
        n_channels = len(self.config.channels)
        return f"MMMPipeline(channels={n_channels}, status={status})"
