"""
Marketing Mix Model implementation with Ridge regression.

This module provides the MarketingMixModel class that fits a Ridge regression
model to attribute marketing impact and calculate ROAS metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class ModelConfig:
    """Configuration for the Marketing Mix Model.

    Attributes:
        alphas: Regularization strengths to try in cross-validation.
        cv_folds: Number of cross-validation folds.
        fit_intercept: Whether to fit an intercept term.
        scale_features: Whether to standardize features before fitting.
    """

    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    cv_folds: int = 5
    fit_intercept: bool = True
    scale_features: bool = True


@dataclass
class ModelMetrics:
    """Model performance metrics.

    Attributes:
        r2: Coefficient of determination (R-squared).
        adj_r2: Adjusted R-squared.
        rmse: Root Mean Squared Error.
        mae: Mean Absolute Error.
        mape: Mean Absolute Percentage Error.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
        durbin_watson: Durbin-Watson statistic for autocorrelation.
    """

    r2: float
    adj_r2: float
    rmse: float
    mae: float
    mape: float
    aic: float
    bic: float
    durbin_watson: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "r2": self.r2,
            "adj_r2": self.adj_r2,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "aic": self.aic,
            "bic": self.bic,
            "durbin_watson": self.durbin_watson,
        }


@dataclass
class AttributionResult:
    """Attribution analysis results.

    Attributes:
        coefficients: Model coefficients for each feature.
        channel_contributions: Mean contribution by channel to target.
        contribution_share: Normalized share of contribution (0-1).
        roas_index: Relative ROAS index (higher = better efficiency).
        confidence_intervals: 95% CI for coefficients.
    """

    coefficients: pd.Series
    channel_contributions: pd.Series
    contribution_share: pd.Series
    roas_index: pd.Series
    confidence_intervals: pd.DataFrame


@dataclass
class ModelResults:
    """Complete model results container.

    Attributes:
        metrics: Model performance metrics.
        attribution: Attribution analysis results.
        predictions: Model predictions on training data.
        residuals: Model residuals.
        feature_importance: Feature importance ranking.
        selected_alpha: Best regularization strength from CV.
        vif: Variance Inflation Factors for multicollinearity.
    """

    metrics: ModelMetrics
    attribution: AttributionResult
    predictions: NDArray[np.float64]
    residuals: NDArray[np.float64]
    feature_importance: pd.Series
    selected_alpha: float
    vif: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def summary(self) -> str:
        """Generate a summary report string."""
        lines = [
            "=" * 60,
            "MARKETING MIX MODEL RESULTS",
            "=" * 60,
            "",
            "MODEL FIT METRICS:",
            f"  R-squared:          {self.metrics.r2:.4f}",
            f"  Adjusted R-squared: {self.metrics.adj_r2:.4f}",
            f"  RMSE:               {self.metrics.rmse:.4f}",
            f"  MAE:                {self.metrics.mae:.4f}",
            f"  MAPE:               {self.metrics.mape:.2f}%",
            f"  Durbin-Watson:      {self.metrics.durbin_watson:.4f}",
            "",
            f"REGULARIZATION: alpha = {self.selected_alpha}",
            "",
            "CHANNEL CONTRIBUTION SHARE:",
        ]

        for channel, share in self.attribution.contribution_share.items():
            lines.append(f"  {channel}: {share:.2%}")

        lines.extend(
            [
                "",
                "FEATURE COEFFICIENTS:",
            ]
        )

        for feat, coef in self.attribution.coefficients.items():
            lines.append(f"  {feat}: {coef:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class MarketingMixModel:
    """Marketing Mix Model using Ridge Regression.

    This model fits a regularized linear regression to attribute marketing
    spend impact on a target KPI. It provides:
    - Model fitting with cross-validated regularization
    - Coefficient estimation with confidence intervals
    - Channel contribution and ROAS analysis
    - Comprehensive model diagnostics

    Example:
        >>> model = MarketingMixModel()
        >>> results = model.fit(X, y)
        >>> print(results.summary())
        >>> contributions = results.attribution.contribution_share

    Attributes:
        config: ModelConfig with fitting parameters.
        model_: Fitted sklearn Ridge model.
        scaler_: Feature scaler (if scale_features=True).
        results_: ModelResults after fitting.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize the MarketingMixModel.

        Args:
            config: Model configuration. Uses defaults if None.
        """
        self.config = config or ModelConfig()
        self.model_: RidgeCV | None = None
        self.scaler_: StandardScaler | None = None
        self.results_: ModelResults | None = None
        self._feature_names: list[str] = []
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | NDArray[np.floating],
        channel_features: Sequence[str] | None = None,
    ) -> ModelResults:
        """Fit the Marketing Mix Model.

        Args:
            X: Feature DataFrame from FeatureEngineer.
            y: Target variable (KPI to model).
            channel_features: List of column names that are channel features.
                If None, infers from column names ending with '_transformed'.

        Returns:
            ModelResults with all analysis outputs.

        Raises:
            ValueError: If X and y have different lengths.
        """
        # Convert y to numpy array
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        if len(X) != len(y_arr):
            raise ValueError(f"X has {len(X)} rows but y has {len(y_arr)} values")

        self._feature_names = list(X.columns)

        # Infer channel features if not provided
        if channel_features is None:
            channel_features = [c for c in X.columns if c.endswith("_transformed")]

        # Scale features if configured
        X_scaled = self._scale_features(X)

        # Fit Ridge model with cross-validation
        self.model_ = RidgeCV(
            alphas=list(self.config.alphas),
            cv=self.config.cv_folds,
            fit_intercept=self.config.fit_intercept,
            scoring="neg_mean_squared_error",
        )
        self.model_.fit(X_scaled, y_arr)

        # Generate predictions and residuals
        predictions = self.model_.predict(X_scaled)
        residuals = y_arr - predictions

        # Calculate metrics
        metrics = self._calculate_metrics(y_arr, predictions, X_scaled.shape[1])

        # Calculate attribution
        attribution = self._calculate_attribution(
            X_scaled, y_arr, list(channel_features)
        )

        # Calculate VIF for multicollinearity
        vif = self._calculate_vif(X_scaled)

        # Feature importance (absolute standardized coefficients)
        importance = pd.Series(
            np.abs(self.model_.coef_), index=self._feature_names
        ).sort_values(ascending=False)

        self.results_ = ModelResults(
            metrics=metrics,
            attribution=attribution,
            predictions=predictions,
            residuals=residuals,
            feature_importance=importance,
            selected_alpha=float(self.model_.alpha_),
            vif=vif,
        )

        self._is_fitted = True
        return self.results_

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Generate predictions for new data.

        Args:
            X: Feature DataFrame with same columns as training data.

        Returns:
            Predicted values.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self._is_fitted or self.model_ is None:
            raise RuntimeError("Model must be fitted before predict")

        X_scaled = self._scale_features(X, fit=False)
        return self.model_.predict(X_scaled)

    def _scale_features(
        self, X: pd.DataFrame, fit: bool = True
    ) -> NDArray[np.float64]:
        """Scale features using StandardScaler if configured.

        Args:
            X: Feature DataFrame.
            fit: Whether to fit the scaler (True for training).

        Returns:
            Scaled feature array.
        """
        X_arr = X.values.astype(np.float64)

        if not self.config.scale_features:
            return X_arr

        if fit:
            self.scaler_ = StandardScaler()
            return self.scaler_.fit_transform(X_arr)

        if self.scaler_ is None:
            raise RuntimeError("Scaler not fitted")
        return self.scaler_.transform(X_arr)

    def _calculate_metrics(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        n_features: int,
    ) -> ModelMetrics:
        """Calculate comprehensive model metrics."""
        n = len(y_true)

        # Basic metrics
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        # Adjusted R-squared
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n > n_features + 1 else r2

        # MAPE (avoid division by zero)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        else:
            mape = float("inf")

        # Information criteria
        residuals = y_true - y_pred
        sse = float(np.sum(residuals**2))
        k = n_features + 1  # number of parameters including intercept

        # Log-likelihood assuming normal errors
        sigma2 = sse / n
        log_likelihood = -n / 2 * (np.log(2 * np.pi) + np.log(sigma2) + 1)

        aic = float(-2 * log_likelihood + 2 * k)
        bic = float(-2 * log_likelihood + k * np.log(n))

        # Durbin-Watson statistic for autocorrelation
        diff_residuals = np.diff(residuals)
        durbin_watson = float(np.sum(diff_residuals**2) / (np.sum(residuals**2) + 1e-10))

        return ModelMetrics(
            r2=r2,
            adj_r2=adj_r2,
            rmse=rmse,
            mae=mae,
            mape=mape,
            aic=aic,
            bic=bic,
            durbin_watson=durbin_watson,
        )

    def _calculate_attribution(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        channel_features: list[str],
    ) -> AttributionResult:
        """Calculate channel attribution and ROAS metrics."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")

        # Coefficients
        coef_series = pd.Series(self.model_.coef_, index=self._feature_names)

        # Get channel-specific analysis
        channel_indices = [self._feature_names.index(c) for c in channel_features]

        # Mean contribution = mean(X * coef) for each channel
        contributions = {}
        for i, ch in zip(channel_indices, channel_features):
            contributions[ch] = float(np.mean(X[:, i] * self.model_.coef_[i]))

        contribution_series = pd.Series(contributions).sort_values(ascending=False)

        # Contribution share (normalize positive contributions)
        pos_contrib = contribution_series.clip(lower=0)
        if pos_contrib.sum() > 0:
            share = pos_contrib / pos_contrib.sum()
        else:
            share = pd.Series(0.0, index=contribution_series.index)

        # ROAS index (relative efficiency)
        # Higher coefficient = more efficient spend
        channel_coefs = coef_series[channel_features]
        roas_index = channel_coefs / channel_coefs.max() if channel_coefs.max() > 0 else channel_coefs

        # Confidence intervals using bootstrap approximation
        ci = self._bootstrap_confidence_intervals(X, y)

        return AttributionResult(
            coefficients=coef_series.sort_values(ascending=False),
            channel_contributions=contribution_series,
            contribution_share=share.sort_values(ascending=False),
            roas_index=roas_index.sort_values(ascending=False),
            confidence_intervals=ci,
        )

    def _bootstrap_confidence_intervals(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        n_bootstrap: int = 100,
        confidence: float = 0.95,
    ) -> pd.DataFrame:
        """Calculate confidence intervals using bootstrap resampling."""
        n_samples = X.shape[0]
        n_features = X.shape[1]
        rng = np.random.default_rng(42)

        coef_samples = np.zeros((n_bootstrap, n_features))

        for i in range(n_bootstrap):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            model = RidgeCV(
                alphas=list(self.config.alphas),
                cv=min(self.config.cv_folds, len(np.unique(indices))),
                fit_intercept=self.config.fit_intercept,
            )
            model.fit(X_boot, y_boot)
            coef_samples[i] = model.coef_

        alpha = 1 - confidence
        lower = np.percentile(coef_samples, alpha / 2 * 100, axis=0)
        upper = np.percentile(coef_samples, (1 - alpha / 2) * 100, axis=0)

        return pd.DataFrame(
            {"lower": lower, "upper": upper},
            index=self._feature_names,
        )

    def _calculate_vif(self, X: NDArray[np.float64]) -> pd.Series:
        """Calculate Variance Inflation Factors for multicollinearity check.

        VIF > 5 indicates moderate multicollinearity.
        VIF > 10 indicates severe multicollinearity.
        """
        n_features = X.shape[1]
        vif_values = []

        for i in range(n_features):
            # Regress feature i on all other features
            mask = np.ones(n_features, dtype=bool)
            mask[i] = False

            X_other = X[:, mask]
            y_i = X[:, i]

            # Calculate R-squared
            if X_other.shape[1] > 0:
                corr_matrix = np.corrcoef(y_i, X_other.T)
                r_squared = 1 - 1 / (1 + np.sum(corr_matrix[0, 1:] ** 2))
                vif = 1 / (1 - r_squared + 1e-10)
            else:
                vif = 1.0

            vif_values.append(float(vif))

        return pd.Series(vif_values, index=self._feature_names)

    @property
    def coef_(self) -> NDArray[np.float64] | None:
        """Return model coefficients."""
        return self.model_.coef_ if self.model_ is not None else None

    @property
    def intercept_(self) -> float | None:
        """Return model intercept."""
        return float(self.model_.intercept_) if self.model_ is not None else None

    def __repr__(self) -> str:
        """Return string representation."""
        status = "fitted" if self._is_fitted else "not fitted"
        return f"MarketingMixModel(config={self.config}, status={status})"
