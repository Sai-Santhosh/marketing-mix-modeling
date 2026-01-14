"""
Statistical metrics and diagnostics for MMM evaluation.

This module provides comprehensive diagnostic metrics for evaluating
Marketing Mix Models, including goodness-of-fit, multicollinearity,
and residual analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for model evaluation.

    Attributes:
        goodness_of_fit: Dictionary of fit metrics (R², RMSE, etc.).
        residual_analysis: Dictionary of residual diagnostics.
        multicollinearity: VIF values for each feature.
        normality_tests: Normality test results for residuals.
        autocorrelation: Autocorrelation metrics.
    """

    goodness_of_fit: dict[str, float]
    residual_analysis: dict[str, float]
    multicollinearity: pd.Series
    normality_tests: dict[str, Any]
    autocorrelation: dict[str, float]

    def summary(self) -> str:
        """Generate diagnostic summary string."""
        lines = [
            "=" * 60,
            "MODEL DIAGNOSTIC REPORT",
            "=" * 60,
            "",
            "GOODNESS OF FIT:",
        ]

        for metric, value in self.goodness_of_fit.items():
            lines.append(f"  {metric}: {value:.4f}")

        lines.extend(["", "RESIDUAL ANALYSIS:"])
        for metric, value in self.residual_analysis.items():
            lines.append(f"  {metric}: {value:.4f}")

        lines.extend(["", "AUTOCORRELATION:"])
        for metric, value in self.autocorrelation.items():
            lines.append(f"  {metric}: {value:.4f}")

        lines.extend(["", "NORMALITY TESTS:"])
        for test_name, result in self.normality_tests.items():
            if isinstance(result, dict) and "p_value" in result:
                pval = result["p_value"]
                lines.append(f"  {test_name}: p-value = {pval:.4f}")

        lines.extend(["", "MULTICOLLINEARITY (VIF):"])
        for feat, vif_val in self.multicollinearity.items():
            flag = " ⚠️" if vif_val > 5 else ""
            lines.append(f"  {feat}: {vif_val:.2f}{flag}")

        lines.append("")
        lines.append("VIF > 5 indicates moderate multicollinearity")
        lines.append("VIF > 10 indicates severe multicollinearity")
        lines.append("=" * 60)

        return "\n".join(lines)


def calculate_mape(
    y_true: NDArray[np.floating] | pd.Series,
    y_pred: NDArray[np.floating] | pd.Series,
) -> float:
    """Calculate Mean Absolute Percentage Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        MAPE as a percentage (0-100).
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    # Avoid division by zero
    mask = y_true_arr != 0
    if mask.sum() == 0:
        return float("inf")

    return float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100)


def calculate_wmape(
    y_true: NDArray[np.floating] | pd.Series,
    y_pred: NDArray[np.floating] | pd.Series,
) -> float:
    """Calculate Weighted Mean Absolute Percentage Error.

    This variant weights errors by the magnitude of actual values,
    giving more importance to larger values.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        WMAPE as a percentage.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    total_actual = np.sum(np.abs(y_true_arr))
    if total_actual == 0:
        return float("inf")

    return float(np.sum(np.abs(y_true_arr - y_pred_arr)) / total_actual * 100)


def calculate_vif(X: pd.DataFrame | NDArray[np.floating]) -> pd.Series:
    """Calculate Variance Inflation Factor for multicollinearity detection.

    VIF measures how much the variance of an estimated regression coefficient
    is increased due to collinearity.

    Interpretation:
    - VIF = 1: No correlation
    - VIF < 5: Low correlation (acceptable)
    - VIF 5-10: Moderate correlation (concerning)
    - VIF > 10: High correlation (problematic)

    Args:
        X: Feature matrix (DataFrame or array).

    Returns:
        Series with VIF for each feature.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = X.values.astype(np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

    n_features = X_arr.shape[1]
    vif_values = []

    for i in range(n_features):
        # Get column i and all other columns
        y_i = X_arr[:, i]
        X_other = np.delete(X_arr, i, axis=1)

        if X_other.shape[1] == 0:
            vif_values.append(1.0)
            continue

        # Fit regression of feature i on all others
        X_other_with_const = np.column_stack([np.ones(len(y_i)), X_other])

        try:
            # Solve least squares
            coef, residuals, rank, s = np.linalg.lstsq(X_other_with_const, y_i, rcond=None)
            y_pred = X_other_with_const @ coef

            # Calculate R-squared
            ss_res = np.sum((y_i - y_pred) ** 2)
            ss_tot = np.sum((y_i - y_i.mean()) ** 2)

            if ss_tot == 0:
                vif = float("inf")
            else:
                r_squared = 1 - ss_res / ss_tot
                vif = 1 / (1 - r_squared) if r_squared < 1 else float("inf")
        except np.linalg.LinAlgError:
            vif = float("inf")

        vif_values.append(vif)

    return pd.Series(vif_values, index=feature_names)


def calculate_durbin_watson(residuals: NDArray[np.floating]) -> float:
    """Calculate Durbin-Watson statistic for autocorrelation.

    The Durbin-Watson statistic tests for first-order autocorrelation
    in regression residuals.

    Interpretation:
    - DW ≈ 2: No autocorrelation
    - DW < 2: Positive autocorrelation
    - DW > 2: Negative autocorrelation
    - DW < 1 or DW > 3: Strong autocorrelation (concerning)

    Args:
        residuals: Model residuals.

    Returns:
        Durbin-Watson statistic (0 to 4).
    """
    residuals_arr = np.asarray(residuals, dtype=np.float64)
    diff = np.diff(residuals_arr)
    return float(np.sum(diff**2) / (np.sum(residuals_arr**2) + 1e-10))


def calculate_ljung_box(
    residuals: NDArray[np.floating],
    lags: int = 10,
) -> dict[str, float]:
    """Calculate Ljung-Box test for autocorrelation.

    Tests the null hypothesis that residuals are independently distributed
    (no autocorrelation).

    Args:
        residuals: Model residuals.
        lags: Number of lags to test.

    Returns:
        Dictionary with test statistic and p-value.
    """
    residuals_arr = np.asarray(residuals, dtype=np.float64)
    n = len(residuals_arr)

    # Calculate autocorrelations
    autocorr = []
    for k in range(1, lags + 1):
        if k >= n:
            break
        r = np.corrcoef(residuals_arr[:-k], residuals_arr[k:])[0, 1]
        autocorr.append(r if not np.isnan(r) else 0.0)

    # Ljung-Box Q statistic
    q_stat = 0.0
    for k, r in enumerate(autocorr, 1):
        q_stat += (r**2) / (n - k)
    q_stat = n * (n + 2) * q_stat

    # P-value from chi-squared distribution
    p_value = 1 - stats.chi2.cdf(q_stat, df=len(autocorr))

    return {
        "q_statistic": float(q_stat),
        "p_value": float(p_value),
        "lags": len(autocorr),
    }


def check_residual_normality(
    residuals: NDArray[np.floating],
) -> dict[str, dict[str, float]]:
    """Check residuals for normality using multiple tests.

    Args:
        residuals: Model residuals.

    Returns:
        Dictionary with results from multiple normality tests.
    """
    residuals_arr = np.asarray(residuals, dtype=np.float64)
    results = {}

    # Shapiro-Wilk test (best for small samples)
    if len(residuals_arr) <= 5000:
        stat, p_value = stats.shapiro(residuals_arr)
        results["shapiro_wilk"] = {
            "statistic": float(stat),
            "p_value": float(p_value),
        }

    # Jarque-Bera test (based on skewness and kurtosis)
    stat, p_value = stats.jarque_bera(residuals_arr)
    results["jarque_bera"] = {
        "statistic": float(stat),
        "p_value": float(p_value),
    }

    # D'Agostino's K-squared test
    if len(residuals_arr) >= 20:
        stat, p_value = stats.normaltest(residuals_arr)
        results["dagostino_k2"] = {
            "statistic": float(stat),
            "p_value": float(p_value),
        }

    # Skewness and kurtosis
    results["descriptive"] = {
        "skewness": float(stats.skew(residuals_arr)),
        "kurtosis": float(stats.kurtosis(residuals_arr)),
        "mean": float(np.mean(residuals_arr)),
        "std": float(np.std(residuals_arr)),
    }

    return results


def calculate_model_diagnostics(
    y_true: NDArray[np.floating] | pd.Series,
    y_pred: NDArray[np.floating] | pd.Series,
    X: pd.DataFrame | NDArray[np.floating],
) -> DiagnosticReport:
    """Calculate comprehensive model diagnostics.

    Args:
        y_true: Actual target values.
        y_pred: Model predictions.
        X: Feature matrix.

    Returns:
        DiagnosticReport with all diagnostics.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    residuals = y_true_arr - y_pred_arr

    # Goodness of fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true_arr - y_true_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n = len(y_true_arr)
    p = X.shape[1] if hasattr(X, "shape") else len(X.columns)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    goodness_of_fit = {
        "r_squared": r2,
        "adj_r_squared": adj_r2,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "mape": calculate_mape(y_true_arr, y_pred_arr),
        "wmape": calculate_wmape(y_true_arr, y_pred_arr),
    }

    # Residual analysis
    residual_analysis = {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "skewness": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
    }

    # Multicollinearity
    vif = calculate_vif(X)

    # Normality tests
    normality = check_residual_normality(residuals)

    # Autocorrelation
    autocorrelation = {
        "durbin_watson": calculate_durbin_watson(residuals),
        **calculate_ljung_box(residuals),
    }

    return DiagnosticReport(
        goodness_of_fit=goodness_of_fit,
        residual_analysis=residual_analysis,
        multicollinearity=vif,
        normality_tests=normality,
        autocorrelation=autocorrelation,
    )
