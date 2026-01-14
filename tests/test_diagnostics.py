"""
Tests for diagnostic metrics and plots.

Tests cover:
- Metric calculations
- VIF and multicollinearity checks
- Residual analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_analytics.diagnostics.metrics import (
    DiagnosticReport,
    calculate_durbin_watson,
    calculate_ljung_box,
    calculate_mape,
    calculate_model_diagnostics,
    calculate_vif,
    calculate_wmape,
    check_residual_normality,
)


class TestMAPE:
    """Tests for MAPE calculation."""

    def test_perfect_prediction(self) -> None:
        """Test MAPE for perfect prediction."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])

        mape = calculate_mape(y_true, y_pred)
        assert mape == 0.0

    def test_basic_mape(self) -> None:
        """Test basic MAPE calculation."""
        y_true = np.array([100.0, 100.0, 100.0])
        y_pred = np.array([90.0, 110.0, 100.0])

        mape = calculate_mape(y_true, y_pred)
        # Average of |10/100|, |10/100|, |0/100| = 6.67%
        assert mape == pytest.approx(6.666, rel=0.01)

    def test_mape_with_zeros(self) -> None:
        """Test MAPE skips zero actual values."""
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([10.0, 100.0, 200.0])

        mape = calculate_mape(y_true, y_pred)
        # Should only consider non-zero values
        assert np.isfinite(mape)

    def test_mape_all_zeros(self) -> None:
        """Test MAPE with all zero actual values."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([10.0, 20.0, 30.0])

        mape = calculate_mape(y_true, y_pred)
        assert mape == float("inf")


class TestWMAPE:
    """Tests for Weighted MAPE calculation."""

    def test_perfect_prediction(self) -> None:
        """Test WMAPE for perfect prediction."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])

        wmape = calculate_wmape(y_true, y_pred)
        assert wmape == 0.0

    def test_basic_wmape(self) -> None:
        """Test basic WMAPE calculation."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([90.0, 220.0])

        # Total actual = 300
        # Total error = |10| + |20| = 30
        # WMAPE = 30/300 * 100 = 10%
        wmape = calculate_wmape(y_true, y_pred)
        assert wmape == pytest.approx(10.0, rel=0.01)


class TestVIF:
    """Tests for VIF calculation."""

    def test_uncorrelated_features(self) -> None:
        """Test VIF for uncorrelated features."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "a": rng.standard_normal(100),
            "b": rng.standard_normal(100),
            "c": rng.standard_normal(100),
        })

        vif = calculate_vif(X)

        # VIF should be close to 1 for uncorrelated features
        assert all(1.0 <= v <= 2.0 for v in vif)

    def test_correlated_features(self) -> None:
        """Test VIF for correlated features."""
        rng = np.random.default_rng(42)
        x1 = rng.standard_normal(100)
        x2 = x1 + rng.standard_normal(100) * 0.1  # Highly correlated

        X = pd.DataFrame({"x1": x1, "x2": x2})

        vif = calculate_vif(X)

        # VIF should be high for correlated features
        assert any(v > 5 for v in vif)

    def test_vif_from_array(self) -> None:
        """Test VIF calculation from numpy array."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))

        vif = calculate_vif(X)
        assert len(vif) == 3


class TestDurbinWatson:
    """Tests for Durbin-Watson statistic."""

    def test_no_autocorrelation(self) -> None:
        """Test DW for residuals with no autocorrelation."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal(100)

        dw = calculate_durbin_watson(residuals)
        # Should be close to 2
        assert 1.5 < dw < 2.5

    def test_positive_autocorrelation(self) -> None:
        """Test DW for residuals with positive autocorrelation."""
        # Create positively autocorrelated series
        rng = np.random.default_rng(42)
        residuals = np.zeros(100)
        residuals[0] = rng.standard_normal()
        for i in range(1, 100):
            residuals[i] = 0.9 * residuals[i - 1] + rng.standard_normal() * 0.1

        dw = calculate_durbin_watson(residuals)
        # Should be less than 2
        assert dw < 1.5

    def test_bounds(self) -> None:
        """Test DW is in valid range [0, 4]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            residuals = rng.standard_normal(50)
            dw = calculate_durbin_watson(residuals)
            assert 0 <= dw <= 4


class TestLjungBox:
    """Tests for Ljung-Box test."""

    def test_white_noise(self) -> None:
        """Test Ljung-Box for white noise."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal(200)

        result = calculate_ljung_box(residuals, lags=10)

        assert "q_statistic" in result
        assert "p_value" in result
        # P-value should be high for white noise (no autocorrelation)
        assert result["p_value"] > 0.05

    def test_autocorrelated(self) -> None:
        """Test Ljung-Box for autocorrelated series."""
        rng = np.random.default_rng(42)
        residuals = np.zeros(200)
        residuals[0] = rng.standard_normal()
        for i in range(1, 200):
            residuals[i] = 0.9 * residuals[i - 1] + rng.standard_normal() * 0.1

        result = calculate_ljung_box(residuals, lags=10)

        # P-value should be low for autocorrelated series
        assert result["p_value"] < 0.05


class TestNormalityChecks:
    """Tests for residual normality checks."""

    def test_normal_residuals(self) -> None:
        """Test normality checks on normal data."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal(200)

        results = check_residual_normality(residuals)

        assert "jarque_bera" in results
        # Normal data should have high p-values
        assert results["jarque_bera"]["p_value"] > 0.05

    def test_non_normal_residuals(self) -> None:
        """Test normality checks on non-normal data."""
        rng = np.random.default_rng(42)
        # Uniform distribution - not normal
        residuals = rng.uniform(-1, 1, 200)

        results = check_residual_normality(residuals)

        assert "descriptive" in results
        assert "skewness" in results["descriptive"]
        assert "kurtosis" in results["descriptive"]


class TestModelDiagnostics:
    """Tests for comprehensive model diagnostics."""

    def test_full_diagnostics(self) -> None:
        """Test complete diagnostic calculation."""
        rng = np.random.default_rng(42)

        # Create simple data
        n = 100
        X = pd.DataFrame({
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        })
        y_true = 2 * X["x1"] + 3 * X["x2"] + rng.standard_normal(n) * 0.1
        y_pred = 2 * X["x1"] + 3 * X["x2"]  # Perfect model

        report = calculate_model_diagnostics(y_true, y_pred, X)

        assert isinstance(report, DiagnosticReport)
        assert "r_squared" in report.goodness_of_fit
        assert "rmse" in report.goodness_of_fit
        assert len(report.multicollinearity) == 2

    def test_summary_generation(self) -> None:
        """Test diagnostic report summary."""
        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        y_true = X["x1"] + rng.standard_normal(n) * 0.1
        y_pred = X["x1"]

        report = calculate_model_diagnostics(y_true, y_pred, X)
        summary = report.summary()

        assert "DIAGNOSTIC REPORT" in summary
        assert "GOODNESS OF FIT" in summary
        assert "VIF" in summary
