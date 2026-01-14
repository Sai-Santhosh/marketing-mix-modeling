"""
Tests for media transformation functions.

Tests cover:
- Adstock transformation correctness and properties
- Saturation transformation bounds and behavior
- Edge cases and input validation
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mmm_analytics.core.transforms import (
    AdstockParams,
    AdstockTransformer,
    SaturationParams,
    SaturationTransformer,
    apply_adstock,
    apply_saturation,
)


class TestAdstockParams:
    """Tests for AdstockParams dataclass."""

    def test_valid_params(self) -> None:
        """Test valid parameter creation."""
        params = AdstockParams(decay=0.5, max_lag=8)
        assert params.decay == 0.5
        assert params.max_lag == 8

    def test_invalid_decay_too_high(self) -> None:
        """Test that decay > 1 raises error."""
        with pytest.raises(ValueError, match="decay must be in"):
            AdstockParams(decay=1.5)

    def test_invalid_decay_negative(self) -> None:
        """Test that negative decay raises error."""
        with pytest.raises(ValueError, match="decay must be in"):
            AdstockParams(decay=-0.1)

    def test_invalid_max_lag(self) -> None:
        """Test that max_lag < 1 raises error."""
        with pytest.raises(ValueError, match="max_lag must be"):
            AdstockParams(decay=0.5, max_lag=0)


class TestAdstockTransformer:
    """Tests for AdstockTransformer."""

    def test_basic_transform(self) -> None:
        """Test basic adstock transformation."""
        transformer = AdstockTransformer(decay=0.5)
        x = np.array([100.0, 0.0, 0.0, 0.0])
        result = transformer.transform(x)

        assert result[0] == 100.0
        assert result[1] == pytest.approx(50.0)
        assert result[2] == pytest.approx(25.0)
        assert result[3] == pytest.approx(12.5)

    def test_zero_decay(self) -> None:
        """Test that zero decay returns input unchanged."""
        transformer = AdstockTransformer(decay=0.0)
        x = np.array([100.0, 50.0, 25.0])
        result = transformer.transform(x)

        np.testing.assert_array_equal(result, x)

    def test_full_decay(self) -> None:
        """Test that decay=1 accumulates fully."""
        transformer = AdstockTransformer(decay=1.0)
        x = np.array([100.0, 100.0, 100.0])
        result = transformer.transform(x)

        assert result[0] == 100.0
        assert result[1] == 200.0
        assert result[2] == 300.0

    def test_monotone_decay(self) -> None:
        """Test that adstock decays monotonically after impulse."""
        transformer = AdstockTransformer(decay=0.5)
        x = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
        result = transformer.transform(x)

        # After initial impulse, values should decrease
        assert result[1] < result[0]
        assert result[2] < result[1]
        assert result[3] < result[2]

    def test_empty_input(self) -> None:
        """Test handling of empty input."""
        transformer = AdstockTransformer(decay=0.5)
        result = transformer.transform(np.array([]))
        assert len(result) == 0

    def test_negative_input_raises(self) -> None:
        """Test that negative input raises error."""
        transformer = AdstockTransformer(decay=0.5)
        with pytest.raises(ValueError, match="non-negative"):
            transformer.transform(np.array([100.0, -50.0, 25.0]))

    def test_2d_input_raises(self) -> None:
        """Test that 2D input raises error."""
        transformer = AdstockTransformer(decay=0.5)
        with pytest.raises(ValueError, match="1D array"):
            transformer.transform(np.array([[100.0], [50.0]]))

    def test_carryover_weights(self) -> None:
        """Test carryover weight calculation."""
        transformer = AdstockTransformer(decay=0.5, max_lag=4)
        weights = transformer.get_carryover_weights()

        expected = np.array([1.0, 0.5, 0.25, 0.125])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_inverse_transform(self) -> None:
        """Test that inverse approximately recovers original."""
        transformer = AdstockTransformer(decay=0.5)
        original = np.array([100.0, 200.0, 150.0, 100.0])
        adstocked = transformer.transform(original)
        recovered = transformer.inverse_transform(adstocked)

        np.testing.assert_array_almost_equal(recovered, original, decimal=10)

    @given(
        x=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=20),
            elements=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        ),
        decay=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=20, deadline=None)
    def test_adstock_properties(self, x: np.ndarray, decay: float) -> None:
        """Property-based test: adstock output >= input (for non-negative)."""
        transformer = AdstockTransformer(decay=decay)
        result = transformer.transform(x)

        # Adstock should never be less than original spend
        assert np.all(result >= x)

    def test_repr(self) -> None:
        """Test string representation."""
        transformer = AdstockTransformer(decay=0.6, max_lag=10)
        assert "0.6" in repr(transformer)
        assert "10" in repr(transformer)


class TestSaturationParams:
    """Tests for SaturationParams dataclass."""

    def test_valid_params(self) -> None:
        """Test valid parameter creation."""
        params = SaturationParams(alpha=2.0, k=500.0)
        assert params.alpha == 2.0
        assert params.k == 500.0

    def test_invalid_alpha(self) -> None:
        """Test that alpha <= 0 raises error."""
        with pytest.raises(ValueError, match="alpha must be > 0"):
            SaturationParams(alpha=0.0)

    def test_invalid_k(self) -> None:
        """Test that k <= 0 raises error."""
        with pytest.raises(ValueError, match="k must be > 0"):
            SaturationParams(alpha=2.0, k=0.0)


class TestSaturationTransformer:
    """Tests for SaturationTransformer."""

    def test_basic_transform(self) -> None:
        """Test basic saturation transformation."""
        transformer = SaturationTransformer(alpha=2.0, k=500.0)
        x = np.array([0.0, 500.0, 1000.0])
        result = transformer.transform(x)

        # At x=0, output should be 0
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        # At x=k, output should be 0.5 (half-saturation)
        assert result[1] == pytest.approx(0.5, rel=0.01)
        # At x=2k, output should be > 0.5 but < 1
        assert 0.5 < result[2] < 1.0

    def test_bounds(self) -> None:
        """Test that output is bounded in [0, 1]."""
        transformer = SaturationTransformer(alpha=2.0, k=500.0)
        x = np.linspace(0, 10000, 100)
        result = transformer.transform(x)

        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_monotone_increasing(self) -> None:
        """Test that saturation is monotonically increasing."""
        transformer = SaturationTransformer(alpha=2.0, k=500.0)
        x = np.linspace(0, 2000, 50)
        result = transformer.transform(x)

        # Check monotonicity
        diff = np.diff(result)
        assert np.all(diff >= 0)

    def test_empty_input(self) -> None:
        """Test handling of empty input."""
        transformer = SaturationTransformer(alpha=2.0, k=500.0)
        result = transformer.transform(np.array([]))
        assert len(result) == 0

    def test_negative_input_raises(self) -> None:
        """Test that negative input raises error."""
        transformer = SaturationTransformer(alpha=2.0, k=500.0)
        with pytest.raises(ValueError, match="non-negative"):
            transformer.transform(np.array([100.0, -50.0]))

    def test_ec50(self) -> None:
        """Test EC50 calculation."""
        transformer = SaturationTransformer(alpha=2.0, k=500.0)
        assert transformer.get_ec50() == 500.0

    def test_ec90(self) -> None:
        """Test EC90 calculation."""
        transformer = SaturationTransformer(alpha=2.0, k=500.0)
        ec90 = transformer.get_ec90()

        # Verify that at EC90, response is approximately 0.9
        response = transformer.transform(np.array([ec90]))[0]
        assert response == pytest.approx(0.9, rel=0.01)

    def test_marginal_response(self) -> None:
        """Test marginal response calculation."""
        transformer = SaturationTransformer(alpha=2.0, k=500.0)
        x = np.array([100.0, 500.0, 1000.0])
        marginal = transformer.marginal_response(x)

        # Marginal response should be positive
        assert np.all(marginal > 0)
        # For alpha=2.0, marginal response peaks at k and decreases after
        # At x < k, it's increasing; at x > k, it's decreasing
        # So at 500 (k), it's at peak, then decreases at 1000
        assert marginal[1] > marginal[2]  # Diminishing returns after k

    def test_inverse_transform(self) -> None:
        """Test that inverse approximately recovers original."""
        transformer = SaturationTransformer(alpha=2.0, k=500.0)
        original = np.array([100.0, 300.0, 500.0, 800.0])
        saturated = transformer.transform(original)
        recovered = transformer.inverse_transform(saturated)

        np.testing.assert_array_almost_equal(recovered, original, decimal=5)

    @given(
        x=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=50),
            elements=st.floats(min_value=0, max_value=5000, allow_nan=False, allow_infinity=False),
        ),
        alpha=st.floats(min_value=0.1, max_value=5.0),
        k=st.floats(min_value=1.0, max_value=1000.0),
    )
    @settings(max_examples=50)
    def test_saturation_bounds_property(
        self, x: np.ndarray, alpha: float, k: float
    ) -> None:
        """Property-based test: saturation always in [0, 1]."""
        transformer = SaturationTransformer(alpha=alpha, k=k)
        result = transformer.transform(x)

        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_repr(self) -> None:
        """Test string representation."""
        transformer = SaturationTransformer(alpha=1.5, k=750.0)
        assert "1.5" in repr(transformer)
        assert "750" in repr(transformer)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_apply_adstock(self) -> None:
        """Test apply_adstock convenience function."""
        x = np.array([100.0, 0.0, 0.0])
        result = apply_adstock(x, decay=0.5)

        assert result[0] == 100.0
        assert result[1] == pytest.approx(50.0)

    def test_apply_saturation(self) -> None:
        """Test apply_saturation convenience function."""
        x = np.array([0.0, 500.0])
        result = apply_saturation(x, alpha=2.0, k=500.0)

        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[1] == pytest.approx(0.5, rel=0.01)
