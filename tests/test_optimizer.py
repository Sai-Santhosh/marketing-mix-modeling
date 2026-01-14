"""
Tests for budget optimization module.

Tests cover:
- Different optimization methods
- Constraint handling
- Budget conservation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_analytics.core.optimizer import (
    BudgetOptimizer,
    OptimizationConstraints,
    OptimizationMethod,
    OptimizationResult,
)


class TestOptimizationConstraints:
    """Tests for OptimizationConstraints."""

    def test_default_constraints(self) -> None:
        """Test default constraint values."""
        constraints = OptimizationConstraints()
        assert constraints.total_budget is None
        assert constraints.min_spend == 0.0
        assert constraints.max_spend == float("inf")

    def test_custom_constraints(self) -> None:
        """Test custom constraint setting."""
        constraints = OptimizationConstraints(
            total_budget=100000,
            min_spend={"search": 5000, "social": 3000},
            max_spend=50000,
        )
        assert constraints.total_budget == 100000


class TestBudgetOptimizer:
    """Tests for BudgetOptimizer."""

    @pytest.fixture
    def sample_spend(self) -> pd.Series:
        """Sample current spend allocation."""
        return pd.Series({
            "search": 10000.0,
            "social": 8000.0,
            "display": 6000.0,
            "audio": 4000.0,
        })

    @pytest.fixture
    def sample_coefficients(self) -> pd.Series:
        """Sample channel coefficients."""
        return pd.Series({
            "search": 1.2,
            "social": 0.8,
            "display": 0.4,
            "audio": 0.6,
        })

    def test_greedy_optimization(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test greedy optimization method."""
        optimizer = BudgetOptimizer(method=OptimizationMethod.GREEDY)
        result = optimizer.optimize(sample_spend, sample_coefficients)

        assert isinstance(result, OptimizationResult)
        # Budget should be conserved
        assert result.optimized_allocation.sum() == pytest.approx(
            sample_spend.sum(), rel=0.01
        )

    def test_scipy_optimization(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test scipy optimization method."""
        optimizer = BudgetOptimizer(method="scipy")
        result = optimizer.optimize(sample_spend, sample_coefficients)

        assert isinstance(result, OptimizationResult)
        assert result.optimized_allocation.sum() == pytest.approx(
            sample_spend.sum(), rel=0.01
        )

    def test_gradient_optimization(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test gradient optimization method."""
        optimizer = BudgetOptimizer(method=OptimizationMethod.GRADIENT)
        result = optimizer.optimize(sample_spend, sample_coefficients)

        assert isinstance(result, OptimizationResult)
        assert result.optimized_allocation.sum() == pytest.approx(
            sample_spend.sum(), rel=0.01
        )

    def test_budget_conservation(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test that total budget is conserved."""
        for method in OptimizationMethod:
            optimizer = BudgetOptimizer(method=method)
            result = optimizer.optimize(sample_spend, sample_coefficients)

            original_total = sample_spend.sum()
            optimized_total = result.optimized_allocation.sum()
            assert optimized_total == pytest.approx(original_total, rel=0.01)

    def test_custom_total_budget(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test optimization with custom total budget."""
        optimizer = BudgetOptimizer(method="scipy")
        constraints = OptimizationConstraints(total_budget=50000)

        result = optimizer.optimize(
            sample_spend, sample_coefficients, constraints=constraints
        )

        assert result.optimized_allocation.sum() == pytest.approx(50000, rel=0.01)

    def test_min_spend_constraints(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test minimum spend constraints."""
        optimizer = BudgetOptimizer(method="scipy")
        constraints = OptimizationConstraints(
            min_spend={"audio": 5000},  # Higher than current
        )

        result = optimizer.optimize(
            sample_spend, sample_coefficients, constraints=constraints
        )

        # Audio spend should be at least 5000
        assert result.optimized_allocation["audio"] >= 4999  # Allow small tolerance

    def test_max_spend_constraints(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test maximum spend constraints."""
        optimizer = BudgetOptimizer(method="scipy")
        constraints = OptimizationConstraints(
            max_spend={"search": 8000},  # Lower than current
        )

        result = optimizer.optimize(
            sample_spend, sample_coefficients, constraints=constraints
        )

        # Search spend should be at most 8000
        assert result.optimized_allocation["search"] <= 8001  # Allow small tolerance

    def test_with_saturation_params(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test optimization with saturation parameters."""
        optimizer = BudgetOptimizer(method="scipy")
        sat_params = {
            "search": (2.0, 500.0),
            "social": (2.0, 400.0),
            "display": (2.0, 300.0),
            "audio": (2.0, 200.0),
        }

        result = optimizer.optimize(
            sample_spend,
            sample_coefficients,
            saturation_params=sat_params,
        )

        assert isinstance(result, OptimizationResult)

    def test_expected_lift_calculated(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test that expected lift is calculated."""
        optimizer = BudgetOptimizer(method="scipy")
        result = optimizer.optimize(sample_spend, sample_coefficients)

        # Lift should be a number
        assert isinstance(result.expected_lift, float)

    def test_reallocation_delta(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test reallocation delta calculation."""
        optimizer = BudgetOptimizer(method="scipy")
        result = optimizer.optimize(sample_spend, sample_coefficients)

        # Delta should equal optimized - current
        expected_delta = result.optimized_allocation - result.current_allocation
        pd.testing.assert_series_equal(result.reallocation_delta, expected_delta)

    def test_convergence_info(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test convergence information is returned."""
        optimizer = BudgetOptimizer(method="scipy")
        result = optimizer.optimize(sample_spend, sample_coefficients)

        assert "method" in result.convergence_info
        assert "converged" in result.convergence_info

    def test_summary_string(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test summary string generation."""
        optimizer = BudgetOptimizer(method="scipy")
        result = optimizer.optimize(sample_spend, sample_coefficients)

        summary = result.summary()
        assert "BUDGET OPTIMIZATION" in summary
        assert "Expected Response Lift" in summary
        assert "Current" in summary
        assert "Optimized" in summary

    def test_non_negative_allocation(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test that all allocations are non-negative."""
        for method in OptimizationMethod:
            optimizer = BudgetOptimizer(method=method)
            result = optimizer.optimize(sample_spend, sample_coefficients)

            assert all(result.optimized_allocation >= 0)

    def test_shifts_to_higher_coefficient(
        self, sample_spend: pd.Series, sample_coefficients: pd.Series
    ) -> None:
        """Test that optimization shifts budget to higher-coefficient channels."""
        optimizer = BudgetOptimizer(method="scipy")
        result = optimizer.optimize(sample_spend, sample_coefficients)

        # Best channel (search with coef 1.2) should get more
        # Worst channel (display with coef 0.4) should get less
        search_delta = result.reallocation_delta["search"]
        display_delta = result.reallocation_delta["display"]

        # Search should increase relative to display
        assert search_delta > display_delta

    def test_repr(self) -> None:
        """Test string representation."""
        optimizer = BudgetOptimizer(method="scipy")
        assert "scipy" in repr(optimizer)
