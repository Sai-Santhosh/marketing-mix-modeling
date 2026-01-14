"""
Budget optimization for Marketing Mix Modeling.

This module provides budget optimization algorithms to reallocate
marketing spend across channels for maximum efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize


if TYPE_CHECKING:
    from mmm_analytics.core.transforms import SaturationTransformer


class OptimizationMethod(str, Enum):
    """Available optimization methods."""

    GREEDY = "greedy"
    SCIPY = "scipy"
    GRADIENT = "gradient"


@dataclass
class OptimizationConstraints:
    """Constraints for budget optimization.

    Attributes:
        total_budget: Total budget to allocate (None = use current total).
        min_spend: Minimum spend per channel (dict or single value).
        max_spend: Maximum spend per channel (dict or single value).
        min_change_pct: Minimum percentage change allowed per channel.
        max_change_pct: Maximum percentage change allowed per channel.
    """

    total_budget: float | None = None
    min_spend: float | dict[str, float] = 0.0
    max_spend: float | dict[str, float] = float("inf")
    min_change_pct: float = -1.0  # -100% = can reduce to zero
    max_change_pct: float = 1.0  # +100% = can double


@dataclass
class OptimizationResult:
    """Results from budget optimization.

    Attributes:
        current_allocation: Original spend allocation.
        optimized_allocation: Recommended spend allocation.
        expected_lift: Expected percentage lift in response.
        reallocation_delta: Change in spend per channel.
        convergence_info: Optimization convergence details.
    """

    current_allocation: pd.Series
    optimized_allocation: pd.Series
    expected_lift: float
    reallocation_delta: pd.Series
    convergence_info: dict[str, float | int | str]

    def summary(self) -> str:
        """Generate optimization summary string."""
        lines = [
            "=" * 50,
            "BUDGET OPTIMIZATION RESULTS",
            "=" * 50,
            "",
            f"Expected Response Lift: {self.expected_lift:+.2%}",
            "",
            "REALLOCATION:",
            "-" * 50,
            f"{'Channel':<20} {'Current':>12} {'Optimized':>12} {'Change':>10}",
            "-" * 50,
        ]

        for channel in self.current_allocation.index:
            current = self.current_allocation[channel]
            optimized = self.optimized_allocation[channel]
            delta = self.reallocation_delta[channel]
            pct_change = delta / current * 100 if current > 0 else 0

            lines.append(
                f"{channel:<20} ${current:>11,.0f} ${optimized:>11,.0f} {pct_change:>+9.1f}%"
            )

        lines.extend(
            [
                "-" * 50,
                f"{'Total':<20} ${self.current_allocation.sum():>11,.0f} "
                f"${self.optimized_allocation.sum():>11,.0f}",
                "=" * 50,
            ]
        )

        return "\n".join(lines)


class BudgetOptimizer:
    """Optimize marketing budget allocation across channels.

    This optimizer uses response curves (from saturation functions) and
    channel coefficients to recommend budget reallocations that maximize
    expected response.

    Example:
        >>> optimizer = BudgetOptimizer(method="scipy")
        >>> result = optimizer.optimize(
        ...     current_spend=current_allocation,
        ...     coefficients=channel_coefficients,
        ...     saturation_params=sat_params,
        ...     constraints=OptimizationConstraints(total_budget=100000),
        ... )
        >>> print(result.summary())

    Attributes:
        method: Optimization algorithm to use.
        max_iterations: Maximum iterations for iterative methods.
        tolerance: Convergence tolerance.
    """

    def __init__(
        self,
        method: OptimizationMethod | str = OptimizationMethod.SCIPY,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> None:
        """Initialize BudgetOptimizer.

        Args:
            method: Optimization method to use.
            max_iterations: Maximum iterations.
            tolerance: Convergence tolerance.
        """
        if isinstance(method, str):
            method = OptimizationMethod(method)
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(
        self,
        current_spend: pd.Series,
        coefficients: pd.Series,
        saturation_params: dict[str, tuple[float, float]] | None = None,
        constraints: OptimizationConstraints | None = None,
    ) -> OptimizationResult:
        """Optimize budget allocation.

        Args:
            current_spend: Current spend by channel (Series).
            coefficients: Model coefficients by channel (higher = more efficient).
            saturation_params: Optional dict of channel -> (alpha, k) for
                Hill saturation curves. If None, uses linear response.
            constraints: Optimization constraints.

        Returns:
            OptimizationResult with recommendations.
        """
        constraints = constraints or OptimizationConstraints()
        total_budget = constraints.total_budget or current_spend.sum()

        if self.method == OptimizationMethod.GREEDY:
            optimized, info = self._greedy_optimize(
                current_spend, coefficients, total_budget, constraints
            )
        elif self.method == OptimizationMethod.SCIPY:
            optimized, info = self._scipy_optimize(
                current_spend, coefficients, saturation_params, total_budget, constraints
            )
        else:
            optimized, info = self._gradient_optimize(
                current_spend, coefficients, saturation_params, total_budget, constraints
            )

        # Calculate expected lift
        current_response = self._calculate_response(
            current_spend, coefficients, saturation_params
        )
        optimized_response = self._calculate_response(
            optimized, coefficients, saturation_params
        )
        expected_lift = (
            (optimized_response - current_response) / current_response
            if current_response > 0
            else 0.0
        )

        return OptimizationResult(
            current_allocation=current_spend,
            optimized_allocation=optimized,
            expected_lift=float(expected_lift),
            reallocation_delta=optimized - current_spend,
            convergence_info=info,
        )

    def _greedy_optimize(
        self,
        current_spend: pd.Series,
        coefficients: pd.Series,
        total_budget: float,
        constraints: OptimizationConstraints,
    ) -> tuple[pd.Series, dict[str, float | int | str]]:
        """Greedy optimization: shift budget to highest-coefficient channels."""
        spend = current_spend.copy().astype(float)
        chunk_size = total_budget * 0.01  # 1% chunks

        iterations = 0
        for _ in range(self.max_iterations):
            # Find channels to give (lowest coef with spend) and take (highest coef)
            available = spend[spend > chunk_size]
            if available.empty:
                break

            # Weight by coefficient to decide source and destination
            source = (coefficients[available.index]).idxmin()
            dest = coefficients.idxmax()

            if source == dest:
                break

            # Check constraints
            min_src = self._get_channel_constraint(
                source, constraints.min_spend, 0.0
            )
            max_dst = self._get_channel_constraint(
                dest, constraints.max_spend, float("inf")
            )

            if spend[source] - chunk_size < min_src:
                continue
            if spend[dest] + chunk_size > max_dst:
                continue

            spend[source] -= chunk_size
            spend[dest] += chunk_size
            iterations += 1

        # Normalize to total budget
        spend = spend * (total_budget / spend.sum())

        return spend, {
            "method": "greedy",
            "iterations": iterations,
            "converged": True,
        }

    def _scipy_optimize(
        self,
        current_spend: pd.Series,
        coefficients: pd.Series,
        saturation_params: dict[str, tuple[float, float]] | None,
        total_budget: float,
        constraints: OptimizationConstraints,
    ) -> tuple[pd.Series, dict[str, float | int | str]]:
        """Scipy-based optimization using SLSQP."""
        channels = list(current_spend.index)
        n_channels = len(channels)
        x0 = current_spend.values.astype(float)

        def objective(x: NDArray[np.float64]) -> float:
            spend_series = pd.Series(x, index=channels)
            # Negative because we minimize
            return -self._calculate_response(spend_series, coefficients, saturation_params)

        # Bounds
        bounds = []
        for ch in channels:
            lb = self._get_channel_constraint(ch, constraints.min_spend, 0.0)
            ub = self._get_channel_constraint(ch, constraints.max_spend, total_budget)
            bounds.append((lb, ub))

        # Budget constraint: sum = total_budget
        budget_constraint = {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[budget_constraint],
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )

        optimized = pd.Series(result.x, index=channels)

        return optimized, {
            "method": "scipy_slsqp",
            "iterations": int(result.nit),
            "converged": result.success,
            "message": str(result.message),
        }

    def _gradient_optimize(
        self,
        current_spend: pd.Series,
        coefficients: pd.Series,
        saturation_params: dict[str, tuple[float, float]] | None,
        total_budget: float,
        constraints: OptimizationConstraints,
    ) -> tuple[pd.Series, dict[str, float | int | str]]:
        """Gradient-based optimization with projected gradient descent."""
        channels = list(current_spend.index)
        spend = current_spend.values.astype(float)
        learning_rate = total_budget * 0.01

        for iteration in range(self.max_iterations):
            # Calculate gradient (marginal response per channel)
            grad = np.zeros_like(spend)
            for i, ch in enumerate(channels):
                if saturation_params and ch in saturation_params:
                    alpha, k = saturation_params[ch]
                    # Derivative of Hill function * coefficient
                    x = spend[i]
                    x_alpha = x**alpha
                    k_alpha = k**alpha
                    marginal = alpha * k_alpha * (x ** (alpha - 1)) / ((x_alpha + k_alpha) ** 2)
                    grad[i] = coefficients[ch] * marginal
                else:
                    grad[i] = coefficients[ch]

            # Gradient ascent step
            spend_new = spend + learning_rate * grad

            # Project to constraints
            spend_new = np.maximum(spend_new, 0)
            for i, ch in enumerate(channels):
                lb = self._get_channel_constraint(ch, constraints.min_spend, 0.0)
                ub = self._get_channel_constraint(ch, constraints.max_spend, total_budget)
                spend_new[i] = np.clip(spend_new[i], lb, ub)

            # Normalize to budget
            spend_new = spend_new * (total_budget / (spend_new.sum() + 1e-10))

            # Check convergence
            if np.max(np.abs(spend_new - spend)) < self.tolerance:
                spend = spend_new
                break

            spend = spend_new

        return pd.Series(spend, index=channels), {
            "method": "gradient",
            "iterations": iteration + 1,
            "converged": iteration < self.max_iterations - 1,
        }

    def _calculate_response(
        self,
        spend: pd.Series,
        coefficients: pd.Series,
        saturation_params: dict[str, tuple[float, float]] | None,
    ) -> float:
        """Calculate expected response for a given spend allocation."""
        response = 0.0

        for channel in spend.index:
            x = spend[channel]
            coef = coefficients.get(channel, 0.0)

            if saturation_params and channel in saturation_params:
                alpha, k = saturation_params[channel]
                # Hill function
                x_alpha = x**alpha
                k_alpha = k**alpha
                saturated = x_alpha / (x_alpha + k_alpha + 1e-10)
                response += coef * saturated
            else:
                response += coef * x

        return float(response)

    @staticmethod
    def _get_channel_constraint(
        channel: str,
        constraint: float | dict[str, float],
        default: float,
    ) -> float:
        """Get constraint value for a specific channel."""
        if isinstance(constraint, dict):
            return constraint.get(channel, default)
        return constraint

    def __repr__(self) -> str:
        """Return string representation."""
        return f"BudgetOptimizer(method={self.method.value})"
