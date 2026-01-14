"""
Media transformation functions for Marketing Mix Modeling.

This module implements the core transformations applied to media spend data:
- Adstock (geometric decay): Captures the carryover effect of advertising
- Saturation (Hill function): Models diminishing returns at high spend levels

These transformations are fundamental to MMM as they account for the delayed
and non-linear effects of advertising on business outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class AdstockParams:
    """Parameters for adstock transformation.

    Attributes:
        decay: Decay rate (0-1). Higher values = longer carryover effect.
            Typical range: 0.3-0.8 depending on media channel.
        max_lag: Maximum lag periods to consider for memory efficiency.
    """

    decay: float = 0.5
    max_lag: int = 8

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {self.decay}")
        if self.max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {self.max_lag}")


@dataclass(frozen=True, slots=True)
class SaturationParams:
    """Parameters for Hill saturation transformation.

    Attributes:
        alpha: Shape parameter controlling curve steepness.
            alpha > 1: S-curve (slow start, rapid growth, plateau)
            alpha = 1: Standard Michaelis-Menten curve
            alpha < 1: Concave curve (rapid initial growth)
        k: Half-saturation point (inflection point for S-curve).
            The spend level at which response is 50% of maximum.
    """

    alpha: float = 2.0
    k: float = 500.0

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.k <= 0:
            raise ValueError(f"k must be > 0, got {self.k}")


class AdstockTransformer:
    """Apply geometric adstock transformation to media spend data.

    The adstock model captures the carryover effect of advertising, where
    the impact of an ad exposure persists and decays over time.

    Mathematical formulation:
        A(t) = X(t) + decay * A(t-1)

    Where:
        - A(t) is the adstocked value at time t
        - X(t) is the raw spend at time t
        - decay is the decay rate (0-1)

    Example:
        >>> transformer = AdstockTransformer(decay=0.6)
        >>> spend = np.array([100, 0, 0, 0, 0])
        >>> adstocked = transformer.transform(spend)
        >>> print(adstocked)  # [100.0, 60.0, 36.0, 21.6, 12.96]

    Attributes:
        params: AdstockParams containing decay and max_lag settings.
    """

    def __init__(self, decay: float = 0.5, max_lag: int = 8) -> None:
        """Initialize the AdstockTransformer.

        Args:
            decay: Decay rate between 0 and 1. Default is 0.5.
            max_lag: Maximum lag periods for memory efficiency. Default is 8.

        Raises:
            ValueError: If decay is not in [0, 1] or max_lag < 1.
        """
        self.params = AdstockParams(decay=decay, max_lag=max_lag)

    @property
    def decay(self) -> float:
        """Return the decay rate."""
        return self.params.decay

    @property
    def max_lag(self) -> int:
        """Return the maximum lag."""
        return self.params.max_lag

    def transform(self, x: NDArray[np.floating] | Sequence[float]) -> NDArray[np.float64]:
        """Apply geometric adstock transformation.

        Args:
            x: Array-like of media spend values (must be non-negative).

        Returns:
            Transformed array with adstock effect applied.

        Raises:
            ValueError: If input contains negative values.
        """
        x_arr = np.asarray(x, dtype=np.float64)

        if x_arr.ndim != 1:
            raise ValueError(f"Expected 1D array, got {x_arr.ndim}D")
        if np.any(x_arr < 0):
            raise ValueError("Input values must be non-negative")

        n = len(x_arr)
        if n == 0:
            return np.array([], dtype=np.float64)

        result = np.zeros(n, dtype=np.float64)
        carry = 0.0

        for i in range(n):
            carry = x_arr[i] + self.decay * carry
            result[i] = carry

        return result

    def inverse_transform(self, a: NDArray[np.floating]) -> NDArray[np.float64]:
        """Recover original spend from adstocked values (approximate).

        Note: This is only exact if the original series was transformed
        with the same parameters.

        Args:
            a: Adstocked values.

        Returns:
            Approximate original spend values.
        """
        a_arr = np.asarray(a, dtype=np.float64)
        n = len(a_arr)
        if n == 0:
            return np.array([], dtype=np.float64)

        result = np.zeros(n, dtype=np.float64)
        result[0] = a_arr[0]

        for i in range(1, n):
            result[i] = a_arr[i] - self.decay * a_arr[i - 1]

        return np.maximum(result, 0.0)

    def get_carryover_weights(self, n_periods: int | None = None) -> NDArray[np.float64]:
        """Get the decay weights for each lag period.

        Args:
            n_periods: Number of periods to compute. Defaults to max_lag.

        Returns:
            Array of weights [1, decay, decay^2, ..., decay^(n-1)].
        """
        n = n_periods or self.max_lag
        return np.array([self.decay**i for i in range(n)], dtype=np.float64)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AdstockTransformer(decay={self.decay}, max_lag={self.max_lag})"


class SaturationTransformer:
    """Apply Hill saturation transformation to media spend data.

    The Hill function models diminishing returns, where additional spend
    yields progressively smaller incremental effects.

    Mathematical formulation (Hill function):
        S(x) = x^alpha / (x^alpha + k^alpha)

    Where:
        - S(x) is the saturated value (bounded between 0 and 1)
        - x is the input spend (typically after adstock transformation)
        - alpha is the shape parameter
        - k is the half-saturation point

    Example:
        >>> transformer = SaturationTransformer(alpha=2.0, k=500.0)
        >>> spend = np.array([0, 250, 500, 750, 1000])
        >>> saturated = transformer.transform(spend)
        >>> print(saturated)  # [0.0, 0.2, 0.5, 0.692, 0.8]

    Attributes:
        params: SaturationParams containing alpha and k settings.
    """

    def __init__(self, alpha: float = 2.0, k: float = 500.0) -> None:
        """Initialize the SaturationTransformer.

        Args:
            alpha: Shape parameter (> 0). Default is 2.0.
            k: Half-saturation point (> 0). Default is 500.0.

        Raises:
            ValueError: If alpha <= 0 or k <= 0.
        """
        self.params = SaturationParams(alpha=alpha, k=k)

    @property
    def alpha(self) -> float:
        """Return the shape parameter."""
        return self.params.alpha

    @property
    def k(self) -> float:
        """Return the half-saturation point."""
        return self.params.k

    def transform(self, x: NDArray[np.floating] | Sequence[float]) -> NDArray[np.float64]:
        """Apply Hill saturation transformation.

        Args:
            x: Array-like of values to transform (must be non-negative).

        Returns:
            Transformed array with values in [0, 1].

        Raises:
            ValueError: If input contains negative values.
        """
        x_arr = np.asarray(x, dtype=np.float64)

        if x_arr.ndim != 1:
            raise ValueError(f"Expected 1D array, got {x_arr.ndim}D")
        if np.any(x_arr < 0):
            raise ValueError("Input values must be non-negative")

        if len(x_arr) == 0:
            return np.array([], dtype=np.float64)

        # Hill function with numerical stability
        x_alpha = np.power(x_arr, self.alpha)
        k_alpha = self.k**self.alpha

        return x_alpha / (x_alpha + k_alpha + 1e-10)

    def inverse_transform(self, s: NDArray[np.floating]) -> NDArray[np.float64]:
        """Recover original values from saturated output.

        Args:
            s: Saturated values in (0, 1).

        Returns:
            Approximate original values.
        """
        s_arr = np.asarray(s, dtype=np.float64)

        # Clip to valid range (0, 1)
        s_clipped = np.clip(s_arr, 1e-10, 1 - 1e-10)

        # Inverse Hill: x = k * (s / (1 - s))^(1/alpha)
        return self.k * np.power(s_clipped / (1 - s_clipped), 1 / self.alpha)

    def marginal_response(self, x: NDArray[np.floating]) -> NDArray[np.float64]:
        """Calculate the marginal response (derivative) at given spend levels.

        This measures how much additional response is generated per unit
        of additional spend - useful for optimization.

        Args:
            x: Spend levels at which to calculate marginal response.

        Returns:
            Marginal response values (derivative of Hill function).
        """
        x_arr = np.asarray(x, dtype=np.float64)
        x_arr = np.maximum(x_arr, 1e-10)  # Avoid division by zero

        x_alpha = np.power(x_arr, self.alpha)
        k_alpha = self.k**self.alpha

        numerator = self.alpha * k_alpha * np.power(x_arr, self.alpha - 1)
        denominator = np.power(x_alpha + k_alpha, 2)

        return numerator / (denominator + 1e-10)

    def get_ec50(self) -> float:
        """Get the EC50 (half-maximal effective concentration).

        This is the spend level at which 50% of maximum response is achieved.

        Returns:
            The k parameter, which equals EC50 for Hill function.
        """
        return self.k

    def get_ec90(self) -> float:
        """Get the EC90 (90% effective concentration).

        Returns:
            Spend level at which 90% of maximum response is achieved.
        """
        # Solve: 0.9 = x^alpha / (x^alpha + k^alpha)
        # x = k * (0.9 / 0.1)^(1/alpha) = k * 9^(1/alpha)
        return self.k * (9.0 ** (1 / self.alpha))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SaturationTransformer(alpha={self.alpha}, k={self.k})"


def apply_adstock(
    x: NDArray[np.floating] | Sequence[float],
    decay: float = 0.5,
) -> NDArray[np.float64]:
    """Convenience function to apply adstock transformation.

    Args:
        x: Array-like of media spend values.
        decay: Decay rate between 0 and 1.

    Returns:
        Adstocked values.
    """
    return AdstockTransformer(decay=decay).transform(x)


def apply_saturation(
    x: NDArray[np.floating] | Sequence[float],
    alpha: float = 2.0,
    k: float = 500.0,
) -> NDArray[np.float64]:
    """Convenience function to apply Hill saturation transformation.

    Args:
        x: Array-like of values to transform.
        alpha: Shape parameter.
        k: Half-saturation point.

    Returns:
        Saturated values in [0, 1].
    """
    return SaturationTransformer(alpha=alpha, k=k).transform(x)
