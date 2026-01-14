"""
Visualization tools for Marketing Mix Modeling.

This module provides publication-quality plots for MMM analysis,
including contribution charts, response curves, and diagnostic plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import NDArray


if TYPE_CHECKING:
    from mmm_analytics.core.model import ModelResults
    from mmm_analytics.core.pipeline import PipelineResults


# Set style defaults
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class MMMPlotter:
    """Generate visualizations for MMM analysis.

    This class provides methods to create various plots for analyzing
    and presenting MMM results.

    Example:
        >>> plotter = MMMPlotter(figsize=(12, 8))
        >>> fig = plotter.plot_contribution_waterfall(results)
        >>> fig.savefig("contributions.png")

    Attributes:
        figsize: Default figure size for plots.
        style: Matplotlib style to use.
        dpi: Resolution for saved figures.
    """

    def __init__(
        self,
        figsize: tuple[int, int] = (12, 8),
        style: str = "seaborn-v0_8-whitegrid",
        dpi: int = 150,
    ) -> None:
        """Initialize MMMPlotter.

        Args:
            figsize: Default figure size (width, height).
            style: Matplotlib style name.
            dpi: Resolution for saved figures.
        """
        self.figsize = figsize
        self.style = style
        self.dpi = dpi

    def plot_contribution_share(
        self,
        contribution_share: pd.Series,
        title: str = "Channel Contribution Share",
        colors: list[str] | None = None,
    ) -> Figure:
        """Create pie chart of channel contribution shares.

        Args:
            contribution_share: Series with channel -> share mapping.
            title: Plot title.
            colors: Optional list of colors for channels.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if colors is None:
            colors = sns.color_palette("husl", len(contribution_share))

        wedges, texts, autotexts = ax.pie(
            contribution_share.values,
            labels=contribution_share.index,
            autopct="%1.1f%%",
            colors=colors,
            explode=[0.02] * len(contribution_share),
            shadow=True,
            startangle=90,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")

        # Style the percentage labels
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        plt.tight_layout()
        return fig

    def plot_contribution_bars(
        self,
        contribution_share: pd.Series,
        title: str = "Channel Contribution Share",
    ) -> Figure:
        """Create horizontal bar chart of contributions.

        Args:
            contribution_share: Series with channel -> share mapping.
            title: Plot title.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        sorted_share = contribution_share.sort_values(ascending=True)
        colors = sns.color_palette("viridis", len(sorted_share))

        bars = ax.barh(sorted_share.index, sorted_share.values, color=colors)

        # Add value labels
        for bar, val in zip(bars, sorted_share.values):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}",
                va="center",
                fontsize=10,
            )

        ax.set_xlabel("Contribution Share", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0, sorted_share.max() * 1.15)

        plt.tight_layout()
        return fig

    def plot_actual_vs_predicted(
        self,
        y_actual: NDArray[np.floating] | pd.Series,
        y_predicted: NDArray[np.floating] | pd.Series,
        title: str = "Actual vs Predicted",
    ) -> Figure:
        """Create scatter plot of actual vs predicted values.

        Args:
            y_actual: Actual target values.
            y_predicted: Model predictions.
            title: Plot title.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        y_act = np.asarray(y_actual)
        y_pred = np.asarray(y_predicted)

        # Scatter plot
        ax.scatter(y_act, y_pred, alpha=0.6, edgecolors="white", linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_act.min(), y_pred.min())
        max_val = max(y_act.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect fit")

        # Calculate R²
        ss_res = np.sum((y_act - y_pred) ** 2)
        ss_tot = np.sum((y_act - y_act.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        ax.set_xlabel("Actual", fontsize=12)
        ax.set_ylabel("Predicted", fontsize=12)
        ax.set_title(f"{title} (R² = {r2:.4f})", fontsize=14, fontweight="bold")
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_residuals(
        self,
        residuals: NDArray[np.floating] | pd.Series,
        predictions: NDArray[np.floating] | pd.Series | None = None,
    ) -> Figure:
        """Create residual diagnostic plots.

        Args:
            residuals: Model residuals.
            predictions: Optional predictions for residual vs fitted plot.

        Returns:
            Matplotlib Figure with 2x2 subplot grid.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        residuals_arr = np.asarray(residuals)

        # 1. Residuals histogram
        ax1 = axes[0, 0]
        ax1.hist(residuals_arr, bins=30, edgecolor="white", alpha=0.7)
        ax1.axvline(0, color="red", linestyle="--", linewidth=2)
        ax1.set_xlabel("Residual")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Residuals Distribution")

        # 2. Q-Q plot
        ax2 = axes[0, 1]
        from scipy import stats

        stats.probplot(residuals_arr, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normality Check)")

        # 3. Residuals vs Fitted (if predictions provided)
        ax3 = axes[1, 0]
        if predictions is not None:
            pred_arr = np.asarray(predictions)
            ax3.scatter(pred_arr, residuals_arr, alpha=0.6)
            ax3.axhline(0, color="red", linestyle="--", linewidth=2)
            ax3.set_xlabel("Fitted Values")
            ax3.set_ylabel("Residuals")
            ax3.set_title("Residuals vs Fitted")
        else:
            ax3.scatter(range(len(residuals_arr)), residuals_arr, alpha=0.6)
            ax3.axhline(0, color="red", linestyle="--", linewidth=2)
            ax3.set_xlabel("Observation Index")
            ax3.set_ylabel("Residuals")
            ax3.set_title("Residuals Over Time")

        # 4. Residuals autocorrelation
        ax4 = axes[1, 1]
        n_lags = min(20, len(residuals_arr) // 4)
        autocorr = [
            np.corrcoef(residuals_arr[:-k], residuals_arr[k:])[0, 1]
            for k in range(1, n_lags + 1)
        ]
        ax4.bar(range(1, n_lags + 1), autocorr, alpha=0.7)
        ax4.axhline(0, color="black", linewidth=0.5)
        ax4.axhline(1.96 / np.sqrt(len(residuals_arr)), color="red", linestyle="--")
        ax4.axhline(-1.96 / np.sqrt(len(residuals_arr)), color="red", linestyle="--")
        ax4.set_xlabel("Lag")
        ax4.set_ylabel("Autocorrelation")
        ax4.set_title("Residual Autocorrelation")

        plt.suptitle("Residual Diagnostics", fontsize=16, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_response_curves(
        self,
        channel_params: dict[str, tuple[float, float]],
        spend_range: tuple[float, float] = (0, 2000),
        n_points: int = 100,
    ) -> Figure:
        """Plot saturation response curves for each channel.

        Args:
            channel_params: Dict of channel -> (alpha, k) saturation params.
            spend_range: Range of spend values to plot.
            n_points: Number of points for curve.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        x = np.linspace(spend_range[0], spend_range[1], n_points)
        colors = sns.color_palette("husl", len(channel_params))

        for (channel, (alpha, k)), color in zip(channel_params.items(), colors):
            # Hill function
            y = (x**alpha) / (x**alpha + k**alpha + 1e-10)
            ax.plot(x, y, label=f"{channel} (α={alpha:.1f}, k={k:.0f})", color=color, linewidth=2)

            # Mark EC50 (half-saturation point)
            ax.axvline(k, color=color, linestyle=":", alpha=0.5)

        ax.set_xlabel("Spend", fontsize=12)
        ax.set_ylabel("Response (Saturated)", fontsize=12)
        ax.set_title("Channel Response Curves (Hill Saturation)", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        return fig

    def plot_coefficient_importance(
        self,
        coefficients: pd.Series,
        confidence_intervals: pd.DataFrame | None = None,
    ) -> Figure:
        """Plot feature coefficients with optional confidence intervals.

        Args:
            coefficients: Series of feature coefficients.
            confidence_intervals: DataFrame with 'lower' and 'upper' columns.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        sorted_coef = coefficients.sort_values(ascending=True)
        y_pos = np.arange(len(sorted_coef))

        # Color bars by sign
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sorted_coef.values]

        bars = ax.barh(y_pos, sorted_coef.values, color=colors, alpha=0.8)

        # Add error bars if CI provided
        if confidence_intervals is not None:
            ci = confidence_intervals.loc[sorted_coef.index]
            xerr = np.array(
                [
                    sorted_coef.values - ci["lower"].values,
                    ci["upper"].values - sorted_coef.values,
                ]
            )
            ax.errorbar(
                sorted_coef.values,
                y_pos,
                xerr=xerr,
                fmt="none",
                color="black",
                capsize=3,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_coef.index)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Coefficient Value", fontsize=12)
        ax.set_title("Feature Coefficients", fontsize=14, fontweight="bold")

        plt.tight_layout()
        return fig

    def plot_budget_optimization(
        self,
        current: pd.Series,
        optimized: pd.Series,
        title: str = "Budget Reallocation",
    ) -> Figure:
        """Plot current vs optimized budget allocation.

        Args:
            current: Current spend by channel.
            optimized: Optimized spend by channel.
            title: Plot title.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        channels = current.index
        x = np.arange(len(channels))
        width = 0.35

        bars1 = ax.bar(x - width / 2, current.values, width, label="Current", color="#3498db")
        bars2 = ax.bar(x + width / 2, optimized.values, width, label="Optimized", color="#2ecc71")

        ax.set_xlabel("Channel", fontsize=12)
        ax.set_ylabel("Spend ($)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45, ha="right")
        ax.legend()

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(
                f"${height:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(
                f"${height:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        return fig

    def plot_time_series_decomposition(
        self,
        data: pd.DataFrame,
        date_column: str,
        kpi_column: str,
        predictions: NDArray[np.floating] | None = None,
    ) -> Figure:
        """Plot time series with actual, predicted, and components.

        Args:
            data: DataFrame with date and KPI columns.
            date_column: Name of date column.
            kpi_column: Name of KPI column.
            predictions: Optional model predictions.

        Returns:
            Matplotlib Figure.
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        dates = data[date_column]
        actual = data[kpi_column]

        # Top plot: Actual vs Predicted
        ax1 = axes[0]
        ax1.plot(dates, actual, label="Actual", linewidth=2, color="#3498db")
        if predictions is not None:
            ax1.plot(dates, predictions, label="Predicted", linewidth=2, color="#e74c3c", linestyle="--")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("KPI")
        ax1.set_title("Actual vs Predicted Over Time", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom plot: Residuals over time
        ax2 = axes[1]
        if predictions is not None:
            residuals = actual.values - predictions
            ax2.fill_between(dates, residuals, alpha=0.3, color="#9b59b6")
            ax2.plot(dates, residuals, color="#9b59b6", linewidth=1)
            ax2.axhline(0, color="red", linestyle="--")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Residual")
            ax2.set_title("Residuals Over Time", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_all_plots(
        self,
        results: Any,  # PipelineResults
        output_dir: str | Path,
    ) -> list[Path]:
        """Save all standard plots to directory.

        Args:
            results: PipelineResults from pipeline.fit().
            output_dir: Directory to save plots.

        Returns:
            List of saved file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # Contribution share
        fig = self.plot_contribution_share(results.model_results.attribution.contribution_share)
        path = output_path / "contribution_share.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)

        # Contribution bars
        fig = self.plot_contribution_bars(results.model_results.attribution.contribution_share)
        path = output_path / "contribution_bars.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)

        # Actual vs Predicted
        fig = self.plot_actual_vs_predicted(
            results.raw_data[results.config.target_column],
            results.model_results.predictions,
        )
        path = output_path / "actual_vs_predicted.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)

        # Residuals
        fig = self.plot_residuals(
            results.model_results.residuals,
            results.model_results.predictions,
        )
        path = output_path / "residual_diagnostics.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)

        # Coefficients
        fig = self.plot_coefficient_importance(
            results.model_results.attribution.coefficients,
            results.model_results.attribution.confidence_intervals,
        )
        path = output_path / "coefficients.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)

        # Optimization (if available)
        if results.optimization_results is not None:
            fig = self.plot_budget_optimization(
                results.optimization_results.current_allocation,
                results.optimization_results.optimized_allocation,
            )
            path = output_path / "budget_optimization.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(path)

        return saved_files

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MMMPlotter(figsize={self.figsize}, dpi={self.dpi})"
