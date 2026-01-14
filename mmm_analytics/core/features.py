"""
Feature engineering for Marketing Mix Modeling.

This module provides the FeatureEngineer class that constructs the design matrix
for MMM by applying transformations (adstock + saturation) to media spend columns
and combining them with control variables (trend, seasonality, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mmm_analytics.core.transforms import AdstockTransformer, SaturationTransformer


if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class ChannelConfig:
    """Configuration for a single media channel.

    Attributes:
        name: Channel identifier (e.g., 'search', 'social').
        spend_column: Column name in DataFrame containing spend data.
        adstock_decay: Decay rate for adstock transformation.
        saturation_alpha: Shape parameter for saturation curve.
        saturation_k: Half-saturation point for saturation curve.
    """

    name: str
    spend_column: str | None = None
    adstock_decay: float = 0.5
    saturation_alpha: float = 2.0
    saturation_k: float = 500.0

    def __post_init__(self) -> None:
        """Set default spend column if not provided."""
        if self.spend_column is None:
            self.spend_column = f"spend_{self.name}"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering.

    Attributes:
        channels: List of channel configurations.
        include_trend: Whether to include linear trend feature.
        include_seasonality: Whether to include seasonality features.
        seasonality_period: Period for seasonal features (default 52 for weekly).
        n_harmonics: Number of Fourier harmonics for seasonality.
        control_columns: Additional control variables to include.
    """

    channels: list[ChannelConfig] = field(default_factory=list)
    include_trend: bool = True
    include_seasonality: bool = True
    seasonality_period: int = 52
    n_harmonics: int = 2
    control_columns: list[str] = field(default_factory=list)


class FeatureEngineer:
    """Construct feature matrix for Marketing Mix Modeling.

    This class applies the following transformations to create model features:
    1. Adstock transformation to capture carryover effects
    2. Saturation transformation to model diminishing returns
    3. Trend and seasonality features for baseline modeling
    4. Control variables as-is

    Example:
        >>> config = FeatureConfig(
        ...     channels=[
        ...         ChannelConfig("search", adstock_decay=0.6),
        ...         ChannelConfig("social", adstock_decay=0.4),
        ...     ],
        ...     include_trend=True,
        ...     include_seasonality=True,
        ... )
        >>> engineer = FeatureEngineer(config)
        >>> X = engineer.fit_transform(df)

    Attributes:
        config: FeatureConfig with transformation settings.
        feature_names_: List of feature names after fit.
        channel_feature_names_: Mapping from channel to feature name.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """Initialize FeatureEngineer.

        Args:
            config: Feature configuration. If None, uses defaults.
        """
        self.config = config or FeatureConfig()
        self.feature_names_: list[str] = []
        self.channel_feature_names_: dict[str, str] = {}
        self._is_fitted = False

    @classmethod
    def from_channels(
        cls,
        channels: Sequence[str],
        default_decay: float = 0.5,
        default_alpha: float = 2.0,
        default_k: float = 500.0,
    ) -> FeatureEngineer:
        """Create FeatureEngineer with default settings for given channels.

        Args:
            channels: List of channel names.
            default_decay: Default adstock decay for all channels.
            default_alpha: Default saturation alpha for all channels.
            default_k: Default saturation k for all channels.

        Returns:
            Configured FeatureEngineer instance.
        """
        channel_configs = [
            ChannelConfig(
                name=ch,
                adstock_decay=default_decay,
                saturation_alpha=default_alpha,
                saturation_k=default_k,
            )
            for ch in channels
        ]
        config = FeatureConfig(channels=channel_configs)
        return cls(config)

    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        """Fit the feature engineer to the data.

        This validates that all required columns exist and prepares
        the feature name mapping.

        Args:
            df: Input DataFrame with spend and control columns.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If required columns are missing.
        """
        self._validate_columns(df)
        self._build_feature_names()
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into model features.

        Args:
            df: Input DataFrame with spend and control columns.

        Returns:
            Feature DataFrame ready for modeling.

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If required columns are missing.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")

        self._validate_columns(df)
        return self._build_features(df)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Input DataFrame with spend and control columns.

        Returns:
            Feature DataFrame ready for modeling.
        """
        return self.fit(df).transform(df)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist in DataFrame."""
        missing_cols = []

        for channel_cfg in self.config.channels:
            col = channel_cfg.spend_column
            if col and col not in df.columns:
                missing_cols.append(col)

        for col in self.config.control_columns:
            if col not in df.columns:
                missing_cols.append(col)

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _build_feature_names(self) -> None:
        """Construct list of feature names."""
        names: list[str] = []
        self.channel_feature_names_ = {}

        # Channel features
        for channel_cfg in self.config.channels:
            feat_name = f"{channel_cfg.name}_transformed"
            names.append(feat_name)
            self.channel_feature_names_[channel_cfg.name] = feat_name

        # Trend
        if self.config.include_trend:
            names.append("trend")

        # Seasonality
        if self.config.include_seasonality:
            for h in range(1, self.config.n_harmonics + 1):
                names.extend([f"sin_{h}", f"cos_{h}"])

        # Control variables
        names.extend(self.config.control_columns)

        self.feature_names_ = names

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the complete feature matrix."""
        n_rows = len(df)
        features: dict[str, np.ndarray] = {}

        # Transform each channel
        for channel_cfg in self.config.channels:
            col = channel_cfg.spend_column
            if col is None:
                continue

            spend = df[col].values

            # Apply adstock
            adstock_transformer = AdstockTransformer(decay=channel_cfg.adstock_decay)
            adstocked = adstock_transformer.transform(spend)

            # Apply saturation
            sat_transformer = SaturationTransformer(
                alpha=channel_cfg.saturation_alpha,
                k=channel_cfg.saturation_k,
            )
            saturated = sat_transformer.transform(adstocked)

            feat_name = self.channel_feature_names_[channel_cfg.name]
            features[feat_name] = saturated

        # Trend feature
        if self.config.include_trend:
            features["trend"] = np.arange(n_rows, dtype=np.float64)

        # Seasonality features (Fourier terms)
        if self.config.include_seasonality:
            t = np.arange(n_rows, dtype=np.float64)
            period = self.config.seasonality_period

            for h in range(1, self.config.n_harmonics + 1):
                features[f"sin_{h}"] = np.sin(2 * np.pi * h * t / period)
                features[f"cos_{h}"] = np.cos(2 * np.pi * h * t / period)

        # Control variables
        for col in self.config.control_columns:
            features[col] = df[col].values.astype(np.float64)

        return pd.DataFrame(features, index=df.index)

    def get_channel_features(self) -> list[str]:
        """Get list of transformed channel feature names.

        Returns:
            List of feature names corresponding to media channels.
        """
        return list(self.channel_feature_names_.values())

    def get_control_features(self) -> list[str]:
        """Get list of control/baseline feature names.

        Returns:
            List of control feature names (trend, seasonality, etc.).
        """
        channel_feats = set(self.get_channel_features())
        return [f for f in self.feature_names_ if f not in channel_feats]

    def __repr__(self) -> str:
        """Return string representation."""
        n_channels = len(self.config.channels)
        return f"FeatureEngineer(channels={n_channels}, fitted={self._is_fitted})"
