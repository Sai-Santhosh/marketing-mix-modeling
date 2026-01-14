"""
Configuration management for MMM Analytics.

This module provides configuration handling using Pydantic for
validation and environment variable support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChannelSettings(BaseModel):
    """Settings for a single media channel."""

    name: str
    adstock_decay: float = Field(default=0.5, ge=0.0, le=1.0)
    saturation_alpha: float = Field(default=2.0, gt=0.0)
    saturation_k: float = Field(default=500.0, gt=0.0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure channel name is valid."""
        if not v or not v.strip():
            raise ValueError("Channel name cannot be empty")
        return v.strip().lower()


class ModelSettings(BaseModel):
    """Settings for the MMM model."""

    alphas: list[float] = Field(default=[0.01, 0.1, 1.0, 10.0, 100.0])
    cv_folds: int = Field(default=5, ge=2, le=20)
    fit_intercept: bool = True
    scale_features: bool = True


class OptimizationSettings(BaseModel):
    """Settings for budget optimization."""

    method: str = Field(default="scipy", pattern="^(greedy|scipy|gradient)$")
    max_iterations: int = Field(default=100, ge=1)
    tolerance: float = Field(default=1e-6, gt=0)


class PipelineSettings(BaseModel):
    """Complete pipeline settings."""

    channels: list[ChannelSettings] = Field(default_factory=list)
    target_column: str = "kpi"
    date_column: str | None = None
    include_trend: bool = True
    include_seasonality: bool = True
    seasonality_period: int = Field(default=52, ge=2)
    n_harmonics: int = Field(default=2, ge=1, le=10)
    model: ModelSettings = Field(default_factory=ModelSettings)
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings)


class MMMSettings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="MMM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Output
    output_dir: Path = Path("./mmm_output")
    save_plots: bool = True
    plot_dpi: int = 150

    # Pipeline defaults
    default_adstock_decay: float = 0.5
    default_saturation_alpha: float = 2.0
    default_saturation_k: float = 500.0


def load_config(config_path: str | Path) -> PipelineSettings:
    """Load pipeline configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Validated PipelineSettings object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            config_dict = yaml.safe_load(f)
        else:
            import json

            config_dict = json.load(f)

    return PipelineSettings(**config_dict)


def save_config(settings: PipelineSettings, config_path: str | Path) -> None:
    """Save pipeline configuration to file.

    Args:
        settings: PipelineSettings to save.
        config_path: Output file path.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = settings.model_dump()

    with open(path, "w") as f:
        if path.suffix in (".yaml", ".yml"):
            yaml.safe_dump(config_dict, f, default_flow_style=False)
        else:
            import json

            json.dump(config_dict, f, indent=2)


def get_default_config() -> PipelineSettings:
    """Get default pipeline configuration.

    Returns:
        PipelineSettings with sensible defaults.
    """
    return PipelineSettings(
        channels=[
            ChannelSettings(name="search", adstock_decay=0.4, saturation_k=600),
            ChannelSettings(name="social", adstock_decay=0.3, saturation_k=400),
            ChannelSettings(name="display", adstock_decay=0.6, saturation_k=500),
            ChannelSettings(name="audio", adstock_decay=0.5, saturation_k=300),
        ],
        target_column="kpi",
        include_trend=True,
        include_seasonality=True,
    )
