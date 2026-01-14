"""
MMM Analytics - Production-grade Marketing Mix Modeling for Ad Tech & Media Analytics.

This package provides a complete toolkit for Marketing Mix Modeling (MMM),
including data transformation, model fitting, attribution analysis, and
budget optimization for media spend allocation.

Example:
    >>> from mmm_analytics import MMMPipeline
    >>> from mmm_analytics.data import DataSimulator
    >>>
    >>> # Generate synthetic data
    >>> simulator = DataSimulator(n_weeks=104)
    >>> data = simulator.generate()
    >>>
    >>> # Run MMM pipeline
    >>> pipeline = MMMPipeline()
    >>> results = pipeline.fit(data)
    >>> print(results.summary())
"""

from mmm_analytics.core.model import MarketingMixModel
from mmm_analytics.core.pipeline import MMMPipeline
from mmm_analytics.core.transforms import AdstockTransformer, SaturationTransformer


__version__ = "1.0.0"
__author__ = "Sai Santhosh V"
__license__ = "MIT"

__all__ = [
    "MarketingMixModel",
    "MMMPipeline",
    "AdstockTransformer",
    "SaturationTransformer",
    "__version__",
    "__author__",
    "__license__",
]
