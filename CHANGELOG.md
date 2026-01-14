# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-14

### Added

- **Core MMM Framework**
  - `AdstockTransformer`: Geometric decay for media carryover effects
  - `SaturationTransformer`: Hill function for diminishing returns
  - `FeatureEngineer`: Automated feature matrix construction
  - `MarketingMixModel`: Ridge regression with cross-validation
  - `BudgetOptimizer`: Multi-method budget reallocation (greedy, scipy, gradient)
  - `MMMPipeline`: End-to-end pipeline for complete analysis

- **Data Generation**
  - `DataSimulator`: Synthetic data with known ground truth
  - Configurable channel parameters and noise levels
  - Ground truth recovery for model validation

- **Statistical Diagnostics**
  - R², Adjusted R², RMSE, MAE, MAPE metrics
  - VIF for multicollinearity detection
  - Durbin-Watson and Ljung-Box for autocorrelation
  - Normality tests (Shapiro-Wilk, Jarque-Bera)
  - Bootstrap confidence intervals

- **Visualization**
  - Contribution share charts (pie, bar)
  - Actual vs predicted plots
  - Residual diagnostics (4-panel)
  - Response curves
  - Budget optimization comparison

- **CLI Interface**
  - `mmm run`: Execute analysis on data
  - `mmm generate`: Create synthetic data
  - `mmm init`: Generate config template
  - `mmm demo`: Run complete demo

- **CI/CD**
  - GitHub Actions for testing, linting, type checking
  - Multi-OS, multi-Python version matrix
  - Security scanning with Bandit and Safety
  - Automated release workflow
  - Code coverage with Codecov

- **Documentation**
  - Comprehensive README with examples
  - API documentation with Google-style docstrings
  - Contributing guide
  - Issue and PR templates

### Technical Details

- Python 3.10+ support
- Type hints throughout (mypy strict mode)
- Ruff for linting and formatting
- pytest with hypothesis for property-based testing
- Pydantic for configuration validation

---

## [Unreleased]

### Planned Features

- Bayesian MMM with PyMC
- Time-varying coefficients
- Cross-channel interaction effects
- Automated hyperparameter tuning
- Response curve visualization improvements

[1.0.0]: https://github.com/saisanthoshv/mmm-analytics/releases/tag/v1.0.0
[Unreleased]: https://github.com/saisanthoshv/mmm-analytics/compare/v1.0.0...HEAD
