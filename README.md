<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Code%20Style-Ruff-purple?style=for-the-badge" alt="Ruff">
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a>
</p>

---

# MMM Analytics

**Production-grade Marketing Mix Modeling for Ad Tech & Media Analytics**

A comprehensive Python toolkit for Marketing Mix Modeling (MMM) that enables data scientists to measure advertising effectiveness, attribute marketing impact, and optimize budget allocation across media channels.

## 🎯 What is Marketing Mix Modeling?

Marketing Mix Modeling is a statistical analysis technique used to estimate the impact of various marketing tactics on sales and other business outcomes. This package implements:

- **Adstock Transformation**: Captures the carryover effect of advertising (geometric decay)
- **Saturation Curves**: Models diminishing returns using Hill functions
- **Ridge Regression**: Regularized linear model for stable coefficient estimation
- **Budget Optimization**: Algorithms to reallocate spend for maximum ROI

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📊 **Complete MMM Pipeline** | End-to-end workflow from raw data to insights |
| 🔄 **Adstock & Saturation** | Industry-standard media transformations |
| 📈 **Statistical Diagnostics** | R², MAPE, VIF, Durbin-Watson, confidence intervals |
| 💰 **Budget Optimization** | Multiple algorithms (scipy, greedy, gradient) |
| 🎨 **Visualization Suite** | Publication-ready plots and charts |
| 🖥️ **CLI Interface** | Command-line tools for automation |
| 🧪 **Synthetic Data** | Built-in data generator with known ground truth |
| ✅ **Production Ready** | Type hints, tests, CI/CD, documentation |

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install mmm-analytics
```

### From Source

```bash
git clone https://github.com/saisanthoshv/mmm-analytics.git
cd mmm-analytics
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- NumPy, Pandas, Scikit-learn, SciPy
- Matplotlib, Seaborn (visualization)
- Click, Rich (CLI)

## 🚀 Quick Start

### Python API

```python
from mmm_analytics import MMMPipeline
from mmm_analytics.data import DataSimulator

# Generate synthetic marketing data
simulator = DataSimulator(n_weeks=104, seed=42)
data = simulator.generate()

# Run the MMM pipeline
pipeline = MMMPipeline()
results = pipeline.fit(data)

# View results
print(results.model_results.summary())
```

**Output:**
```
============================================================
MARKETING MIX MODEL RESULTS
============================================================

MODEL FIT METRICS:
  R-squared:          0.9523
  Adjusted R-squared: 0.9487
  RMSE:               18.4521
  MAE:                14.2103
  MAPE:               4.82%
  Durbin-Watson:      1.9234

REGULARIZATION: alpha = 1.0

CHANNEL CONTRIBUTION SHARE:
  search_transformed: 38.24%
  social_transformed: 25.67%
  audio_transformed: 19.43%
  display_transformed: 16.66%
============================================================
```

### Command Line Interface

```bash
# Run complete demo
mmm demo

# Generate synthetic data
mmm generate --weeks 104 --output data.csv

# Run analysis on data
mmm run --data data.csv --output ./results

# Generate config template
mmm init --output config.json
```

## 📖 Documentation

### Core Concepts

#### 1. Adstock Transformation

Adstock models the carryover effect of advertising where the impact persists and decays over time.

```python
from mmm_analytics.core.transforms import AdstockTransformer

transformer = AdstockTransformer(decay=0.5)
spend = [100, 0, 0, 0, 0]
adstocked = transformer.transform(spend)
# Result: [100.0, 50.0, 25.0, 12.5, 6.25]
```

**Mathematical formulation:**
```
A(t) = X(t) + decay × A(t-1)
```

#### 2. Saturation (Hill Function)

The Hill function models diminishing returns at high spend levels.

```python
from mmm_analytics.core.transforms import SaturationTransformer

transformer = SaturationTransformer(alpha=2.0, k=500.0)
spend = [0, 250, 500, 750, 1000]
saturated = transformer.transform(spend)
# Result: [0.0, 0.2, 0.5, 0.692, 0.8]
```

**Mathematical formulation:**
```
S(x) = x^α / (x^α + k^α)
```

Where:
- `α` (alpha): Shape parameter controlling curve steepness
- `k`: Half-saturation point (EC50)

#### 3. Feature Engineering

Combine transformations to build the model design matrix:

```python
from mmm_analytics.core.features import FeatureEngineer, FeatureConfig, ChannelConfig

config = FeatureConfig(
    channels=[
        ChannelConfig(name="search", adstock_decay=0.4, saturation_k=600),
        ChannelConfig(name="social", adstock_decay=0.3, saturation_k=400),
        ChannelConfig(name="display", adstock_decay=0.6, saturation_k=500),
    ],
    include_trend=True,
    include_seasonality=True,
)

engineer = FeatureEngineer(config)
X = engineer.fit_transform(data)
```

#### 4. Model Fitting

Fit a Ridge regression model with cross-validated regularization:

```python
from mmm_analytics.core.model import MarketingMixModel, ModelConfig

config = ModelConfig(
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
    cv_folds=5,
    scale_features=True,
)

model = MarketingMixModel(config)
results = model.fit(X, y)

print(f"R²: {results.metrics.r2:.4f}")
print(f"Selected alpha: {results.selected_alpha}")
print(f"\nChannel contributions:\n{results.attribution.contribution_share}")
```

#### 5. Budget Optimization

Optimize budget allocation across channels:

```python
from mmm_analytics.core.optimizer import BudgetOptimizer, OptimizationConstraints

optimizer = BudgetOptimizer(method="scipy")

constraints = OptimizationConstraints(
    total_budget=100000,
    min_spend={"search": 10000},  # Minimum spend per channel
    max_spend=50000,               # Maximum spend per channel
)

result = optimizer.optimize(
    current_spend=current_allocation,
    coefficients=model_coefficients,
    saturation_params={"search": (2.0, 600), "social": (2.0, 400)},
    constraints=constraints,
)

print(result.summary())
```

### Statistical Diagnostics

The package provides comprehensive diagnostic metrics:

```python
from mmm_analytics.diagnostics import calculate_model_diagnostics

diagnostics = calculate_model_diagnostics(y_true, y_pred, X)
print(diagnostics.summary())
```

**Available Metrics:**

| Category | Metrics |
|----------|---------|
| Goodness of Fit | R², Adjusted R², RMSE, MAE, MAPE, WMAPE |
| Multicollinearity | Variance Inflation Factor (VIF) |
| Autocorrelation | Durbin-Watson, Ljung-Box Q-statistic |
| Normality | Shapiro-Wilk, Jarque-Bera, D'Agostino K² |
| Uncertainty | Bootstrap confidence intervals |

### Visualization

Generate publication-ready plots:

```python
from mmm_analytics.diagnostics.plots import MMMPlotter

plotter = MMMPlotter(figsize=(12, 8), dpi=150)

# Contribution share
fig = plotter.plot_contribution_share(results.attribution.contribution_share)
fig.savefig("contribution_share.png")

# Actual vs Predicted
fig = plotter.plot_actual_vs_predicted(y_actual, y_predicted)

# Residual diagnostics
fig = plotter.plot_residuals(results.residuals, results.predictions)

# Response curves
fig = plotter.plot_response_curves({
    "search": (2.0, 600),
    "social": (2.0, 400),
})

# Save all plots at once
plotter.save_all_plots(pipeline_results, "./plots")
```

### Configuration

Use YAML or JSON configuration files:

```python
from mmm_analytics.config import load_config, PipelineSettings

# Load from file
settings = load_config("config.yaml")

# Or create programmatically
from mmm_analytics.config import ChannelSettings, ModelSettings

settings = PipelineSettings(
    channels=[
        ChannelSettings(name="search", adstock_decay=0.4),
        ChannelSettings(name="social", adstock_decay=0.3),
    ],
    model=ModelSettings(alphas=[0.1, 1.0, 10.0], cv_folds=5),
)
```

**Example config.yaml:**
```yaml
channels:
  - name: search
    adstock_decay: 0.4
    saturation_alpha: 2.0
    saturation_k: 600
  - name: social
    adstock_decay: 0.3
    saturation_k: 400

target_column: conversions
include_trend: true
include_seasonality: true

model:
  alphas: [0.01, 0.1, 1.0, 10.0]
  cv_folds: 5

optimization:
  method: scipy
  max_iterations: 100
```

## 🧪 Testing

```bash
# Run all tests
pytest tests -v

# Run with coverage
pytest tests --cov=mmm_analytics --cov-report=html

# Run specific test module
pytest tests/test_transforms.py -v

# Run only fast tests
pytest tests -v -m "not slow"
```

## 📊 Example Results

### Channel Attribution

| Channel | Coefficient | Contribution | Share | ROAS Index |
|---------|------------|--------------|-------|------------|
| Search  | 118.45     | 45.23        | 38.2% | 1.00       |
| Social  | 76.32      | 30.41        | 25.7% | 0.64       |
| Audio   | 52.18      | 23.02        | 19.4% | 0.44       |
| Display | 38.91      | 19.72        | 16.7% | 0.33       |

### Model Performance

| Metric | Value |
|--------|-------|
| R² | 0.952 |
| Adjusted R² | 0.949 |
| RMSE | 18.45 |
| MAE | 14.21 |
| MAPE | 4.82% |
| Durbin-Watson | 1.92 |

## 🏗️ Project Structure

```
mmm-analytics/
├── mmm_analytics/
│   ├── __init__.py           # Package exports
│   ├── cli.py                # CLI interface
│   ├── config.py             # Configuration management
│   ├── core/
│   │   ├── transforms.py     # Adstock & saturation
│   │   ├── features.py       # Feature engineering
│   │   ├── model.py          # Ridge regression model
│   │   ├── optimizer.py      # Budget optimization
│   │   └── pipeline.py       # End-to-end pipeline
│   ├── data/
│   │   └── simulator.py      # Synthetic data generation
│   └── diagnostics/
│       ├── metrics.py        # Statistical metrics
│       └── plots.py          # Visualization
├── tests/                    # Comprehensive test suite
├── .github/workflows/        # CI/CD pipelines
├── pyproject.toml           # Project configuration
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
└── LICENSE
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/saisanthoshv/mmm-analytics.git
cd mmm-analytics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests -v

# Run linting
ruff check mmm_analytics tests
```

## 📚 References

### Academic Papers

1. Jin, Y., et al. (2017). "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects." Google Research.
2. Chan, D., & Perry, M. (2017). "Challenges and Opportunities in Media Mix Modeling." Google Research.
3. Zhang, S., & Vaver, J. (2017). "Introduction to the Aggregate Marketing System Simulator." Google Research.

### Related Projects

- [LightweightMMM](https://github.com/google/lightweight_mmm) - Google's Bayesian MMM
- [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) - PyMC Labs MMM
- [Robyn](https://github.com/facebookexperimental/Robyn) - Meta's MMM solution

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sai Santhosh V**

- GitHub: [@saisanthoshv](https://github.com/saisanthoshv)

---

<p align="center">
  Made with ❤️ for the Ad Tech & Media Analytics community
</p>
