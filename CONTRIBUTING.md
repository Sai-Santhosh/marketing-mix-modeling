# Contributing to MMM Analytics

Thank you for your interest in contributing to MMM Analytics! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/mmm-analytics.git
   cd mmm-analytics
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/saisanthoshv/mmm-analytics.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Verify Installation

```bash
# Run tests
pytest tests -v

# Run linting
ruff check mmm_analytics tests

# Run type checking
mypy mmm_analytics
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-bayesian-model`
- `bugfix/fix-saturation-overflow`
- `docs/update-quickstart`
- `refactor/simplify-optimizer`

### Workflow

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, focused commits

3. **Keep your branch updated**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

## Coding Standards

### Style Guide

We use **Ruff** for linting and formatting. Configuration is in `pyproject.toml`.

```bash
# Check for issues
ruff check mmm_analytics tests

# Auto-fix issues
ruff check mmm_analytics tests --fix

# Format code
ruff format mmm_analytics tests
```

### Type Hints

All public functions must have type hints:

```python
def calculate_adstock(
    spend: np.ndarray,
    decay: float = 0.5,
) -> np.ndarray:
    """Apply adstock transformation to spend data.
    
    Args:
        spend: Array of spend values.
        decay: Decay rate between 0 and 1.
        
    Returns:
        Adstocked spend values.
    """
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> ModelResults:
    """Fit the marketing mix model.
    
    This function fits a Ridge regression model with cross-validation
    to find the optimal regularization strength.
    
    Args:
        X: Feature matrix with transformed media variables.
        y: Target variable (KPI to model).
        alphas: Regularization strengths to try.
        
    Returns:
        ModelResults containing coefficients, metrics, and predictions.
        
    Raises:
        ValueError: If X and y have different lengths.
        
    Example:
        >>> X = engineer.fit_transform(data)
        >>> results = fit_model(X, data['kpi'])
        >>> print(results.metrics.r2)
    """
    ...
```

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add Bayesian MMM model implementation

- Implement PyMC-based model
- Add posterior sampling
- Include convergence diagnostics
```

Prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

## Testing

### Running Tests

```bash
# Run all tests
pytest tests -v

# Run specific test file
pytest tests/test_transforms.py -v

# Run with coverage
pytest tests --cov=mmm_analytics --cov-report=html

# Run only fast tests
pytest tests -v -m "not slow"
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_adstock_decays_monotonically`
- Use pytest fixtures from `conftest.py`
- Add property-based tests with Hypothesis for numerical code

```python
def test_saturation_bounds() -> None:
    """Test that saturation output is bounded in [0, 1]."""
    transformer = SaturationTransformer(alpha=2.0, k=500.0)
    x = np.linspace(0, 10000, 100)
    result = transformer.transform(x)
    
    assert np.all(result >= 0)
    assert np.all(result <= 1)
```

### Test Coverage

- Aim for >80% coverage
- Cover edge cases and error conditions
- Include integration tests for the pipeline

## Submitting Changes

### Pull Request Process

1. **Ensure all tests pass**:
   ```bash
   pytest tests -v
   ruff check mmm_analytics tests
   mypy mmm_analytics
   ```

2. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub:
   - Use the PR template
   - Link related issues
   - Describe your changes clearly

4. **Address review feedback**:
   - Respond to comments
   - Make requested changes
   - Push updates to your branch

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated (if needed)
- [ ] Commit messages are clear
- [ ] PR description is complete

## Questions?

- Open a [Discussion](https://github.com/saisanthoshv/mmm-analytics/discussions)
- Check existing [Issues](https://github.com/saisanthoshv/mmm-analytics/issues)

Thank you for contributing! 🎉
