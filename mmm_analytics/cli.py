"""
Command-line interface for MMM Analytics.

This module provides a CLI for running MMM analysis from the terminal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mmm_analytics import __version__
from mmm_analytics.core.pipeline import MMMPipeline, PipelineConfig
from mmm_analytics.data.simulator import DataSimulator, SimulationConfig
from mmm_analytics.diagnostics.plots import MMMPlotter


# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

console = Console()
logger = structlog.get_logger(__name__)


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """MMM Analytics - Marketing Mix Modeling CLI.

    Production-grade Marketing Mix Modeling for Ad Tech & Media Analytics.
    """
    pass


@main.command()
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    help="Path to CSV data file.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML/JSON config file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./mmm_output",
    help="Output directory for results.",
)
@click.option(
    "--target",
    "-t",
    default="kpi",
    help="Target column name (default: kpi).",
)
@click.option(
    "--channels",
    multiple=True,
    help="Channel names (can specify multiple times).",
)
@click.option(
    "--no-plots",
    is_flag=True,
    help="Skip generating plots.",
)
@click.option(
    "--no-optimize",
    is_flag=True,
    help="Skip budget optimization.",
)
def run(
    data: str | None,
    config: str | None,
    output: str,
    target: str,
    channels: tuple[str, ...],
    no_plots: bool,
    no_optimize: bool,
) -> None:
    """Run MMM analysis on data.

    If no data file is provided, generates synthetic demo data.

    Examples:

        # Run with synthetic data
        mmm run --output ./results

        # Run with real data
        mmm run --data marketing_data.csv --target conversions --output ./results

        # Specify channels explicitly
        mmm run --data data.csv --channels search --channels social --channels display
    """
    console.print(Panel.fit("🚀 MMM Analytics", style="bold blue"))

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load or generate data
    if data:
        console.print(f"📂 Loading data from: {data}")
        df = pd.read_csv(data)
    else:
        console.print("📊 Generating synthetic demo data...")
        simulator = DataSimulator(n_weeks=104, seed=42)
        df = simulator.generate()
        df.to_csv(output_path / "synthetic_data.csv", index=False)
        console.print(f"   Saved synthetic data to: {output_path / 'synthetic_data.csv'}")

    console.print(f"   Data shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Configure pipeline
    pipeline_config = PipelineConfig(
        channels=list(channels) if channels else [],
        target_column=target,
    )

    if config:
        console.print(f"⚙️  Loading config from: {config}")
        config_path = Path(config)
        if config_path.suffix == ".json":
            with open(config_path) as f:
                cfg_dict = json.load(f)
            # Update pipeline config from file
            for key, value in cfg_dict.items():
                if hasattr(pipeline_config, key):
                    setattr(pipeline_config, key, value)

    # Run pipeline
    console.print("\n🔧 Running MMM Pipeline...")

    pipeline = MMMPipeline(pipeline_config)
    results = pipeline.fit(df, run_optimization=not no_optimize)

    # Display results
    _display_results(results)

    # Save results
    console.print(f"\n💾 Saving results to: {output_path}")
    results.save_results(output_path)

    # Generate plots
    if not no_plots:
        console.print("📈 Generating plots...")
        plotter = MMMPlotter()
        saved_plots = plotter.save_all_plots(results, output_path / "plots")
        console.print(f"   Saved {len(saved_plots)} plots")

    console.print("\n✅ Analysis complete!", style="bold green")


@main.command()
@click.option(
    "--weeks",
    "-w",
    default=104,
    help="Number of weeks to simulate.",
)
@click.option(
    "--seed",
    "-s",
    default=42,
    help="Random seed for reproducibility.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./synthetic_data.csv",
    help="Output CSV file path.",
)
def generate(weeks: int, seed: int, output: str) -> None:
    """Generate synthetic marketing data for testing.

    Examples:

        # Generate 2 years of weekly data
        mmm generate --weeks 104 --output training_data.csv

        # Generate with specific seed
        mmm generate --weeks 52 --seed 123 --output test_data.csv
    """
    console.print(Panel.fit("📊 Generating Synthetic Data", style="bold blue"))

    simulator = DataSimulator(n_weeks=weeks, seed=seed)
    df = simulator.generate()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Show data summary
    table = Table(title="Generated Data Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Rows", str(len(df)))
    table.add_row("Columns", str(len(df.columns)))
    table.add_row("Date Range", f"{df['date'].min()} to {df['date'].max()}")
    table.add_row("Channels", ", ".join([c.replace("spend_", "") for c in df.columns if c.startswith("spend_")]))

    console.print(table)
    console.print(f"\n💾 Saved to: {output_path}", style="bold green")


@main.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./mmm_config.json",
    help="Output config file path.",
)
def init(output: str) -> None:
    """Generate a sample configuration file.

    Examples:

        # Generate default config
        mmm init --output my_config.json
    """
    console.print(Panel.fit("⚙️  Generating Configuration", style="bold blue"))

    config = {
        "channels": ["search", "social", "display", "audio"],
        "target_column": "kpi",
        "date_column": "date",
        "adstock_decay": 0.5,
        "saturation_alpha": 2.0,
        "saturation_k": 500.0,
        "model_alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
        "cv_folds": 5,
        "include_trend": True,
        "include_seasonality": True,
        "optimization_method": "scipy",
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    console.print(f"💾 Saved config to: {output_path}", style="bold green")
    console.print("\nEdit this file to customize your MMM analysis.")


@main.command()
def demo() -> None:
    """Run a complete demo with synthetic data.

    This command generates synthetic data, runs the MMM pipeline,
    and displays results - all in one step.
    """
    console.print(Panel.fit("🎯 MMM Analytics Demo", style="bold blue"))

    # Generate data
    console.print("\n📊 Step 1: Generating synthetic data...")
    simulator = DataSimulator(n_weeks=104, seed=42)
    df = simulator.generate()
    console.print(f"   Generated {len(df)} weeks of data")

    # Run pipeline
    console.print("\n🔧 Step 2: Running MMM analysis...")
    pipeline = MMMPipeline()
    results = pipeline.fit(df)

    # Display results
    console.print("\n📈 Step 3: Results")
    _display_results(results)

    # Compare with ground truth
    console.print("\n🎯 Ground Truth Comparison:")
    true_shares = simulator.get_true_contribution_shares()
    est_shares = results.get_channel_summary()["share"]

    table = Table(title="Contribution Share: True vs Estimated")
    table.add_column("Channel", style="cyan")
    table.add_column("True Share", style="green")
    table.add_column("Estimated Share", style="yellow")
    table.add_column("Error", style="red")

    for channel in true_shares.index:
        true_val = true_shares[channel]
        est_key = f"{channel}_transformed" if f"{channel}_transformed" in est_shares.index else channel
        if est_key in est_shares.index:
            est_val = est_shares[est_key]
            error = abs(true_val - est_val)
            table.add_row(
                channel,
                f"{true_val:.2%}",
                f"{est_val:.2%}",
                f"{error:.2%}",
            )

    console.print(table)
    console.print("\n✅ Demo complete!", style="bold green")


def _display_results(results: Any) -> None:
    """Display results in a formatted table."""
    # Metrics table
    metrics = results.model_results.metrics
    table = Table(title="Model Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("R-squared", f"{metrics.r2:.4f}")
    table.add_row("Adjusted R²", f"{metrics.adj_r2:.4f}")
    table.add_row("RMSE", f"{metrics.rmse:.4f}")
    table.add_row("MAE", f"{metrics.mae:.4f}")
    table.add_row("MAPE", f"{metrics.mape:.2f}%")
    table.add_row("Durbin-Watson", f"{metrics.durbin_watson:.4f}")

    console.print(table)

    # Channel contributions
    contrib = results.model_results.attribution.contribution_share
    table2 = Table(title="Channel Contribution Share")
    table2.add_column("Channel", style="cyan")
    table2.add_column("Share", style="green")

    for channel, share in contrib.items():
        table2.add_row(str(channel).replace("_transformed", ""), f"{share:.2%}")

    console.print(table2)

    # Optimization results
    if results.optimization_results:
        opt = results.optimization_results
        console.print(f"\n📊 Expected lift from optimization: {opt.expected_lift:+.2%}")


if __name__ == "__main__":
    main()
