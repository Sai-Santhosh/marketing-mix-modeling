#!/usr/bin/env python3
"""
Demo script for MMM Analytics.

This script demonstrates the complete MMM workflow:
1. Generate synthetic marketing data
2. Run the MMM pipeline
3. Display results and diagnostics
4. Perform budget optimization
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmm_analytics import MMMPipeline
from mmm_analytics.core.pipeline import PipelineConfig
from mmm_analytics.data import DataSimulator
from mmm_analytics.diagnostics.plots import MMMPlotter


def main() -> None:
    """Run the complete MMM demo."""
    print("=" * 60)
    print("MMM Analytics Demo")
    print("=" * 60)

    # Step 1: Generate synthetic data
    print("\n[Step 1] Generating synthetic data...")
    simulator = DataSimulator(n_weeks=104, seed=42)
    data = simulator.generate()
    print(f"   Generated {len(data)} weeks of data")
    print(f"   Channels: {[c.replace('spend_', '') for c in data.columns if c.startswith('spend_')]}")
    print(f"   Date range: {data['date'].min().date()} to {data['date'].max().date()}")

    # Step 2: Run MMM Pipeline
    print("\n[Step 2] Running MMM Pipeline...")
    config = PipelineConfig(
        channels=["search", "social", "display", "audio"],
        target_column="kpi",
        adstock_decay=0.5,
        saturation_alpha=2.0,
        saturation_k=500.0,
        cv_folds=5,
    )

    pipeline = MMMPipeline(config)
    results = pipeline.fit(data, run_optimization=True)

    # Step 3: Display Results
    print("\n[Step 3] Model Results")
    print(results.model_results.summary())

    # Step 4: Channel Summary
    print("\n[Step 4] Channel Performance Summary:")
    summary = results.get_channel_summary()
    print(summary.to_string())

    # Step 5: Optimization Results
    if results.optimization_results:
        print("\n[Step 5] Budget Optimization:")
        print(results.optimization_results.summary())

    # Step 6: Compare with Ground Truth
    print("\n[Step 6] Ground Truth Comparison:")
    true_shares = simulator.get_true_contribution_shares()
    est_shares = summary["share"]

    print(f"{'Channel':<15} {'True Share':>12} {'Estimated':>12} {'Error':>10}")
    print("-" * 50)
    for channel in true_shares.index:
        true_val = true_shares[channel]
        est_val = est_shares.get(channel, 0)
        error = abs(true_val - est_val)
        print(f"{channel:<15} {true_val:>11.2%} {est_val:>11.2%} {error:>9.2%}")

    # Step 7: Diagnostics
    print("\n[Step 7] Model Diagnostics:")
    metrics = results.model_results.metrics
    print(f"   R-squared:     {metrics.r2:.4f}")
    print(f"   Adjusted R2:   {metrics.adj_r2:.4f}")
    print(f"   RMSE:          {metrics.rmse:.2f}")
    print(f"   MAPE:          {metrics.mape:.2f}%")
    print(f"   Durbin-Watson: {metrics.durbin_watson:.4f}")

    # Step 8: Save plots (optional)
    print("\n[Step 8] Generating plots...")
    output_dir = Path("mmm_output")
    output_dir.mkdir(exist_ok=True)

    plotter = MMMPlotter()
    try:
        saved_plots = plotter.save_all_plots(results, output_dir / "plots")
        print(f"   Saved {len(saved_plots)} plots to {output_dir / 'plots'}")
    except Exception as e:
        print(f"   Could not save plots: {e}")

    # Save results
    results.save_results(output_dir)
    print(f"   Results saved to {output_dir}")

    print("\n[SUCCESS] Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
