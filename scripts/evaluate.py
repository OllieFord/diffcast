#!/usr/bin/env python
"""Evaluate trained DiffCast model."""

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
import typer
from omegaconf import OmegaConf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to model checkpoint",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to config file",
    ),
    output_dir: Path = typer.Option(
        Path("results"),
        "--output",
        "-o",
        help="Output directory for results",
    ),
    n_samples: int = typer.Option(
        100,
        "--n-samples",
        "-n",
        help="Number of samples for evaluation",
    ),
    compare_baselines: bool = typer.Option(
        True,
        "--baselines/--no-baselines",
        help="Compare with baseline models",
    ),
    save_predictions: bool = typer.Option(
        False,
        "--save-predictions",
        help="Save all predictions to file",
    ),
) -> None:
    """Evaluate DiffCast model on test set."""
    # Load config
    config = OmegaConf.load(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {checkpoint}")

    # Load model
    from diffcast.inference.forecast import Forecaster

    forecaster = Forecaster.from_checkpoint(
        checkpoint,
        statistics_path=config.data.statistics,
        device=device,
    )

    # Load test data
    logger.info("Loading test data...")
    from diffcast.data.dataset import DiffCastDataset

    test_dataset = DiffCastDataset(
        data_path=Path(config.data.splits_dir) / "test.parquet",
        context_length=config.training.context_length,
        forecast_length=config.training.forecast_length,
        statistics_path=config.data.statistics,
    )

    # Evaluate
    logger.info("Evaluating model...")
    from diffcast.evaluation.metrics import evaluate_forecast

    all_samples = []
    all_targets = []
    all_contexts = []

    # Use DataLoader for batched evaluation
    from torch.utils.data import DataLoader

    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    for batch in tqdm(loader, desc="Generating forecasts"):
        context = batch["context"]
        target = batch["target"]
        covariates = batch.get("covariates")
        zone_idx = batch.get("zone_idx")

        # Generate samples
        with torch.no_grad():
            samples = forecaster.model.sample(
                context=context.to(device),
                covariates=covariates.to(device) if covariates is not None else None,
                zone_idx=zone_idx.to(device) if zone_idx is not None else None,
                n_samples=n_samples,
                use_ddim=True,
            )

        # Denormalize
        samples = test_dataset.denormalize_prices(samples.cpu())
        target = test_dataset.denormalize_prices(target)

        all_samples.append(samples.numpy())
        all_targets.append(target.numpy())
        all_contexts.append(context.numpy())

    # Concatenate all batches
    all_samples = np.concatenate(all_samples, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_contexts = np.concatenate(all_contexts, axis=0)

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = evaluate_forecast(all_samples, all_targets)

    logger.info("\n=== CSDI Model Results ===")
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")

    # Compare with baselines
    if compare_baselines:
        logger.info("\nEvaluating baselines...")
        from diffcast.models.baselines import (
            PersistenceModel,
            SeasonalPersistenceModel,
        )

        persistence = PersistenceModel(forecast_length=config.training.forecast_length)
        seasonal = SeasonalPersistenceModel(forecast_length=config.training.forecast_length)

        # Extract prices from context (first feature)
        context_prices = all_contexts[:, :, 0]
        # Denormalize context prices
        context_prices = test_dataset.denormalize_prices(
            torch.tensor(context_prices)
        ).numpy()

        persistence_preds = persistence.predict(context_prices, n_samples=n_samples)
        seasonal_preds = seasonal.predict(context_prices, n_samples=n_samples)

        persistence_metrics = evaluate_forecast(persistence_preds, all_targets)
        seasonal_metrics = evaluate_forecast(seasonal_preds, all_targets)

        logger.info("\n=== Persistence Model ===")
        for name, value in persistence_metrics.items():
            logger.info(f"{name}: {value:.4f}")

        logger.info("\n=== Seasonal Persistence ===")
        for name, value in seasonal_metrics.items():
            logger.info(f"{name}: {value:.4f}")

        # Compute improvement
        crps_improvement = (
            (persistence_metrics["crps"] - metrics["crps"]) / persistence_metrics["crps"] * 100
        )
        logger.info(f"\n=== Improvement over Persistence ===")
        logger.info(f"CRPS improvement: {crps_improvement:.1f}%")

        # Save comparison
        comparison = {
            "csdi": metrics,
            "persistence": persistence_metrics,
            "seasonal_persistence": seasonal_metrics,
            "crps_improvement_pct": crps_improvement,
        }

        with open(output_dir / "baseline_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions if requested
    if save_predictions:
        logger.info("Saving predictions...")
        np.savez(
            output_dir / "predictions.npz",
            samples=all_samples,
            targets=all_targets,
            contexts=all_contexts,
        )

    # Generate visualizations
    logger.info("Generating visualizations...")
    from diffcast.evaluation.visualization import plot_calibration, plot_fan_chart

    import matplotlib.pyplot as plt

    # Fan chart for first few examples
    for i in range(min(5, len(all_samples))):
        fig = plot_fan_chart(
            timestamps=np.arange(config.training.forecast_length),
            samples=all_samples[i],
            target=all_targets[i],
            title=f"Forecast Example {i + 1}",
        )
        fig.savefig(output_dir / f"forecast_example_{i + 1}.png", dpi=150)
        plt.close(fig)

    # Calibration plot
    fig = plot_calibration(all_samples, all_targets)
    fig.savefig(output_dir / "calibration.png", dpi=150)
    plt.close(fig)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    app()
