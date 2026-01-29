#!/usr/bin/env python
"""Process raw data and create train/val/test splits."""

import logging
from pathlib import Path

import typer
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to config file",
    ),
    raw_dir: Path = typer.Option(
        None,
        "--raw-dir",
        help="Directory with raw data",
    ),
    processed_dir: Path = typer.Option(
        None,
        "--processed-dir",
        help="Output directory for processed data",
    ),
    splits_dir: Path = typer.Option(
        None,
        "--splits-dir",
        help="Output directory for splits",
    ),
) -> None:
    """Process raw data and create dataset splits."""
    # Load config
    config = OmegaConf.load(config_path)

    # Set directories
    raw_dir = raw_dir or Path(config.data.raw_dir)
    processed_dir = processed_dir or Path(config.data.processed_dir)
    splits_dir = splits_dir or Path(config.data.splits_dir)

    logger.info(f"Raw data: {raw_dir}")
    logger.info(f"Processed output: {processed_dir}")
    logger.info(f"Splits output: {splits_dir}")

    # Run pipeline
    from diffcast.data.pipeline import DataPipeline

    pipeline = DataPipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
    )

    splits = pipeline.run()

    logger.info("Dataset preparation complete")
    logger.info(f"Train samples: {len(splits['train'])}")
    logger.info(f"Val samples: {len(splits['val'])}")
    logger.info(f"Test samples: {len(splits['test'])}")


if __name__ == "__main__":
    app()
