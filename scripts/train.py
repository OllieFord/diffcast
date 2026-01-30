#!/usr/bin/env python
"""Train DiffCast model."""

import logging
from pathlib import Path

import lightning as L
import typer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
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
    checkpoint: Path = typer.Option(
        None,
        "--checkpoint",
        help="Resume from checkpoint",
    ),
    fast_dev_run: bool = typer.Option(
        False,
        "--fast-dev-run",
        help="Run quick validation",
    ),
    max_epochs: int = typer.Option(
        None,
        "--max-epochs",
        "-e",
        help="Override max epochs",
    ),
    devices: int = typer.Option(
        1,
        "--devices",
        "-d",
        help="Number of devices",
    ),
    accelerator: str = typer.Option(
        "auto",
        "--accelerator",
        "-a",
        help="Accelerator type (auto, cpu, gpu, mps)",
    ),
) -> None:
    """Train DiffCast diffusion model."""
    # Load config
    config = OmegaConf.load(config_path)

    logger.info("Initializing training...")

    # Create data module
    from diffcast.training.trainer import DiffCastDataModule, DiffCastLightningModule

    data_module = DiffCastDataModule(
        train_path=Path(config.data.splits_dir) / "train.parquet",
        val_path=Path(config.data.splits_dir) / "val.parquet",
        test_path=Path(config.data.splits_dir) / "test.parquet",
        statistics_path=Path(config.data.statistics),
        context_length=config.training.context_length,
        forecast_length=config.training.forecast_length,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        multi_zone=config.model.multi_zone,
    )

    # Create model
    model = DiffCastLightningModule(
        n_input_features=config.model.n_input_features,
        n_covariates=config.model.n_covariates,
        n_zones=config.model.n_zones,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        dropout=config.model.dropout,
        n_diffusion_steps=config.diffusion.n_steps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        noise_schedule=config.diffusion.schedule,
        context_length=config.training.context_length,
        forecast_length=config.training.forecast_length,
        multi_zone=config.model.multi_zone,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_epochs=config.training.warmup_epochs,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.checkpointing.dirpath,
            filename=config.checkpointing.filename,
            save_top_k=config.logging.save_top_k,
            monitor=config.logging.monitor,
            mode=config.logging.mode,
            save_last=config.checkpointing.save_last,
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor=config.logging.monitor,
            patience=15,
            mode=config.logging.mode,
        ),
    ]

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name=config.logging.project,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs or config.training.epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=tb_logger,
        gradient_clip_val=config.training.gradient_clip,
        log_every_n_steps=config.logging.log_every_n_steps,
        fast_dev_run=fast_dev_run,
        precision="16-mixed" if accelerator == "gpu" else 32,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, data_module, ckpt_path=checkpoint)

    # Test (skip in fast_dev_run mode)
    if not fast_dev_run:
        logger.info("Running test evaluation...")
        trainer.test(model, data_module, ckpt_path="best")

    logger.info("Training complete!")


if __name__ == "__main__":
    app()
