"""PyTorch Lightning training module for DiffCast."""

from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffcast.evaluation.metrics import crps, mae, rmse
from diffcast.models.csdi import CSDI


class DiffCastLightningModule(L.LightningModule):
    """Lightning module for training CSDI forecasting model."""

    def __init__(
        self,
        n_input_features: int = 10,
        n_covariates: int = 4,
        n_zones: int = 10,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_diffusion_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        noise_schedule: str = "quadratic",
        context_length: int = 168,
        forecast_length: int = 24,
        multi_zone: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        n_val_samples: int = 50,
    ) -> None:
        """Initialize Lightning module.

        Args:
            n_input_features: Number of input features
            n_covariates: Number of future covariates
            n_zones: Number of bidding zones
            d_model: Model hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            n_diffusion_steps: Number of diffusion steps
            beta_start: Starting noise level
            beta_end: Ending noise level
            noise_schedule: Noise schedule type
            context_length: Historical context length
            forecast_length: Forecast horizon length
            multi_zone: Whether to use multi-zone model
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_epochs: Number of warmup epochs
            n_val_samples: Number of samples for validation
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = CSDI(
            n_input_features=n_input_features,
            n_covariates=n_covariates,
            n_zones=n_zones,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            n_diffusion_steps=n_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            noise_schedule=noise_schedule,
            context_length=context_length,
            forecast_length=forecast_length,
            multi_zone=multi_zone,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.n_val_samples = n_val_samples

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss = self.model.compute_loss(
            target=batch["target"],
            context=batch["context"],
            covariates=batch.get("covariates"),
            zone_idx=batch.get("zone_idx"),
            mask=batch.get("mask"),
        )

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict:
        """Validation step with sampling."""
        # Compute diffusion loss
        loss = self.model.compute_loss(
            target=batch["target"],
            context=batch["context"],
            covariates=batch.get("covariates"),
            zone_idx=batch.get("zone_idx"),
            mask=batch.get("mask"),
        )

        # Generate samples for probabilistic metrics
        samples = self.model.sample(
            context=batch["context"],
            covariates=batch.get("covariates"),
            zone_idx=batch.get("zone_idx"),
            n_samples=self.n_val_samples,
            use_ddim=True,
            ddim_steps=20,
        )

        # Compute metrics
        target = batch["target"]
        median_pred = torch.median(samples, dim=1).values

        # MAE and RMSE on median
        mae_val = mae(median_pred, target)
        rmse_val = rmse(median_pred, target)

        # CRPS
        crps_val = crps(samples, target)

        # Log metrics
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/mae", mae_val, on_epoch=True)
        self.log("val/rmse", rmse_val, on_epoch=True)
        self.log("val/crps", crps_val, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "mae": mae_val,
            "rmse": rmse_val,
            "crps": crps_val,
        }

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict:
        """Test step (same as validation)."""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> dict:
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - self.warmup_epochs,
            eta_min=self.learning_rate / 100,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def on_train_epoch_start(self) -> None:
        """Learning rate warmup."""
        if self.current_epoch < self.warmup_epochs:
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizers().param_groups:
                param_group["lr"] = self.learning_rate * warmup_factor


class DiffCastDataModule(L.LightningDataModule):
    """Lightning DataModule for DiffCast."""

    def __init__(
        self,
        train_path: Path | str,
        val_path: Path | str,
        test_path: Path | str,
        statistics_path: Path | str,
        context_length: int = 168,
        forecast_length: int = 24,
        batch_size: int = 32,
        num_workers: int = 4,
        multi_zone: bool = False,
    ) -> None:
        """Initialize DataModule.

        Args:
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            statistics_path: Path to statistics.json
            context_length: Historical context length
            forecast_length: Forecast horizon
            batch_size: Batch size
            num_workers: Number of data loading workers
            multi_zone: Whether to use multi-zone dataset
        """
        super().__init__()
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.statistics_path = Path(statistics_path)
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.multi_zone = multi_zone

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets."""
        from diffcast.data.dataset import DiffCastDataset, MultiZoneDataset

        DatasetClass = MultiZoneDataset if self.multi_zone else DiffCastDataset

        if stage == "fit" or stage is None:
            self.train_dataset = DatasetClass(
                data_path=self.train_path,
                context_length=self.context_length,
                forecast_length=self.forecast_length,
                statistics_path=self.statistics_path,
            )
            self.val_dataset = DatasetClass(
                data_path=self.val_path,
                context_length=self.context_length,
                forecast_length=self.forecast_length,
                statistics_path=self.statistics_path,
            )

        if stage == "test" or stage is None:
            self.test_dataset = DatasetClass(
                data_path=self.test_path,
                context_length=self.context_length,
                forecast_length=self.forecast_length,
                statistics_path=self.statistics_path,
            )

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
