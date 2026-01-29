"""Inference pipeline for DiffCast."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from diffcast.data.schema import ZONES
from diffcast.models.csdi import CSDI


class Forecaster:
    """High-level forecasting interface for DiffCast."""

    def __init__(
        self,
        model: CSDI,
        statistics_path: Path | str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize forecaster.

        Args:
            model: Trained CSDI model
            statistics_path: Path to statistics.json for normalization
            device: Device to run inference on
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Load statistics
        self.statistics = {}
        if statistics_path:
            with open(statistics_path) as f:
                self.statistics = json.load(f)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        statistics_path: Path | str | None = None,
        device: str | torch.device = "cpu",
        **model_kwargs,
    ) -> "Forecaster":
        """Load forecaster from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            statistics_path: Path to statistics.json
            device: Device to run inference on
            **model_kwargs: Model configuration overrides

        Returns:
            Initialized Forecaster
        """
        from diffcast.training.trainer import DiffCastLightningModule

        # Load Lightning module
        module = DiffCastLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            **model_kwargs,
        )

        return cls(
            model=module.model,
            statistics_path=statistics_path,
            device=device,
        )

    def _normalize(self, values: np.ndarray, col: str) -> np.ndarray:
        """Normalize values."""
        if col in self.statistics:
            mean = self.statistics[col]["mean"]
            std = self.statistics[col]["std"]
            return (values - mean) / std
        return values

    def _denormalize(self, values: np.ndarray, col: str) -> np.ndarray:
        """Denormalize values."""
        if col in self.statistics:
            mean = self.statistics[col]["mean"]
            std = self.statistics[col]["std"]
            return values * std + mean
        return values

    def _prepare_input(
        self,
        context: np.ndarray | pl.DataFrame,
        covariates: np.ndarray | None = None,
        zone: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Prepare input tensors.

        Args:
            context: Historical features
            covariates: Optional future covariates
            zone: Optional zone identifier

        Returns:
            Dictionary with input tensors
        """
        # Convert DataFrame to numpy if needed
        if isinstance(context, pl.DataFrame):
            feature_cols = [
                "price_eur_mwh", "load_mw", "wind_mw", "solar_mw",
                "hydro_mw", "temp_c", "hour", "dow", "month", "is_holiday"
            ]
            context_data = []
            for col in feature_cols:
                if col in context.columns:
                    vals = context[col].to_numpy()
                    if col == "is_holiday":
                        vals = vals.astype(float)
                    vals = self._normalize(vals, col)
                    context_data.append(vals)
                else:
                    context_data.append(np.zeros(len(context)))
            context = np.stack(context_data, axis=-1)

        # Add batch dimension if needed
        if context.ndim == 2:
            context = context[np.newaxis, ...]

        # Convert to tensor
        context_tensor = torch.tensor(context, dtype=torch.float32, device=self.device)

        result = {"context": context_tensor}

        # Covariates
        if covariates is not None:
            if covariates.ndim == 2:
                covariates = covariates[np.newaxis, ...]
            result["covariates"] = torch.tensor(
                covariates, dtype=torch.float32, device=self.device
            )

        # Zone index
        if zone is not None:
            zone_idx = ZONES.index(zone) if zone in ZONES else 0
            result["zone_idx"] = torch.tensor([zone_idx], device=self.device)

        return result

    @torch.no_grad()
    def forecast(
        self,
        context: np.ndarray | pl.DataFrame,
        covariates: np.ndarray | None = None,
        zone: str | None = None,
        n_samples: int = 100,
        use_ddim: bool = True,
        ddim_steps: int = 20,
        return_quantiles: bool = True,
        quantiles: list[float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate forecast.

        Args:
            context: Historical features
            covariates: Optional future covariates
            zone: Optional zone identifier
            n_samples: Number of samples to generate
            use_ddim: Whether to use DDIM sampling
            ddim_steps: Number of DDIM steps
            return_quantiles: Whether to return quantiles
            quantiles: Quantile levels

        Returns:
            Dictionary with forecasts:
                - samples: (n_samples, forecast_length)
                - median: (forecast_length,)
                - quantiles: dict mapping quantile -> (forecast_length,)
        """
        quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]

        # Prepare inputs
        inputs = self._prepare_input(context, covariates, zone)

        # Generate samples
        samples = self.model.sample(
            context=inputs["context"],
            covariates=inputs.get("covariates"),
            zone_idx=inputs.get("zone_idx"),
            n_samples=n_samples,
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
        )

        # Convert to numpy and denormalize
        samples = samples.cpu().numpy()[0]  # Remove batch dim
        samples = self._denormalize(samples, "price_eur_mwh")

        result = {
            "samples": samples,
            "median": np.median(samples, axis=0),
            "mean": np.mean(samples, axis=0),
        }

        if return_quantiles:
            result["quantiles"] = {
                f"q{int(q * 100)}": np.quantile(samples, q, axis=0)
                for q in quantiles
            }

        return result

    @torch.no_grad()
    def forecast_all_zones(
        self,
        context_dict: dict[str, np.ndarray | pl.DataFrame],
        covariates_dict: dict[str, np.ndarray] | None = None,
        n_samples: int = 100,
        use_ddim: bool = True,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Generate forecasts for all zones.

        Args:
            context_dict: Dictionary mapping zone -> historical features
            covariates_dict: Optional dictionary mapping zone -> covariates
            n_samples: Number of samples
            use_ddim: Whether to use DDIM sampling

        Returns:
            Dictionary mapping zone -> forecast dictionary
        """
        results = {}

        for zone, context in context_dict.items():
            covariates = covariates_dict.get(zone) if covariates_dict else None
            results[zone] = self.forecast(
                context=context,
                covariates=covariates,
                zone=zone,
                n_samples=n_samples,
                use_ddim=use_ddim,
            )

        return results

    def get_prediction_intervals(
        self,
        forecast: dict[str, np.ndarray],
        levels: list[float] | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Extract prediction intervals from forecast.

        Args:
            forecast: Forecast dictionary with samples
            levels: Interval levels (e.g., [0.5, 0.8, 0.9] for 50%, 80%, 90%)

        Returns:
            Dictionary mapping interval level -> (lower, upper) bounds
        """
        levels = levels or [0.5, 0.8, 0.9]
        samples = forecast["samples"]

        intervals = {}
        for level in levels:
            lower_q = (1 - level) / 2
            upper_q = 1 - lower_q
            intervals[f"{int(level * 100)}%"] = (
                np.quantile(samples, lower_q, axis=0),
                np.quantile(samples, upper_q, axis=0),
            )

        return intervals
