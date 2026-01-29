"""Persistence baseline models for forecasting."""

import numpy as np
import torch


class PersistenceModel:
    """Naive persistence model: forecast = last observed value.

    For day-ahead forecasting, predicts tomorrow's prices as today's prices.
    """

    def __init__(self, forecast_length: int = 24) -> None:
        """Initialize persistence model.

        Args:
            forecast_length: Number of hours to forecast
        """
        self.forecast_length = forecast_length

    def predict(
        self,
        context: np.ndarray | torch.Tensor,
        n_samples: int = 1,
    ) -> np.ndarray:
        """Generate predictions.

        Args:
            context: Historical prices of shape (batch, context_length) or (context_length,)
            n_samples: Number of samples (for API compatibility, always returns same value)

        Returns:
            Predictions of shape (batch, n_samples, forecast_length) or (n_samples, forecast_length)
        """
        if isinstance(context, torch.Tensor):
            context = context.detach().cpu().numpy()

        # Get last forecast_length values
        single_instance = context.ndim == 1
        if single_instance:
            context = context[np.newaxis, :]

        last_day = context[:, -self.forecast_length :]

        # Repeat for n_samples
        predictions = np.tile(last_day[:, np.newaxis, :], (1, n_samples, 1))

        if single_instance:
            predictions = predictions[0]

        return predictions

    def __call__(self, context: np.ndarray | torch.Tensor, n_samples: int = 1) -> np.ndarray:
        return self.predict(context, n_samples)


class SeasonalPersistenceModel:
    """Seasonal naive model: forecast = same time last week.

    Uses prices from exactly 7 days ago.
    """

    def __init__(
        self,
        forecast_length: int = 24,
        seasonal_period: int = 168,  # 7 days * 24 hours
    ) -> None:
        """Initialize seasonal persistence model.

        Args:
            forecast_length: Number of hours to forecast
            seasonal_period: Seasonal period in hours (default: 1 week)
        """
        self.forecast_length = forecast_length
        self.seasonal_period = seasonal_period

    def predict(
        self,
        context: np.ndarray | torch.Tensor,
        n_samples: int = 1,
    ) -> np.ndarray:
        """Generate predictions.

        Args:
            context: Historical prices of shape (batch, context_length) or (context_length,)
            n_samples: Number of samples

        Returns:
            Predictions of shape (batch, n_samples, forecast_length) or (n_samples, forecast_length)
        """
        if isinstance(context, torch.Tensor):
            context = context.detach().cpu().numpy()

        single_instance = context.ndim == 1
        if single_instance:
            context = context[np.newaxis, :]

        # Get values from one seasonal period ago
        if context.shape[1] >= self.seasonal_period:
            # Use values from one week ago (relative to forecast start)
            start_idx = context.shape[1] - self.seasonal_period
            seasonal_values = context[:, start_idx : start_idx + self.forecast_length]
        else:
            # Not enough history, fall back to last day
            seasonal_values = context[:, -self.forecast_length :]

        # Repeat for n_samples
        predictions = np.tile(seasonal_values[:, np.newaxis, :], (1, n_samples, 1))

        if single_instance:
            predictions = predictions[0]

        return predictions

    def __call__(self, context: np.ndarray | torch.Tensor, n_samples: int = 1) -> np.ndarray:
        return self.predict(context, n_samples)


class WeightedPersistenceModel:
    """Weighted combination of recent values and seasonal values."""

    def __init__(
        self,
        forecast_length: int = 24,
        seasonal_period: int = 168,
        seasonal_weight: float = 0.5,
    ) -> None:
        """Initialize weighted persistence model.

        Args:
            forecast_length: Number of hours to forecast
            seasonal_period: Seasonal period in hours
            seasonal_weight: Weight for seasonal component (0-1)
        """
        self.forecast_length = forecast_length
        self.seasonal_period = seasonal_period
        self.seasonal_weight = seasonal_weight

    def predict(
        self,
        context: np.ndarray | torch.Tensor,
        n_samples: int = 1,
    ) -> np.ndarray:
        """Generate predictions."""
        if isinstance(context, torch.Tensor):
            context = context.detach().cpu().numpy()

        single_instance = context.ndim == 1
        if single_instance:
            context = context[np.newaxis, :]

        # Recent component (last day)
        recent = context[:, -self.forecast_length :]

        # Seasonal component (same time last week)
        if context.shape[1] >= self.seasonal_period:
            start_idx = context.shape[1] - self.seasonal_period
            seasonal = context[:, start_idx : start_idx + self.forecast_length]
        else:
            seasonal = recent

        # Weighted combination
        predictions = (
            self.seasonal_weight * seasonal + (1 - self.seasonal_weight) * recent
        )

        # Repeat for n_samples
        predictions = np.tile(predictions[:, np.newaxis, :], (1, n_samples, 1))

        if single_instance:
            predictions = predictions[0]

        return predictions

    def __call__(self, context: np.ndarray | torch.Tensor, n_samples: int = 1) -> np.ndarray:
        return self.predict(context, n_samples)
