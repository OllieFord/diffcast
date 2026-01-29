"""Quantile regression baseline for probabilistic forecasting."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class QuantileRegression(nn.Module):
    """Linear quantile regression model.

    Predicts multiple quantiles simultaneously for probabilistic forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 24,
        quantiles: list[float] | None = None,
        hidden_dim: int = 128,
    ) -> None:
        """Initialize quantile regression model.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (forecast length)
            quantiles: Quantile levels to predict
            hidden_dim: Hidden layer dimension (0 for pure linear)
        """
        super().__init__()

        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.n_quantiles = len(self.quantiles)
        self.output_dim = output_dim

        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim * self.n_quantiles),
            )
        else:
            self.net = nn.Linear(input_dim, output_dim * self.n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (batch, input_dim)

        Returns:
            Quantile predictions of shape (batch, n_quantiles, output_dim)
        """
        out = self.net(x)
        return out.view(-1, self.n_quantiles, self.output_dim)

    def pinball_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pinball loss for all quantiles.

        Args:
            pred: Predictions of shape (batch, n_quantiles, output_dim)
            target: Targets of shape (batch, output_dim)

        Returns:
            Total pinball loss
        """
        target = target.unsqueeze(1)  # (batch, 1, output_dim)
        diff = target - pred

        losses = []
        for i, q in enumerate(self.quantiles):
            q_diff = diff[:, i, :]
            loss = torch.maximum(q * q_diff, (q - 1) * q_diff)
            losses.append(loss.mean())

        return sum(losses) / len(losses)

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        verbose: bool = True,
    ) -> list[float]:
        """Train the model.

        Args:
            X: Input features of shape (n_samples, input_dim)
            y: Targets of shape (n_samples, output_dim)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Whether to print progress

        Returns:
            List of training losses
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self(batch_x)
                loss = self.pinball_loss(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            scheduler.step(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return losses

    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray | torch.Tensor,
        n_samples: int = 100,
    ) -> np.ndarray:
        """Generate predictions by sampling from predicted quantiles.

        Interpolates between predicted quantiles to generate samples.

        Args:
            X: Input features
            n_samples: Number of samples to generate

        Returns:
            Samples of shape (batch, n_samples, output_dim)
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        self.eval()
        quantile_preds = self(X).numpy()  # (batch, n_quantiles, output_dim)

        batch_size = quantile_preds.shape[0]
        samples = np.zeros((batch_size, n_samples, self.output_dim))

        # Sample by interpolating between quantiles
        for b in range(batch_size):
            for t in range(self.output_dim):
                q_values = quantile_preds[b, :, t]
                # Generate uniform samples and interpolate
                u = np.random.uniform(0, 1, n_samples)
                samples[b, :, t] = np.interp(u, self.quantiles, q_values)

        return samples

    @torch.no_grad()
    def predict_quantiles(
        self,
        X: np.ndarray | torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """Get quantile predictions directly.

        Args:
            X: Input features

        Returns:
            Dictionary mapping quantile names to predictions
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        self.eval()
        quantile_preds = self(X).numpy()

        return {
            f"q{int(q * 100)}": quantile_preds[:, i, :]
            for i, q in enumerate(self.quantiles)
        }


def prepare_features_for_qr(
    context: np.ndarray,
    covariates: np.ndarray | None = None,
    include_lags: bool = True,
    lag_hours: list[int] | None = None,
) -> np.ndarray:
    """Prepare features for quantile regression.

    Args:
        context: Historical context (batch, context_length) or (batch, context_length, n_features)
        covariates: Optional covariates (batch, forecast_length, n_covariates)
        include_lags: Whether to include lag features
        lag_hours: Specific lag hours to include

    Returns:
        Feature matrix for quantile regression
    """
    if context.ndim == 2:
        # Only prices, add feature dimension
        context = context[:, :, np.newaxis]

    batch_size, context_length, n_features = context.shape
    lag_hours = lag_hours or [1, 24, 48, 168]  # 1h, 1d, 2d, 1w

    features = []

    # Lag features (last n values of each feature)
    if include_lags:
        for lag in lag_hours:
            if lag <= context_length:
                features.append(context[:, -lag, :])

    # Mean and std of recent history
    for window in [24, 168]:
        if window <= context_length:
            window_data = context[:, -window:, :]
            features.append(window_data.mean(axis=1))
            features.append(window_data.std(axis=1))

    # Flatten covariates if provided
    if covariates is not None:
        cov_flat = covariates.reshape(batch_size, -1)
        features.append(cov_flat)

    return np.concatenate(features, axis=1)
