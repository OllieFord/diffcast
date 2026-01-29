"""Loss functions for DiffCast training."""

import torch
import torch.nn.functional as F


def noise_prediction_loss(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE loss for noise prediction.

    Args:
        noise_pred: Predicted noise
        noise: True noise
        mask: Optional mask for valid positions
        reduction: Reduction method ('mean', 'sum', 'none')

    Returns:
        Loss value
    """
    loss = (noise_pred - noise) ** 2

    if mask is not None:
        loss = loss * mask
        if reduction == "mean":
            return loss.sum() / (mask.sum() + 1e-8)
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def crps_loss(
    samples: torch.Tensor,
    target: torch.Tensor,
    sorted_samples: bool = False,
) -> torch.Tensor:
    """Continuous Ranked Probability Score (CRPS) loss.

    Approximates CRPS using samples from the predictive distribution.

    Args:
        samples: Samples from predictive distribution (batch, n_samples, ...)
        target: True values (batch, ...)
        sorted_samples: Whether samples are already sorted

    Returns:
        CRPS loss value
    """
    n_samples = samples.shape[1]

    if not sorted_samples:
        samples = torch.sort(samples, dim=1).values

    # Expand target for broadcasting
    target = target.unsqueeze(1)

    # MAE term: E[|X - y|]
    mae_term = torch.abs(samples - target).mean(dim=1)

    # Dispersion term: 0.5 * E[|X - X'|]
    # Use efficient computation for sorted samples
    diff = samples[:, 1:] - samples[:, :-1]  # (batch, n_samples-1, ...)
    weights = torch.arange(1, n_samples, device=samples.device).float()
    weights = weights * (n_samples - weights)  # Weight by number of pairs
    weights = weights.view(1, -1, *([1] * (diff.dim() - 2)))
    dispersion = (diff * weights).sum(dim=1) / (n_samples * (n_samples - 1) / 2)

    crps = mae_term - 0.5 * dispersion
    return crps.mean()


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pinball (quantile) loss.

    Args:
        pred: Predicted quantile value
        target: True value
        quantile: Quantile level (0-1)
        mask: Optional mask for valid positions

    Returns:
        Pinball loss value
    """
    diff = target - pred
    loss = torch.maximum(quantile * diff, (quantile - 1) * diff)

    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)

    return loss.mean()


def multi_quantile_loss(
    samples: torch.Tensor,
    target: torch.Tensor,
    quantiles: list[float] | None = None,
) -> torch.Tensor:
    """Combined loss over multiple quantiles.

    Args:
        samples: Samples from predictive distribution (batch, n_samples, ...)
        target: True values (batch, ...)
        quantiles: Quantile levels to evaluate

    Returns:
        Average pinball loss across quantiles
    """
    quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]

    total_loss = 0.0
    for q in quantiles:
        pred_quantile = torch.quantile(samples, q, dim=1)
        total_loss = total_loss + pinball_loss(pred_quantile, target, q)

    return total_loss / len(quantiles)


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted MSE loss.

    Args:
        pred: Predictions
        target: Targets
        weights: Optional weights for each position

    Returns:
        Weighted MSE loss
    """
    loss = (pred - target) ** 2

    if weights is not None:
        loss = loss * weights
        return loss.sum() / (weights.sum() + 1e-8)

    return loss.mean()
