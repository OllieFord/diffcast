"""Evaluation metrics for probabilistic forecasting."""

import numpy as np
import torch


def crps(
    samples: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> float:
    """Continuous Ranked Probability Score (CRPS).

    Lower is better. Measures both calibration and sharpness.

    Args:
        samples: Samples from predictive distribution (batch, n_samples, ...)
            or (n_samples, ...) for single instance
        target: True values (batch, ...) or (...)

    Returns:
        Mean CRPS value
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Ensure samples has batch dimension
    if samples.ndim == 2 and target.ndim == 1:
        # Single sequence: (n_samples, seq_len) and (seq_len,)
        samples = samples[np.newaxis, ...]
        target = target[np.newaxis, ...]

    batch_size, n_samples = samples.shape[:2]
    other_dims = samples.shape[2:]

    # Sort samples
    samples_sorted = np.sort(samples, axis=1)

    # Expand target for broadcasting
    target_expanded = np.expand_dims(target, axis=1)

    # MAE term: E[|X - y|]
    mae_term = np.abs(samples - target_expanded).mean(axis=1)

    # Gini coefficient term for dispersion
    # For sorted samples: sum of (2i - n - 1) * x_i / (n^2)
    weights = 2 * np.arange(1, n_samples + 1) - n_samples - 1
    weights = weights.reshape((1, n_samples) + (1,) * len(other_dims))
    gini = (samples_sorted * weights).sum(axis=1) / (n_samples**2)

    crps_values = mae_term - gini
    return float(crps_values.mean())


def pinball_loss(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    quantile: float,
) -> float:
    """Pinball (quantile) loss.

    Args:
        pred: Predicted quantile values
        target: True values
        quantile: Quantile level (0-1)

    Returns:
        Mean pinball loss
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    diff = target - pred
    loss = np.maximum(quantile * diff, (quantile - 1) * diff)
    return float(loss.mean())


def calibration_error(
    samples: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    quantiles: list[float] | None = None,
) -> dict[str, float]:
    """Compute calibration error for prediction intervals.

    Args:
        samples: Samples from predictive distribution
        target: True values
        quantiles: Quantile levels to evaluate

    Returns:
        Dictionary with observed coverage and calibration error per interval
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    quantiles = quantiles or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = {}

    for q in quantiles:
        lower_q = (1 - q) / 2
        upper_q = 1 - lower_q

        lower = np.quantile(samples, lower_q, axis=1)
        upper = np.quantile(samples, upper_q, axis=1)

        # Expand target for comparison
        if target.ndim < lower.ndim:
            target_cmp = np.expand_dims(target, axis=tuple(range(1, lower.ndim)))
        else:
            target_cmp = target

        covered = (target_cmp >= lower) & (target_cmp <= upper)
        observed_coverage = covered.mean()
        expected_coverage = q

        results[f"coverage_{int(q * 100)}"] = float(observed_coverage)
        results[f"calib_error_{int(q * 100)}"] = float(abs(observed_coverage - expected_coverage))

    # Average calibration error
    calib_errors = [v for k, v in results.items() if k.startswith("calib_error")]
    results["mean_calibration_error"] = float(np.mean(calib_errors))

    return results


def coverage(
    samples: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    interval: float = 0.9,
) -> float:
    """Compute coverage of prediction interval.

    Args:
        samples: Samples from predictive distribution
        target: True values
        interval: Prediction interval width (e.g., 0.9 for 90% interval)

    Returns:
        Observed coverage (fraction of targets within interval)
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    lower_q = (1 - interval) / 2
    upper_q = 1 - lower_q

    lower = np.quantile(samples, lower_q, axis=1)
    upper = np.quantile(samples, upper_q, axis=1)

    covered = (target >= lower) & (target <= upper)
    return float(covered.mean())


def winkler_score(
    samples: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    interval: float = 0.9,
) -> float:
    """Winkler score for interval sharpness.

    Lower is better. Rewards narrow intervals that contain the target.

    Args:
        samples: Samples from predictive distribution
        target: True values
        interval: Prediction interval width

    Returns:
        Mean Winkler score
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    alpha = 1 - interval
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    lower = np.quantile(samples, lower_q, axis=1)
    upper = np.quantile(samples, upper_q, axis=1)

    width = upper - lower

    # Penalty for observations outside interval
    below = target < lower
    above = target > upper

    score = width.copy()
    score[below] += (2 / alpha) * (lower[below] - target[below])
    score[above] += (2 / alpha) * (target[above] - upper[above])

    return float(score.mean())


def mae(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> float:
    """Mean Absolute Error.

    Args:
        pred: Predictions
        target: True values

    Returns:
        MAE value
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    return float(np.abs(pred - target).mean())


def rmse(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> float:
    """Root Mean Squared Error.

    Args:
        pred: Predictions
        target: True values

    Returns:
        RMSE value
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    return float(np.sqrt(((pred - target) ** 2).mean()))


def evaluate_forecast(
    samples: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    quantiles: list[float] | None = None,
) -> dict[str, float]:
    """Comprehensive evaluation of probabilistic forecast.

    Args:
        samples: Samples from predictive distribution
        target: True values
        quantiles: Quantile levels for pinball loss

    Returns:
        Dictionary with all metrics
    """
    quantiles = quantiles or [0.1, 0.5, 0.9]

    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    results = {}

    # Point forecast metrics (median)
    median_pred = np.median(samples, axis=1)
    results["mae"] = mae(median_pred, target)
    results["rmse"] = rmse(median_pred, target)

    # Probabilistic metrics
    results["crps"] = crps(samples, target)

    # Pinball loss for each quantile
    for q in quantiles:
        q_pred = np.quantile(samples, q, axis=1)
        results[f"pinball_{int(q * 100)}"] = pinball_loss(q_pred, target, q)

    # Coverage and calibration
    for interval in [0.5, 0.8, 0.9]:
        results[f"coverage_{int(interval * 100)}"] = coverage(samples, target, interval)
        results[f"winkler_{int(interval * 100)}"] = winkler_score(samples, target, interval)

    # Calibration error
    calib = calibration_error(samples, target)
    results["calibration_error"] = calib["mean_calibration_error"]

    return results
