"""Visualization utilities for DiffCast forecasts."""

from datetime import datetime
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_fan_chart(
    timestamps: Sequence[datetime] | np.ndarray,
    samples: np.ndarray,
    target: np.ndarray | None = None,
    context: np.ndarray | None = None,
    context_timestamps: Sequence[datetime] | np.ndarray | None = None,
    quantiles: list[float] | None = None,
    title: str = "Day-Ahead Price Forecast",
    ylabel: str = "Price (EUR/MWh)",
    figsize: tuple[int, int] = (12, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot fan chart showing probabilistic forecast.

    Args:
        timestamps: Forecast timestamps
        samples: Forecast samples (n_samples, forecast_length)
        target: Optional actual values for comparison
        context: Optional historical context prices
        context_timestamps: Timestamps for context
        quantiles: Quantile levels for intervals (default: [0.1, 0.25, 0.5, 0.75, 0.9])
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size
        ax: Optional existing axes

    Returns:
        matplotlib Figure
    """
    quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Convert timestamps to array if needed
    timestamps = np.array(timestamps)

    # Plot context (history)
    if context is not None and context_timestamps is not None:
        context_timestamps = np.array(context_timestamps)
        ax.plot(
            context_timestamps,
            context,
            color="black",
            linewidth=1.5,
            label="History",
        )

    # Compute quantiles
    quantile_values = {q: np.quantile(samples, q, axis=0) for q in quantiles}

    # Plot intervals (fan)
    colors = plt.cm.Blues(np.linspace(0.2, 0.6, len(quantiles) // 2))

    for i, (lower_q, upper_q) in enumerate(zip(quantiles[: len(quantiles) // 2], quantiles[::-1])):
        if lower_q >= upper_q:
            break
        ax.fill_between(
            timestamps,
            quantile_values[lower_q],
            quantile_values[upper_q],
            color=colors[i],
            alpha=0.4,
            label=f"{int((upper_q - lower_q) * 100)}% interval",
        )

    # Plot median
    median = quantile_values.get(0.5, np.median(samples, axis=0))
    ax.plot(
        timestamps,
        median,
        color="blue",
        linewidth=2,
        label="Median forecast",
    )

    # Plot actual if provided
    if target is not None:
        ax.plot(
            timestamps,
            target,
            color="red",
            linewidth=2,
            linestyle="--",
            label="Actual",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig


def plot_calibration(
    samples: np.ndarray,
    targets: np.ndarray,
    quantiles: list[float] | None = None,
    title: str = "Calibration Plot",
    figsize: tuple[int, int] = (8, 8),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot calibration curve (expected vs observed coverage).

    Args:
        samples: Forecast samples (batch, n_samples, forecast_length)
        targets: Actual values (batch, forecast_length)
        quantiles: Quantile levels to evaluate
        title: Plot title
        figsize: Figure size
        ax: Optional existing axes

    Returns:
        matplotlib Figure
    """
    quantiles = quantiles or np.linspace(0.05, 0.95, 19).tolist()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Flatten samples and targets
    samples_flat = samples.reshape(samples.shape[0], samples.shape[1], -1)
    targets_flat = targets.reshape(targets.shape[0], -1)

    observed_coverages = []
    expected_coverages = []

    for q in quantiles:
        # For each quantile q, compute (1-q) central interval coverage
        lower_q = (1 - q) / 2
        upper_q = 1 - lower_q

        lower = np.quantile(samples_flat, lower_q, axis=1)
        upper = np.quantile(samples_flat, upper_q, axis=1)

        covered = (targets_flat >= lower) & (targets_flat <= upper)
        observed = covered.mean()

        observed_coverages.append(observed)
        expected_coverages.append(q)

    # Plot
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(expected_coverages, observed_coverages, "o-", color="blue", label="Model")

    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return fig


def plot_pit_histogram(
    samples: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 20,
    title: str = "PIT Histogram",
    figsize: tuple[int, int] = (8, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot Probability Integral Transform (PIT) histogram.

    A uniform histogram indicates good calibration.

    Args:
        samples: Forecast samples
        targets: Actual values
        n_bins: Number of histogram bins
        title: Plot title
        figsize: Figure size
        ax: Optional existing axes

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Compute PIT values
    samples_flat = samples.reshape(samples.shape[0], samples.shape[1], -1)
    targets_flat = targets.reshape(targets.shape[0], -1)

    pit_values = []
    for i in range(targets_flat.shape[0]):
        for j in range(targets_flat.shape[1]):
            # Empirical CDF at target value
            pit = (samples_flat[i, :, j] <= targets_flat[i, j]).mean()
            pit_values.append(pit)

    # Plot histogram
    ax.hist(pit_values, bins=n_bins, density=True, alpha=0.7, color="blue", edgecolor="black")
    ax.axhline(y=1.0, color="red", linestyle="--", label="Uniform")

    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim([0, 1])

    return fig


def plot_forecast_comparison(
    timestamps: Sequence[datetime] | np.ndarray,
    forecasts: dict[str, np.ndarray],
    target: np.ndarray,
    title: str = "Forecast Comparison",
    ylabel: str = "Price (EUR/MWh)",
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Compare multiple forecasts.

    Args:
        timestamps: Forecast timestamps
        forecasts: Dictionary mapping model name to samples or predictions
        target: Actual values
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    timestamps = np.array(timestamps)
    colors = plt.cm.Set2(np.linspace(0, 1, len(forecasts)))

    for (name, forecast), color in zip(forecasts.items(), colors):
        if forecast.ndim > 1:
            # Probabilistic forecast - plot median and interval
            median = np.median(forecast, axis=0)
            lower = np.quantile(forecast, 0.1, axis=0)
            upper = np.quantile(forecast, 0.9, axis=0)

            ax.fill_between(timestamps, lower, upper, color=color, alpha=0.3)
            ax.plot(timestamps, median, color=color, linewidth=2, label=name)
        else:
            # Point forecast
            ax.plot(timestamps, forecast, color=color, linewidth=2, label=name)

    # Plot actual
    ax.plot(timestamps, target, color="black", linewidth=2, linestyle="--", label="Actual")

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig


def plot_zone_comparison(
    zone_forecasts: dict[str, np.ndarray],
    zone_targets: dict[str, np.ndarray],
    timestamps: Sequence[datetime] | np.ndarray,
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    """Plot forecasts for multiple zones.

    Args:
        zone_forecasts: Dictionary mapping zone -> samples
        zone_targets: Dictionary mapping zone -> actual values
        timestamps: Forecast timestamps
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_zones = len(zone_forecasts)
    n_cols = min(3, n_zones)
    n_rows = (n_zones + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    timestamps = np.array(timestamps)

    for ax, (zone, samples) in zip(axes, zone_forecasts.items()):
        target = zone_targets.get(zone)

        # Plot median and intervals
        median = np.median(samples, axis=0)
        lower = np.quantile(samples, 0.1, axis=0)
        upper = np.quantile(samples, 0.9, axis=0)

        ax.fill_between(timestamps, lower, upper, color="blue", alpha=0.3, label="80% interval")
        ax.plot(timestamps, median, color="blue", linewidth=2, label="Median")

        if target is not None:
            ax.plot(timestamps, target, color="red", linewidth=2, linestyle="--", label="Actual")

        ax.set_title(zone)
        ax.set_xlabel("Time")
        ax.set_ylabel("EUR/MWh")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    # Hide unused axes
    for ax in axes[n_zones:]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig
