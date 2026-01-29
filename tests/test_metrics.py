"""Tests for evaluation metrics."""

import numpy as np
import pytest

from diffcast.evaluation.metrics import (
    calibration_error,
    coverage,
    crps,
    mae,
    pinball_loss,
    rmse,
    winkler_score,
)


class TestPointMetrics:
    """Tests for point forecast metrics."""

    def test_mae(self):
        """Test MAE calculation."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.5, 2.0, 2.5])

        result = mae(pred, target)

        expected = np.mean(np.abs(pred - target))
        assert np.isclose(result, expected)

    def test_rmse(self):
        """Test RMSE calculation."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.5, 2.0, 2.5])

        result = rmse(pred, target)

        expected = np.sqrt(np.mean((pred - target) ** 2))
        assert np.isclose(result, expected)

    def test_mae_perfect_prediction(self):
        """Test MAE is zero for perfect predictions."""
        pred = np.array([1.0, 2.0, 3.0])
        target = pred.copy()

        assert mae(pred, target) == 0.0


class TestProbabilisticMetrics:
    """Tests for probabilistic metrics."""

    def test_crps_deterministic(self):
        """Test CRPS for deterministic (single sample) forecast."""
        # When all samples are identical, CRPS should equal MAE
        samples = np.ones((10, 100, 24)) * 5.0  # All samples = 5
        targets = np.ones((10, 24)) * 6.0  # All targets = 6

        result = crps(samples, targets)

        # CRPS = MAE when forecast is deterministic
        assert np.isclose(result, 1.0, atol=0.1)

    def test_crps_perfect_forecast(self):
        """Test CRPS is zero for perfect probabilistic forecast."""
        # Create samples centered on target
        targets = np.array([[5.0] * 24])
        samples = np.tile(targets, (1, 100, 1))  # Perfect samples

        result = crps(samples, targets)

        assert result < 0.01

    def test_pinball_loss_median(self):
        """Test pinball loss at median (q=0.5)."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.5, 2.0, 2.5])

        result = pinball_loss(pred, target, 0.5)

        # At q=0.5, pinball loss equals 0.5 * MAE
        expected = 0.5 * mae(pred, target)
        assert np.isclose(result, expected)

    def test_coverage_perfect_interval(self):
        """Test coverage for interval that contains all targets."""
        samples = np.random.randn(10, 100, 24) * 0.1  # Very narrow around 0
        samples += 5.0  # Centered at 5
        targets = np.ones((10, 24)) * 5.0

        result = coverage(samples, targets, interval=0.9)

        # All targets should be covered
        assert result > 0.8

    def test_winkler_score_properties(self):
        """Test Winkler score properties."""
        # Narrow interval that misses target should have high score
        samples_narrow = np.ones((10, 100, 24)) * 5.0 + np.random.randn(10, 100, 24) * 0.1
        targets = np.ones((10, 24)) * 10.0  # Far from samples

        # Wide interval should have lower score
        samples_wide = np.ones((10, 100, 24)) * 5.0 + np.random.randn(10, 100, 24) * 10.0

        score_narrow = winkler_score(samples_narrow, targets)
        score_wide = winkler_score(samples_wide, targets)

        # Narrow interval missing target should score worse
        assert score_narrow > score_wide


class TestCalibration:
    """Tests for calibration metrics."""

    def test_calibration_well_calibrated(self):
        """Test calibration for well-calibrated forecast."""
        np.random.seed(42)

        # Generate samples from a distribution
        n_samples = 1000
        n_forecast = 24

        # True distribution: N(0, 1)
        samples = np.random.randn(100, n_samples, n_forecast)
        targets = np.random.randn(100, n_forecast)

        result = calibration_error(samples, targets)

        # Should have low calibration error
        assert result["mean_calibration_error"] < 0.1

    def test_coverage_values_valid(self):
        """Test that coverage values are in [0, 1]."""
        samples = np.random.randn(10, 100, 24)
        targets = np.random.randn(10, 24)

        result = calibration_error(samples, targets)

        for key, value in result.items():
            if key.startswith("coverage"):
                assert 0 <= value <= 1
