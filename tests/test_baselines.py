"""Tests for baseline models."""

import numpy as np
import pytest

from diffcast.models.baselines import (
    PersistenceModel,
    SeasonalPersistenceModel,
)


class TestPersistenceModel:
    """Tests for naive persistence model."""

    def test_prediction_shape(self):
        """Test prediction output shape."""
        model = PersistenceModel(forecast_length=24)

        context = np.random.randn(10, 168)  # batch=10, history=168
        n_samples = 50

        pred = model.predict(context, n_samples=n_samples)

        assert pred.shape == (10, n_samples, 24)

    def test_uses_last_day(self):
        """Test that persistence uses last 24 hours."""
        model = PersistenceModel(forecast_length=24)

        # Create context with known last day
        context = np.zeros(168)
        context[-24:] = np.arange(24)

        pred = model.predict(context, n_samples=1)

        expected = np.arange(24)
        assert np.allclose(pred[0, 0], expected)

    def test_single_instance(self):
        """Test prediction for single instance."""
        model = PersistenceModel(forecast_length=24)

        context = np.random.randn(168)
        pred = model.predict(context, n_samples=10)

        assert pred.shape == (10, 24)


class TestSeasonalPersistenceModel:
    """Tests for seasonal persistence model."""

    def test_prediction_shape(self):
        """Test prediction output shape."""
        model = SeasonalPersistenceModel(forecast_length=24, seasonal_period=168)

        context = np.random.randn(10, 168)
        n_samples = 50

        pred = model.predict(context, n_samples=n_samples)

        assert pred.shape == (10, n_samples, 24)

    def test_uses_week_ago(self):
        """Test that seasonal model uses values from one week ago."""
        model = SeasonalPersistenceModel(forecast_length=24, seasonal_period=168)

        # Create context with known pattern
        context = np.zeros(168)
        context[:24] = np.arange(24)  # First day (one week ago)

        pred = model.predict(context, n_samples=1)

        expected = np.arange(24)
        assert np.allclose(pred[0, 0], expected)

    def test_fallback_short_history(self):
        """Test fallback when history is shorter than seasonal period."""
        model = SeasonalPersistenceModel(forecast_length=24, seasonal_period=168)

        # Short context (less than one week)
        context = np.random.randn(48)

        pred = model.predict(context, n_samples=1)

        # Should fall back to last day
        expected = context[-24:]
        assert np.allclose(pred[0, 0], expected)
