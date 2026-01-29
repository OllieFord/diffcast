"""Baseline models for comparison."""

from diffcast.models.baselines.persistence import PersistenceModel, SeasonalPersistenceModel
from diffcast.models.baselines.quantile_regression import QuantileRegression

__all__ = ["PersistenceModel", "SeasonalPersistenceModel", "QuantileRegression"]
