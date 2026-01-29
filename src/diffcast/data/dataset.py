"""PyTorch Dataset for DiffCast."""

import json
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import Dataset

from diffcast.data.schema import ZONES


class DiffCastDataset(Dataset):
    """PyTorch Dataset for CSDI-style forecasting.

    Creates sliding window samples with context (history) and target (forecast).
    Returns normalized tensors ready for model input.
    """

    # Feature columns in order
    FEATURES = [
        "price_eur_mwh",
        "load_mw",
        "wind_mw",
        "solar_mw",
        "hydro_mw",
        "temp_c",
        "hour",
        "dow",
        "month",
        "is_holiday",
    ]

    def __init__(
        self,
        data_path: Path | str,
        context_length: int = 168,  # 7 days
        forecast_length: int = 24,  # 1 day
        stride: int = 24,  # Create sample every 24 hours
        statistics_path: Path | str | None = None,
        zones: list[str] | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            data_path: Path to parquet file with processed data
            context_length: Number of historical hours to include
            forecast_length: Number of hours to forecast
            stride: Step size between consecutive windows
            statistics_path: Path to statistics.json for normalization
            zones: List of zones to include (defaults to all)
        """
        self.data_path = Path(data_path)
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.stride = stride
        self.zones = zones or list(ZONES)

        # Load data
        df = pl.read_parquet(self.data_path)
        df = df.filter(pl.col("zone").is_in(self.zones))

        # Load statistics for normalization
        self.statistics = {}
        if statistics_path:
            with open(statistics_path) as f:
                self.statistics = json.load(f)

        # Organize data by zone
        self.zone_data: dict[str, pl.DataFrame] = {}
        for zone in self.zones:
            zone_df = df.filter(pl.col("zone") == zone).sort("timestamp")
            if len(zone_df) > 0:
                self.zone_data[zone] = zone_df

        # Build sample index: (zone, start_idx)
        self.samples: list[tuple[str, int]] = []
        window_size = context_length + forecast_length

        for zone, zone_df in self.zone_data.items():
            n_samples = (len(zone_df) - window_size) // stride + 1
            for i in range(n_samples):
                start_idx = i * stride
                self.samples.append((zone, start_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize(self, values: torch.Tensor, col: str) -> torch.Tensor:
        """Normalize values using precomputed statistics."""
        if col in self.statistics:
            mean = self.statistics[col]["mean"]
            std = self.statistics[col]["std"]
            return (values - mean) / std
        return values

    def _denormalize(self, values: torch.Tensor, col: str) -> torch.Tensor:
        """Denormalize values using precomputed statistics."""
        if col in self.statistics:
            mean = self.statistics[col]["mean"]
            std = self.statistics[col]["std"]
            return values * std + mean
        return values

    def _extract_features(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract feature tensor from DataFrame.

        Args:
            df: DataFrame with feature columns

        Returns:
            Tensor of shape (seq_len, n_features)
        """
        features = []
        for col in self.FEATURES:
            if col in df.columns:
                values = df[col].to_numpy()
                # Handle boolean
                if col == "is_holiday":
                    values = values.astype(float)
                tensor = torch.tensor(values, dtype=torch.float32)
                tensor = self._normalize(tensor, col)
            else:
                # Missing feature - fill with zeros
                tensor = torch.zeros(len(df), dtype=torch.float32)
            features.append(tensor)

        return torch.stack(features, dim=-1)  # (seq_len, n_features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dictionary with:
                - context: (context_length, n_features) historical features
                - target: (forecast_length,) target prices (normalized)
                - mask: (forecast_length,) mask for valid targets
                - zone_idx: int zone index
                - covariates: (forecast_length, n_covariates) future known features
        """
        zone, start_idx = self.samples[idx]
        zone_df = self.zone_data[zone]

        # Extract window
        end_idx = start_idx + self.context_length + self.forecast_length
        window = zone_df.slice(start_idx, end_idx - start_idx)

        # Split into context and target
        context_df = window.head(self.context_length)
        target_df = window.tail(self.forecast_length)

        # Extract features
        context = self._extract_features(context_df)

        # Target is just the price
        target_prices = target_df["price_eur_mwh"].to_numpy()
        target = torch.tensor(target_prices, dtype=torch.float32)
        target = self._normalize(target, "price_eur_mwh")

        # Mask for valid targets (1 = valid, 0 = missing)
        mask = torch.isfinite(target).float()
        target = torch.nan_to_num(target, nan=0.0)

        # Future known covariates (calendar features, temperature forecasts)
        covariate_cols = ["hour", "dow", "month", "is_holiday"]
        covariates = []
        for col in covariate_cols:
            if col in target_df.columns:
                values = target_df[col].to_numpy()
                if col == "is_holiday":
                    values = values.astype(float)
                covariates.append(torch.tensor(values, dtype=torch.float32))

        covariates = torch.stack(covariates, dim=-1) if covariates else torch.zeros(
            self.forecast_length, 0
        )

        # Zone index
        zone_idx = self.zones.index(zone)

        return {
            "context": context,
            "target": target,
            "mask": mask,
            "zone_idx": torch.tensor(zone_idx, dtype=torch.long),
            "covariates": covariates,
        }

    def denormalize_prices(self, prices: torch.Tensor) -> torch.Tensor:
        """Denormalize price predictions.

        Args:
            prices: Normalized prices tensor

        Returns:
            Denormalized prices in EUR/MWh
        """
        return self._denormalize(prices, "price_eur_mwh")


class MultiZoneDataset(Dataset):
    """Dataset that returns all zones for a single timestamp.

    Used when the model processes all zones together (cross-zone attention).
    """

    def __init__(
        self,
        data_path: Path | str,
        context_length: int = 168,
        forecast_length: int = 24,
        stride: int = 24,
        statistics_path: Path | str | None = None,
        zones: list[str] | None = None,
    ) -> None:
        """Initialize dataset."""
        self.data_path = Path(data_path)
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.stride = stride
        self.zones = zones or list(ZONES)
        self.n_zones = len(self.zones)

        # Load statistics
        self.statistics = {}
        if statistics_path:
            with open(statistics_path) as f:
                self.statistics = json.load(f)

        # Load and organize data
        df = pl.read_parquet(self.data_path)

        # Pivot to wide format: one row per timestamp, columns for each zone
        self.zone_dfs: dict[str, pl.DataFrame] = {}
        for zone in self.zones:
            zone_df = df.filter(pl.col("zone") == zone).sort("timestamp")
            self.zone_dfs[zone] = zone_df

        # Find common timestamps across all zones
        timestamps_sets = [
            set(self.zone_dfs[zone]["timestamp"].to_list())
            for zone in self.zones
            if zone in self.zone_dfs
        ]

        if timestamps_sets:
            self.common_timestamps = sorted(set.intersection(*timestamps_sets))
        else:
            self.common_timestamps = []

        # Build sample index
        window_size = context_length + forecast_length
        n_samples = (len(self.common_timestamps) - window_size) // stride + 1
        self.n_samples = max(0, n_samples)

    def __len__(self) -> int:
        return self.n_samples

    def _normalize(self, values: torch.Tensor, col: str) -> torch.Tensor:
        """Normalize values."""
        if col in self.statistics:
            mean = self.statistics[col]["mean"]
            std = self.statistics[col]["std"]
            return (values - mean) / std
        return values

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get sample for all zones at a single timestamp window.

        Returns:
            Dictionary with:
                - context: (n_zones, context_length, n_features)
                - target: (n_zones, forecast_length)
                - mask: (n_zones, forecast_length)
                - covariates: (n_zones, forecast_length, n_covariates)
        """
        start_ts_idx = idx * self.stride
        end_ts_idx = start_ts_idx + self.context_length + self.forecast_length

        timestamps = self.common_timestamps[start_ts_idx:end_ts_idx]
        context_ts = timestamps[: self.context_length]
        target_ts = timestamps[self.context_length :]

        contexts = []
        targets = []
        masks = []
        covariates = []

        feature_cols = [
            "price_eur_mwh",
            "load_mw",
            "wind_mw",
            "solar_mw",
            "hydro_mw",
            "temp_c",
            "hour",
            "dow",
            "month",
            "is_holiday",
        ]

        for zone in self.zones:
            zone_df = self.zone_dfs[zone]

            # Get context data
            context_data = zone_df.filter(pl.col("timestamp").is_in(context_ts))
            target_data = zone_df.filter(pl.col("timestamp").is_in(target_ts))

            # Extract features for context
            zone_context = []
            for col in feature_cols:
                if col in context_data.columns:
                    vals = context_data[col].to_numpy()
                    if col == "is_holiday":
                        vals = vals.astype(float)
                    t = torch.tensor(vals, dtype=torch.float32)
                    t = self._normalize(t, col)
                else:
                    t = torch.zeros(len(context_data), dtype=torch.float32)
                zone_context.append(t)

            zone_context = torch.stack(zone_context, dim=-1)
            contexts.append(zone_context)

            # Extract target prices
            target_prices = target_data["price_eur_mwh"].to_numpy()
            target = torch.tensor(target_prices, dtype=torch.float32)
            target = self._normalize(target, "price_eur_mwh")
            mask = torch.isfinite(target).float()
            target = torch.nan_to_num(target, nan=0.0)
            targets.append(target)
            masks.append(mask)

            # Extract covariates
            cov_cols = ["hour", "dow", "month", "is_holiday"]
            zone_cov = []
            for col in cov_cols:
                if col in target_data.columns:
                    vals = target_data[col].to_numpy()
                    if col == "is_holiday":
                        vals = vals.astype(float)
                    zone_cov.append(torch.tensor(vals, dtype=torch.float32))
            zone_cov = torch.stack(zone_cov, dim=-1) if zone_cov else torch.zeros(
                len(target_data), 0
            )
            covariates.append(zone_cov)

        return {
            "context": torch.stack(contexts),  # (n_zones, context_len, n_features)
            "target": torch.stack(targets),  # (n_zones, forecast_len)
            "mask": torch.stack(masks),  # (n_zones, forecast_len)
            "covariates": torch.stack(covariates),  # (n_zones, forecast_len, n_cov)
        }

    def denormalize_prices(self, prices: torch.Tensor) -> torch.Tensor:
        """Denormalize price predictions."""
        if "price_eur_mwh" in self.statistics:
            mean = self.statistics["price_eur_mwh"]["mean"]
            std = self.statistics["price_eur_mwh"]["std"]
            return prices * std + mean
        return prices
