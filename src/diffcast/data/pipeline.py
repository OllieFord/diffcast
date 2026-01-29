"""Data processing pipeline for DiffCast."""

import logging
from datetime import datetime
from pathlib import Path

import polars as pl
from tqdm import tqdm

from diffcast.data.schema import ZONES, DataSplit
from diffcast.data.sources.calendar import add_calendar_features

logger = logging.getLogger(__name__)


class DataPipeline:
    """Pipeline for processing raw data into training-ready format."""

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        splits_dir: Path,
    ) -> None:
        """Initialize pipeline.

        Args:
            raw_dir: Directory containing raw parquet files
            processed_dir: Directory for processed data
            splits_dir: Directory for train/val/test splits
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.splits_dir = Path(splits_dir)

    def load_raw_data(self, zone: str) -> pl.DataFrame | None:
        """Load raw data for a zone.

        Args:
            zone: Bidding zone code

        Returns:
            DataFrame with raw data or None if not found
        """
        raw_path = self.raw_dir / f"{zone}_raw.parquet"
        weather_path = self.raw_dir / f"{zone}_weather.parquet"

        if not raw_path.exists():
            logger.warning(f"Raw data not found for {zone}")
            return None

        df = pl.read_parquet(raw_path)

        # Merge weather if available
        if weather_path.exists():
            weather = pl.read_parquet(weather_path)
            df = df.join(
                weather.select(["timestamp", "temp_c"]),
                on="timestamp",
                how="left",
            )

        return df

    def process_zone(self, zone: str) -> pl.DataFrame | None:
        """Process raw data for a single zone.

        Args:
            zone: Bidding zone code

        Returns:
            Processed DataFrame or None if no data
        """
        df = self.load_raw_data(zone)
        if df is None:
            return None

        # Ensure timestamp is datetime
        if df["timestamp"].dtype != pl.Datetime:
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

        # Sort by timestamp
        df = df.sort("timestamp")

        # Remove duplicates (keep first)
        df = df.unique(subset=["timestamp"], keep="first")

        # Create complete hourly range
        min_ts = df["timestamp"].min()
        max_ts = df["timestamp"].max()
        complete_range = pl.DataFrame({
            "timestamp": pl.datetime_range(min_ts, max_ts, "1h", eager=True)
        })

        # Join with complete range to identify gaps
        df = complete_range.join(df, on="timestamp", how="left")

        # Fill zone column
        df = df.with_columns(pl.lit(zone).alias("zone"))

        # Add calendar features
        df = add_calendar_features(df, zone)

        # Forward fill missing values (within reasonable limits)
        fill_cols = ["price_eur_mwh", "load_mw", "wind_mw", "solar_mw", "hydro_mw", "temp_c"]
        for col in fill_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).forward_fill(limit=24))

        # Log statistics
        n_total = len(df)
        n_missing_price = df["price_eur_mwh"].null_count()
        logger.info(f"{zone}: {n_total} hours, {n_missing_price} missing prices")

        return df

    def process_all_zones(self) -> pl.DataFrame:
        """Process all zones and combine into single DataFrame.

        Returns:
            Combined DataFrame with all zones
        """
        all_dfs = []

        for zone in tqdm(ZONES, desc="Processing zones"):
            df = self.process_zone(zone)
            if df is not None:
                all_dfs.append(df)

        if not all_dfs:
            raise ValueError("No data found for any zone")

        combined = pl.concat(all_dfs)

        # Save processed data
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(self.processed_dir / "all_zones.parquet")

        return combined

    def create_splits(
        self,
        df: pl.DataFrame | None = None,
        split_config: DataSplit | None = None,
    ) -> dict[str, pl.DataFrame]:
        """Create train/val/test splits.

        Args:
            df: Combined DataFrame (loads from processed if None)
            split_config: Split configuration (uses default if None)

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        if df is None:
            df = pl.read_parquet(self.processed_dir / "all_zones.parquet")

        split_config = split_config or DataSplit.default()

        # Create splits based on timestamp
        train = df.filter(
            (pl.col("timestamp") >= split_config.train_start)
            & (pl.col("timestamp") <= split_config.train_end)
        )
        val = df.filter(
            (pl.col("timestamp") >= split_config.val_start)
            & (pl.col("timestamp") <= split_config.val_end)
        )
        test = df.filter(
            (pl.col("timestamp") >= split_config.test_start)
            & (pl.col("timestamp") <= split_config.test_end)
        )

        # Save splits
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        train.write_parquet(self.splits_dir / "train.parquet")
        val.write_parquet(self.splits_dir / "val.parquet")
        test.write_parquet(self.splits_dir / "test.parquet")

        logger.info(f"Train: {len(train)} rows")
        logger.info(f"Val: {len(val)} rows")
        logger.info(f"Test: {len(test)} rows")

        return {"train": train, "val": val, "test": test}

    def get_statistics(self, df: pl.DataFrame) -> dict:
        """Compute dataset statistics for normalization.

        Args:
            df: DataFrame to compute statistics from

        Returns:
            Dictionary with mean and std for each numeric column
        """
        numeric_cols = ["price_eur_mwh", "load_mw", "wind_mw", "solar_mw", "hydro_mw", "temp_c"]

        stats = {}
        for col in numeric_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                stats[col] = {"mean": mean, "std": std if std > 0 else 1.0}

        return stats

    def run(self) -> dict[str, pl.DataFrame]:
        """Run full pipeline.

        Returns:
            Dictionary with train/val/test splits
        """
        logger.info("Processing all zones...")
        combined = self.process_all_zones()

        logger.info("Creating splits...")
        splits = self.create_splits(combined)

        logger.info("Computing statistics...")
        stats = self.get_statistics(splits["train"])

        # Save statistics
        import json

        with open(self.processed_dir / "statistics.json", "w") as f:
            json.dump(stats, f, indent=2)

        return splits
