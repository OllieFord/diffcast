"""Weather data from Open-Meteo Historical API."""

import logging
from datetime import datetime
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

from diffcast.data.schema import ZONE_CENTROIDS, ZONES

logger = logging.getLogger(__name__)

OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoClient:
    """Client for fetching historical weather data from Open-Meteo."""

    def __init__(self) -> None:
        """Initialize client."""
        self.base_url = OPEN_METEO_HISTORICAL_URL

    def fetch_temperature(
        self,
        zone: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch hourly temperature for a zone's centroid.

        Args:
            zone: Bidding zone code
            start: Start datetime (UTC)
            end: End datetime (UTC)

        Returns:
            DataFrame with columns: timestamp, temp_c
        """
        lat, lon = ZONE_CENTROIDS[zone]

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m",
            "timezone": "UTC",
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])

            df = pl.DataFrame({
                "timestamp": [datetime.fromisoformat(t) for t in times],
                "temp_c": temps,
            })
            return df

        except Exception as e:
            logger.warning(f"Failed to fetch weather for {zone}: {e}")
            return pl.DataFrame(schema={"timestamp": pl.Datetime, "temp_c": pl.Float64})

    def fetch_all_zones(
        self,
        zones: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, pl.DataFrame]:
        """Fetch weather data for all zones.

        Args:
            zones: List of zone codes (defaults to all ZONES)
            start: Start datetime (defaults to 2018-01-01)
            end: End datetime (defaults to 2024-12-31)
            output_dir: If provided, save raw data to this directory

        Returns:
            Dictionary mapping zone -> DataFrame with temperature
        """
        zones = zones or list(ZONES)
        start = start or datetime(2018, 1, 1)
        end = end or datetime(2024, 12, 31, 23)

        results = {}

        for zone in tqdm(zones, desc="Fetching weather"):
            logger.info(f"Fetching weather for {zone}")
            df = self.fetch_temperature(zone, start, end)

            if len(df) > 0:
                df = df.with_columns(pl.lit(zone).alias("zone"))
                results[zone] = df

                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    df.write_parquet(output_dir / f"{zone}_weather.parquet")

        return results
