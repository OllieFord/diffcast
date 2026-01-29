"""ENTSO-E Transparency Platform data client."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from entsoe import EntsoePandasClient
from tqdm import tqdm

from diffcast.data.schema import ENTSOE_AREA_CODES, ZONES

logger = logging.getLogger(__name__)


class ENTSOEClient:
    """Client for fetching data from ENTSO-E Transparency Platform."""

    def __init__(self, api_key: str) -> None:
        """Initialize client with API key.

        Args:
            api_key: ENTSO-E API key (get from https://transparency.entsoe.eu/)
        """
        self.client = EntsoePandasClient(api_key=api_key)

    def fetch_day_ahead_prices(
        self,
        zone: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch day-ahead prices for a zone.

        Args:
            zone: Bidding zone code (e.g., 'NO1', 'SE3')
            start: Start datetime (UTC)
            end: End datetime (UTC)

        Returns:
            DataFrame with columns: timestamp, price_eur_mwh
        """
        area_code = ENTSOE_AREA_CODES[zone]
        start_pd = start.strftime("%Y%m%d")
        end_pd = end.strftime("%Y%m%d")

        try:
            prices = self.client.query_day_ahead_prices(
                area_code,
                start=start_pd,
                end=end_pd,
            )
            df = pl.DataFrame({
                "timestamp": prices.index.to_pydatetime(),
                "price_eur_mwh": prices.values,
            })
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch prices for {zone}: {e}")
            return pl.DataFrame(schema={"timestamp": pl.Datetime, "price_eur_mwh": pl.Float64})

    def fetch_load(
        self,
        zone: str,
        start: datetime,
        end: datetime,
        forecast: bool = False,
    ) -> pl.DataFrame:
        """Fetch actual or forecast load for a zone.

        Args:
            zone: Bidding zone code
            start: Start datetime (UTC)
            end: End datetime (UTC)
            forecast: If True, fetch day-ahead forecast; else actual load

        Returns:
            DataFrame with columns: timestamp, load_mw
        """
        area_code = ENTSOE_AREA_CODES[zone]
        start_pd = start.strftime("%Y%m%d")
        end_pd = end.strftime("%Y%m%d")

        try:
            if forecast:
                load = self.client.query_load_forecast(
                    area_code,
                    start=start_pd,
                    end=end_pd,
                )
            else:
                load = self.client.query_load(
                    area_code,
                    start=start_pd,
                    end=end_pd,
                )
            df = pl.DataFrame({
                "timestamp": load.index.to_pydatetime(),
                "load_mw": load.values,
            })
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch load for {zone}: {e}")
            return pl.DataFrame(schema={"timestamp": pl.Datetime, "load_mw": pl.Float64})

    def fetch_generation_by_type(
        self,
        zone: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch generation by type (wind, solar, hydro) for a zone.

        Args:
            zone: Bidding zone code
            start: Start datetime (UTC)
            end: End datetime (UTC)

        Returns:
            DataFrame with columns: timestamp, wind_mw, solar_mw, hydro_mw
        """
        area_code = ENTSOE_AREA_CODES[zone]
        start_pd = start.strftime("%Y%m%d")
        end_pd = end.strftime("%Y%m%d")

        try:
            gen = self.client.query_generation(
                area_code,
                start=start_pd,
                end=end_pd,
            )
            # Map ENTSO-E generation types to our categories
            wind_cols = [c for c in gen.columns if "Wind" in str(c)]
            solar_cols = [c for c in gen.columns if "Solar" in str(c)]
            hydro_cols = [c for c in gen.columns if "Hydro" in str(c)]

            wind_mw = gen[wind_cols].sum(axis=1) if wind_cols else 0
            solar_mw = gen[solar_cols].sum(axis=1) if solar_cols else 0
            hydro_mw = gen[hydro_cols].sum(axis=1) if hydro_cols else 0

            df = pl.DataFrame({
                "timestamp": gen.index.to_pydatetime(),
                "wind_mw": wind_mw.values if hasattr(wind_mw, "values") else [0] * len(gen),
                "solar_mw": solar_mw.values if hasattr(solar_mw, "values") else [0] * len(gen),
                "hydro_mw": hydro_mw.values if hasattr(hydro_mw, "values") else [0] * len(gen),
            })
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch generation for {zone}: {e}")
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "wind_mw": pl.Float64,
                    "solar_mw": pl.Float64,
                    "hydro_mw": pl.Float64,
                }
            )

    def fetch_all_data(
        self,
        zones: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, pl.DataFrame]:
        """Fetch all data types for all zones.

        Args:
            zones: List of zone codes (defaults to all ZONES)
            start: Start datetime (defaults to 2018-01-01)
            end: End datetime (defaults to 2024-12-31)
            output_dir: If provided, save raw data to this directory

        Returns:
            Dictionary mapping zone -> DataFrame with all columns
        """
        zones = zones or list(ZONES)
        start = start or datetime(2018, 1, 1)
        end = end or datetime(2024, 12, 31, 23)

        results = {}

        for zone in tqdm(zones, desc="Fetching zones"):
            logger.info(f"Fetching data for {zone}")

            # Fetch in chunks to avoid API limits
            chunk_size = timedelta(days=365)
            current = start
            all_prices = []
            all_loads = []
            all_gen = []

            while current < end:
                chunk_end = min(current + chunk_size, end)

                prices = self.fetch_day_ahead_prices(zone, current, chunk_end)
                loads = self.fetch_load(zone, current, chunk_end)
                gen = self.fetch_generation_by_type(zone, current, chunk_end)

                all_prices.append(prices)
                all_loads.append(loads)
                all_gen.append(gen)

                current = chunk_end

            # Concatenate chunks
            prices_df = pl.concat(all_prices) if all_prices else pl.DataFrame()
            loads_df = pl.concat(all_loads) if all_loads else pl.DataFrame()
            gen_df = pl.concat(all_gen) if all_gen else pl.DataFrame()

            # Merge all data on timestamp
            if len(prices_df) > 0:
                df = prices_df
                if len(loads_df) > 0:
                    df = df.join(loads_df, on="timestamp", how="left")
                if len(gen_df) > 0:
                    df = df.join(gen_df, on="timestamp", how="left")
                df = df.with_columns(pl.lit(zone).alias("zone"))
                results[zone] = df

                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    df.write_parquet(output_dir / f"{zone}_raw.parquet")

        return results
