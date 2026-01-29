#!/usr/bin/env python
"""Download raw data from ENTSO-E and Open-Meteo."""

import logging
import os
from datetime import datetime
from pathlib import Path

import typer
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    config_path: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to config file",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for raw data",
    ),
    entsoe_api_key: str = typer.Option(
        None,
        "--api-key",
        "-k",
        envvar="ENTSOE_API_KEY",
        help="ENTSO-E API key",
    ),
    skip_entsoe: bool = typer.Option(
        False,
        "--skip-entsoe",
        help="Skip ENTSO-E data download",
    ),
    skip_weather: bool = typer.Option(
        False,
        "--skip-weather",
        help="Skip weather data download",
    ),
    zones: list[str] = typer.Option(
        None,
        "--zone",
        "-z",
        help="Specific zones to download (can be repeated)",
    ),
) -> None:
    """Download raw data for DiffCast training."""
    # Load config
    config = OmegaConf.load(config_path)

    # Set output directory
    output_dir = output_dir or Path(config.data.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse dates
    start_date = datetime.fromisoformat(config.processing.start_date)
    end_date = datetime.fromisoformat(config.processing.end_date)

    # Get zones
    zones_to_download = zones or config.processing.zones
    logger.info(f"Downloading data for zones: {zones_to_download}")
    logger.info(f"Date range: {start_date} to {end_date}")

    # Download ENTSO-E data
    if not skip_entsoe:
        if not entsoe_api_key:
            logger.error(
                "ENTSO-E API key required. Set ENTSOE_API_KEY env var or use --api-key"
            )
            raise typer.Exit(1)

        logger.info("Downloading ENTSO-E data...")
        from diffcast.data.sources.entsoe import ENTSOEClient

        client = ENTSOEClient(entsoe_api_key)
        client.fetch_all_data(
            zones=zones_to_download,
            start=start_date,
            end=end_date,
            output_dir=output_dir,
        )
        logger.info("ENTSO-E download complete")

    # Download weather data
    if not skip_weather:
        logger.info("Downloading weather data...")
        from diffcast.data.sources.weather import OpenMeteoClient

        weather_client = OpenMeteoClient()
        weather_client.fetch_all_zones(
            zones=zones_to_download,
            start=start_date,
            end=end_date,
            output_dir=output_dir,
        )
        logger.info("Weather download complete")

    logger.info(f"All data saved to {output_dir}")


if __name__ == "__main__":
    app()
