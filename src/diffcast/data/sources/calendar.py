"""Calendar and holiday features."""

from datetime import datetime

import holidays as holidays_lib
import polars as pl

# Map zones to country codes for holidays
ZONE_COUNTRIES: dict[str, str] = {
    "NO1": "NO",
    "NO2": "NO",
    "NO3": "NO",
    "NO4": "NO",
    "NO5": "NO",
    "SE1": "SE",
    "SE2": "SE",
    "SE3": "SE",
    "SE4": "SE",
    "FI": "FI",
}


def get_holidays(
    zone: str,
    start_year: int = 2018,
    end_year: int = 2024,
) -> set[datetime]:
    """Get set of holiday dates for a zone's country.

    Args:
        zone: Bidding zone code
        start_year: Start year
        end_year: End year

    Returns:
        Set of holiday dates
    """
    country = ZONE_COUNTRIES[zone]
    holiday_dates = set()

    for year in range(start_year, end_year + 1):
        country_holidays = holidays_lib.country_holidays(country, years=year)
        holiday_dates.update(country_holidays.keys())

    return {datetime.combine(d, datetime.min.time()) for d in holiday_dates}


def add_calendar_features(df: pl.DataFrame, zone: str) -> pl.DataFrame:
    """Add calendar features to a DataFrame.

    Args:
        df: DataFrame with 'timestamp' column
        zone: Bidding zone code for holiday lookup

    Returns:
        DataFrame with added calendar columns: hour, dow, month, is_holiday
    """
    # Get holidays for this zone
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")

    # Extract date range from data
    min_year = df["timestamp"].min().year
    max_year = df["timestamp"].max().year
    holiday_dates = get_holidays(zone, min_year, max_year)

    # Add calendar features
    df = df.with_columns([
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("dow"),
        pl.col("timestamp").dt.month().alias("month"),
    ])

    # Add holiday flag
    df = df.with_columns(
        pl.col("timestamp")
        .dt.date()
        .map_elements(
            lambda d: datetime.combine(d, datetime.min.time()) in holiday_dates,
            return_dtype=pl.Boolean,
        )
        .alias("is_holiday")
    )

    return df
