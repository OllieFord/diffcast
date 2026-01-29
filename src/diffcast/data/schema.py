"""Data schemas for DiffCast."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# Bidding zones for Nord Pool Nordic core
ZONES = ("NO1", "NO2", "NO3", "NO4", "NO5", "SE1", "SE2", "SE3", "SE4", "FI")
ZoneType = Literal["NO1", "NO2", "NO3", "NO4", "NO5", "SE1", "SE2", "SE3", "SE4", "FI"]

# Zone centroids for weather data (lat, lon)
ZONE_CENTROIDS: dict[str, tuple[float, float]] = {
    "NO1": (59.91, 10.75),  # Oslo area
    "NO2": (58.97, 5.73),   # Stavanger/Bergen area
    "NO3": (63.43, 10.40),  # Trondheim area
    "NO4": (69.65, 18.96),  # Tromsø area
    "NO5": (60.39, 5.32),   # Bergen area
    "SE1": (67.86, 20.23),  # Northern Sweden
    "SE2": (63.83, 20.26),  # Central Sweden
    "SE3": (59.33, 18.07),  # Stockholm area
    "SE4": (55.60, 13.00),  # Malmö area
    "FI": (60.17, 24.94),   # Helsinki area
}

# ENTSO-E area codes for bidding zones
ENTSOE_AREA_CODES: dict[str, str] = {
    "NO1": "10YNO-1--------2",
    "NO2": "10YNO-2--------T",
    "NO3": "10YNO-3--------J",
    "NO4": "10YNO-4--------9",
    "NO5": "10Y1001A1001A48H",
    "SE1": "10Y1001A1001A44P",
    "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L",
    "SE4": "10Y1001A1001A47J",
    "FI": "10YFI-1--------U",
}


class PriceRecord(BaseModel):
    """Single hourly price record with all features."""

    timestamp: datetime = Field(description="Hourly UTC timestamp")
    zone: ZoneType = Field(description="Bidding zone code")
    price_eur_mwh: float = Field(description="Day-ahead price in EUR/MWh")
    load_mw: float | None = Field(default=None, description="Actual/forecast load in MW")
    wind_mw: float | None = Field(default=None, description="Wind generation in MW")
    solar_mw: float | None = Field(default=None, description="Solar generation in MW")
    hydro_mw: float | None = Field(default=None, description="Hydro generation in MW")
    temp_c: float | None = Field(default=None, description="Temperature at zone centroid in °C")
    hour: int = Field(ge=0, le=23, description="Hour of day (0-23)")
    dow: int = Field(ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(ge=1, le=12, description="Month (1-12)")
    is_holiday: bool = Field(default=False, description="Public holiday flag")


@dataclass
class ZoneData:
    """Container for zone-specific time series data."""

    zone: str
    timestamps: list[datetime]
    prices: list[float]
    loads: list[float | None]
    wind: list[float | None]
    solar: list[float | None]
    hydro: list[float | None]
    temperatures: list[float | None]

    def __len__(self) -> int:
        return len(self.timestamps)


@dataclass
class ForecastWindow:
    """A single training/inference window with context and target."""

    # Context (history) - shape: (context_length, n_features)
    context: list[PriceRecord]

    # Target (forecast horizon) - shape: (forecast_length,)
    target_prices: list[float]
    target_timestamps: list[datetime]

    # Zone information
    zone: str


@dataclass
class DataSplit:
    """Train/validation/test split configuration."""

    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime

    @classmethod
    def default(cls) -> "DataSplit":
        """Default split: Train 2018-2022, Val 2023, Test 2024."""
        return cls(
            train_start=datetime(2018, 1, 1),
            train_end=datetime(2022, 12, 31, 23),
            val_start=datetime(2023, 1, 1),
            val_end=datetime(2023, 12, 31, 23),
            test_start=datetime(2024, 1, 1),
            test_end=datetime(2024, 12, 31, 23),
        )
