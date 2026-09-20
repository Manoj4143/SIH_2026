"""Configuration and settings for SIH 26072 FastAPI Backend API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application and environment configuration settings."""

    # API Metadata
    APP_NAME: str = "SIH 26072 Weather Nowcasting Backend API"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",
    ]

    # Database Configuration (PostGIS / SQLite Dual Mode)
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/nowcast_db",
        description="Async database connection string"
    )
    SQLITE_FALLBACK_URL: str = f"sqlite+aiosqlite:///{PROJECT_ROOT}/data/nowcast.db"
    USE_SQLITE_FALLBACK: bool = True

    # Spatial Study Domain (Chennai & Adjoining Regions - EPSG:4326)
    DOMAIN_MIN_LON: float = 79.70
    DOMAIN_MIN_LAT: float = 12.50
    DOMAIN_MAX_LON: float = 80.85
    DOMAIN_MAX_LAT: float = 13.65
    GRID_HEIGHT: int = 128
    GRID_WIDTH: int = 128

    # Forecast Horizons (Lead times in minutes)
    LEAD_TIMES_MINUTES: List[int] = [15, 30, 45, 60, 120, 180]

    # Severe Weather Hazard Thresholds
    DBZ_MODERATE: float = 20.0
    DBZ_SEVERE: float = 35.0
    DBZ_EXTREME: float = 45.0

    LIGHTNING_MODERATE: float = 0.30
    LIGHTNING_HIGH: float = 0.70

    # Storage & Cache Directories
    DATA_DIR: Path = PROJECT_ROOT / "data"
    INFERENCE_DIR: Path = PROJECT_ROOT / "data" / "inferences"
    TILE_CACHE_DIR: Path = PROJECT_ROOT / "data" / "cache" / "tiles"
    LATEST_NOWCAST_FILE: Path = PROJECT_ROOT / "data" / "inferences" / "latest_nowcast.npz"
    LATEST_ALERTS_FILE: Path = PROJECT_ROOT / "data" / "inferences" / "latest_alerts.json"

    # Tile Generation Constraints
    MAX_TILE_CACHE_ITEMS: int = 2048
    TILE_TARGET_LATENCY_MS: float = 850.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def domain_bbox(self) -> Tuple[float, float, float, float]:
        """Returns (min_lon, min_lat, max_lon, max_lat)."""
        return (
            self.DOMAIN_MIN_LON,
            self.DOMAIN_MIN_LAT,
            self.DOMAIN_MAX_LON,
            self.DOMAIN_MAX_LAT,
        )

    def ensure_directories(self) -> None:
        """Ensures all required storage and cache directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        self.TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
