"""Pydantic schemas for Point-based Nowcast predictions and ETL triggers."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class LocationInfo(BaseModel):
    """Location metadata for point forecast requests."""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    inside_domain: bool = Field(..., description="Whether location is within radar coverage domain")
    district: Optional[str] = Field(None, description="Identified district or locality")


class ForecastHorizonPoint(BaseModel):
    """Prediction at a specific forecast lead-time for a single coordinate."""
    horizon_index: int = Field(..., ge=0, le=5, description="Index from 0 to 5")
    lead_time_minutes: int = Field(..., description="Lead time in minutes (15, 30, 45, 60, 120, 180)")
    valid_time: str = Field(..., description="ISO 8601 forecast validity timestamp")
    reflectivity_dbz: float = Field(..., description="Predicted radar reflectivity in dBZ")
    rain_rate_mmh: float = Field(..., description="Marshall-Palmer instantaneous rain rate in mm/h")
    convective_risk: str = Field(..., description="Risk category: None, Low, Moderate, Severe, Extreme")
    lightning_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted lightning strike density/prob")
    lightning_risk: str = Field(..., description="Risk category: Low, Moderate, High")
    summary: str = Field(..., description="Meteorological advisory summary")


class NowcastResponse(BaseModel):
    """Complete 6-step spatiotemporal forecast sequence for a given coordinate."""
    location: LocationInfo
    issued_at: str = Field(..., description="Forecast issue timestamp in UTC ISO 8601")
    model_version: str = Field(default="NowcastNet-v1.0", description="Model architecture version")
    forecast_horizons: List[ForecastHorizonPoint] = Field(..., description="6 discrete forecast lead horizons")
    max_dbz_in_domain: Optional[float] = Field(None, description="Max reflectivity detected in domain")
    max_lightning_in_domain: Optional[float] = Field(None, description="Max lightning density detected in domain")


class TriggerResponse(BaseModel):
    """Response returned upon manually triggering an inference forward pass."""
    status: str = Field(..., examples=["completed"])
    inference_id: str = Field(..., description="Unique UUID for inference cycle")
    timestamp: str = Field(..., description="Timestamp of execution")
    execution_time_ms: float = Field(..., description="Processing duration in milliseconds")
    horizons_count: int = Field(default=6, description="Number of future forecast horizons produced")
    peak_dbz: float = Field(..., description="Maximum predicted reflectivity across domain")
    alerts_generated: int = Field(..., description="Number of severe hazard polygons created")
