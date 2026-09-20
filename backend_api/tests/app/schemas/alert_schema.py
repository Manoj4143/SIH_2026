"""RFC 7946 Standard GeoJSON Pydantic Schemas for Severe Weather Alert Polygons."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class GeoJSONGeometry(BaseModel):
    """GeoJSON Polygon or MultiPolygon Geometry definition."""
    type: Literal["Polygon", "MultiPolygon"] = Field(..., description="Geometry type")
    coordinates: List[Any] = Field(..., description="Array of linear rings / coordinates")


class AlertProperties(BaseModel):
    """Metadata properties associated with an active severe weather alert polygon."""
    alert_id: str = Field(..., description="Unique alert identifier")
    lead_time_minutes: int = Field(..., description="Forecast horizon lead time in minutes")
    valid_time: str = Field(..., description="ISO 8601 validity timestamp")
    severity: Literal["ADVISORY", "WARNING", "EMERGENCY"] = Field(..., description="Severity classification")
    hazard_type: str = Field(default="Convective Thunderstorm", description="Hazard nature")
    max_dbz: float = Field(..., description="Peak reflectivity inside contour in dBZ")
    mean_dbz: float = Field(..., description="Average reflectivity inside contour in dBZ")
    max_lightning_prob: float = Field(default=0.0, description="Peak lightning probability inside contour")
    area_km2: float = Field(..., description="Approximate ground footprint in square kilometers")
    headline: str = Field(..., description="Brief alerting headline for public warnings")
    description: Optional[str] = Field(None, description="Detailed guidance or hazard impact")
    color: str = Field(default="#FFA500", description="Hex color styling for dashboard maps")
    created_at: str = Field(..., description="Creation timestamp")
    expires_at: str = Field(..., description="Expiration timestamp")


class GeoJSONFeature(BaseModel):
    """Standard GeoJSON Feature representation."""
    type: Literal["Feature"] = "Feature"
    id: Optional[Union[str, int]] = Field(None, description="Feature ID")
    geometry: GeoJSONGeometry = Field(..., description="Spatial geometry")
    properties: AlertProperties = Field(..., description="Hazard properties")


class GeoJSONFeatureCollection(BaseModel):
    """Standard GeoJSON FeatureCollection containing all active severe warning polygons."""
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[GeoJSONFeature] = Field(default_factory=list, description="Array of alert features")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Collection metadata (total alerts, query timestamp, domain)",
    )
