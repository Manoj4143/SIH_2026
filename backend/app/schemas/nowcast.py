from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class NowcastRequest(BaseModel):
    """
    Request payload for on-demand AIML thunderstorm and lightning nowcast inference.
    """
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Target center latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Target center longitude")
    radius_km: float = Field(default=50.0, ge=5.0, le=250.0, description="Surrounding domain radius")
    lead_times_min: List[int] = Field(default=[15, 30, 60, 120], description="Forecast horizons in minutes")


class GridPointPrediction(BaseModel):
    """
    Prediction item for a single geospatial grid cell and lead time.
    """
    lead_time_min: int
    predicted_max_dbz: float = Field(..., description="Forecasted peak radar reflectivity (dBZ)")
    thunderstorm_probability: float = Field(..., ge=0.0, le=1.0, description="Confidence of severe convection")
    lightning_risk: str = Field(..., description="Categorical risk: Low, Moderate, High, Severe")
    estimated_cape_j_kg: Optional[float] = None
    vertically_integrated_liquid: Optional[float] = None


class CellTrackingSummary(BaseModel):
    """
    Identified convective storm cell vector and projected trajectory.
    """
    cell_id: str
    current_lat: float
    current_lon: float
    centroid_dbz: float
    velocity_km_h: float
    heading_deg: float
    projected_lat_30m: float
    projected_lon_30m: float


class NowcastResponse(BaseModel):
    """
    Comprehensive nowcast output combining grid predictions and cell trajectories.
    """
    generated_at: datetime
    center_latitude: float
    center_longitude: float
    forecast_horizons: List[int]
    status: str = "success"
    model_version: str
    grid_predictions: List[GridPointPrediction]
    convective_cells: List[CellTrackingSummary]
