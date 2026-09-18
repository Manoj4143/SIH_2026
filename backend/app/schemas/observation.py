from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class RadarStationInfo(BaseModel):
    id: int
    station_code: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float
    frequency_band: str
    is_active: bool

    class Config:
        from_attributes = True


class RadarScanSummary(BaseModel):
    station_code: str
    scan_timestamp: datetime
    max_reflectivity_dbz: Optional[float] = None
    vil_kg_m2: Optional[float] = None
    echo_top_km: Optional[float] = None


class LightningStrokeEvent(BaseModel):
    timestamp: datetime
    latitude: float
    longitude: float
    peak_current_ka: float
    stroke_type: str  # 'CG' (cloud-to-ground) or 'IC' (intra-cloud)
    polarity: str     # '+' or '-'


class HazardAlertResponse(BaseModel):
    id: int
    alert_type: str
    severity: str
    region_name: str
    center_latitude: float
    center_longitude: float
    radius_km: float
    issued_at: datetime
    expires_at: datetime
    description: str

    class Config:
        from_attributes = True
