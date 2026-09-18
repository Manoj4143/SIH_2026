from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models.db_models import RadarStation
from app.schemas.observation import RadarStationInfo, RadarScanSummary
from app.services.data_service import data_service

router = APIRouter()


@router.get("/stations", response_model=List[RadarStationInfo], summary="List Doppler Weather Radar stations")
def list_radar_stations(db: Session = Depends(get_db)):
    """
    Returns all registered Doppler Weather Radar stations.
    """
    stations = db.query(RadarStation).filter(RadarStation.is_active == True).all()
    # If database not initialized yet, provide fallback default list
    if not stations:
        return [
            RadarStationInfo(
                id=1,
                station_code="DWR-DEL",
                name="New Delhi Doppler Radar",
                latitude=28.6139,
                longitude=77.2090,
                elevation_m=216.0,
                frequency_band="S-band",
                is_active=True,
            ),
            RadarStationInfo(
                id=2,
                station_code="DWR-MUM",
                name="Mumbai Doppler Radar",
                latitude=18.9220,
                longitude=72.8347,
                elevation_m=42.0,
                frequency_band="S-band",
                is_active=True,
            ),
        ]
    return stations


@router.get("/scan/latest", response_model=RadarScanSummary, summary="Get latest radar scan metrics")
def get_latest_scan(station_code: str = "DWR-DEL"):
    """
    Retrieves the most recent volumetric radar scan summary.
    """
    return data_service.get_latest_radar_scan(station_code=station_code)
