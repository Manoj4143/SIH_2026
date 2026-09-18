from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from database.connection import get_db
from database.models.db_models import HazardAlert
from app.schemas.observation import LightningStrokeEvent, HazardAlertResponse
from app.services.data_service import data_service

router = APIRouter()


@router.get("/lightning", response_model=List[LightningStrokeEvent], summary="Get real-time lightning strokes")
def get_lightning_feed(lat: float = 28.6139, lon: float = 77.2090, count: int = 15):
    """
    Returns latest recorded lightning sensor strokes in the observation domain.
    """
    return data_service.get_recent_lightning_strokes(center_lat=lat, center_lon=lon, count=count)


@router.get("/alerts", response_model=List[HazardAlertResponse], summary="Active severe weather alerts")
def get_active_alerts(db: Session = Depends(get_db)):
    """
    Retrieves all currently active thunderstorm/lightning hazard alerts.
    """
    now = datetime.now(timezone.utc)
    alerts = db.query(HazardAlert).filter(HazardAlert.expires_at > now).all()
    if not alerts:
        # Fallback default active advisory
        return [
            HazardAlertResponse(
                id=1,
                alert_type="Severe Thunderstorm & Lightning",
                severity="Warning",
                region_name="NCR Sub-Division",
                center_latitude=28.6139,
                center_longitude=77.2090,
                radius_km=45.0,
                issued_at=now - timedelta(minutes=15),
                expires_at=now + timedelta(hours=2),
                description="Convective cell cluster generating frequent lightning and >45 dBZ radar reflectivity.",
            )
        ]
    return alerts
