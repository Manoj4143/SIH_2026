from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Boolean,
    Text,
)
from sqlalchemy.orm import relationship
from database.connection import Base


def utc_now():
    return datetime.now(timezone.utc)


class RadarStation(Base):
    """
    Metadata for Doppler Weather Radar (DWR) stations.
    """
    __tablename__ = "radar_stations"

    id = Column(Integer, primary_key=True, index=True)
    station_code = Column(String(32), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation_m = Column(Float, default=0.0)
    frequency_band = Column(String(16), default="S-band")  # S-band, C-band, X-band
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)

    scans = relationship("RadarScan", back_populates="station")


class RadarScan(Base):
    """
    Metadata and summary indicators for individual volumetric radar sweeps.
    """
    __tablename__ = "radar_scans"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(Integer, ForeignKey("radar_stations.id"), nullable=False)
    scan_timestamp = Column(DateTime, index=True, nullable=False)
    max_reflectivity_dbz = Column(Float, nullable=True)
    vil_kg_m2 = Column(Float, nullable=True)  # Vertically Integrated Liquid
    echo_top_km = Column(Float, nullable=True)
    file_path = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=utc_now)

    station = relationship("RadarStation", back_populates="scans")


class LightningObservation(Base):
    """
    High-frequency observations from Lightning Detection Networks (LDN).
    """
    __tablename__ = "lightning_observations"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    latitude = Column(Float, index=True, nullable=False)
    longitude = Column(Float, index=True, nullable=False)
    peak_current_ka = Column(Float, nullable=False)
    stroke_type = Column(String(8), default="CG")  # CG = Cloud-to-Ground, IC = Intra-Cloud
    polarity = Column(String(4), default="+")      # '+' or '-'
    sensors_reporting = Column(Integer, default=4)
    created_at = Column(DateTime, default=utc_now)


class NowcastPrediction(Base):
    """
    AIML Nowcast predictions generated for a spatial grid coordinate and lead time.
    """
    __tablename__ = "nowcast_predictions"

    id = Column(Integer, primary_key=True, index=True)
    forecast_timestamp = Column(DateTime, index=True, nullable=False)
    lead_time_minutes = Column(Integer, nullable=False)  # 15, 30, 60, 120 min
    latitude = Column(Float, index=True, nullable=False)
    longitude = Column(Float, index=True, nullable=False)
    predicted_max_dbz = Column(Float, nullable=False)
    thunderstorm_probability = Column(Float, nullable=False)  # 0.0 to 1.0
    lightning_risk_level = Column(String(16), nullable=False)  # Low, Moderate, High, Severe
    model_version = Column(String(32), default="xgb_v1.0")
    created_at = Column(DateTime, default=utc_now)


class HazardAlert(Base):
    """
    Nowcast-triggered automated hazard warnings and alerts.
    """
    __tablename__ = "hazard_alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String(32), nullable=False)  # Lightning, Severe Storm, Hail
    severity = Column(String(16), nullable=False)    # Watch, Warning, Advisory
    region_name = Column(String(128), nullable=False)
    center_latitude = Column(Float, nullable=False)
    center_longitude = Column(Float, nullable=False)
    radius_km = Column(Float, default=25.0)
    issued_at = Column(DateTime, default=utc_now)
    expires_at = Column(DateTime, nullable=False)
    description = Column(Text, nullable=False)
