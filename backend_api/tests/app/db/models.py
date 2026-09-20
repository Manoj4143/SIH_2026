"""SQLAlchemy & PostGIS ORM Models for SIH 26072 Nowcasting Backend."""

from __future__ import annotations

import datetime
from typing import Any, Dict
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base declarative class for all database models."""
    pass


class AlertPolygon(Base):
    """Severe weather warning polygons extracted from model forecasts.

    Stores hazard polygons for thunderstorm cores (>= 35 dBZ) and high lightning density.
    """
    __tablename__ = "alert_polygons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(64), unique=True, index=True, nullable=False)
    lead_time_minutes = Column(Integer, index=True, nullable=False)
    valid_time = Column(DateTime, index=True, nullable=False)
    severity = Column(String(32), index=True, nullable=False)  # ADVISORY, WARNING, EMERGENCY
    hazard_type = Column(String(64), default="Convective Thunderstorm")
    max_dbz = Column(Float, nullable=False)
    mean_dbz = Column(Float, nullable=False)
    max_lightning_prob = Column(Float, default=0.0)
    area_km2 = Column(Float, nullable=False)
    headline = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    color_hex = Column(String(16), default="#FFA500")
    geometry_wkt = Column(Text, nullable=False)
    geojson_str = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)
    expires_at = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("idx_alert_severity_valid", "severity", "valid_time"),
        Index("idx_alert_lead_time", "lead_time_minutes"),
    )

    def to_geojson_feature(self) -> Dict[str, Any]:
        """Converts model row to a standard GeoJSON Feature dictionary."""
        import json
        try:
            geometry = json.loads(str(self.geojson_str))
        except Exception:
            geometry = {"type": "Polygon", "coordinates": []}

        return {
            "type": "Feature",
            "id": self.alert_id,
            "geometry": geometry,
            "properties": {
                "alert_id": self.alert_id,
                "lead_time_minutes": self.lead_time_minutes,
                "valid_time": self.valid_time.isoformat() if self.valid_time else "",
                "severity": self.severity,
                "hazard_type": self.hazard_type,
                "max_dbz": round(self.max_dbz, 1),
                "mean_dbz": round(self.mean_dbz, 1),
                "max_lightning_prob": round(self.max_lightning_prob, 3),
                "area_km2": round(self.area_km2, 2),
                "headline": self.headline,
                "description": self.description,
                "color": self.color_hex,
                "created_at": self.created_at.isoformat() if self.created_at else "",
                "expires_at": self.expires_at.isoformat() if self.expires_at else "",
            },
        }


class Observation(Base):
    """Metadata tracking for ingested multi-modal raw/preprocessed sensor grids."""
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    modality = Column(String(64), index=True, nullable=False)  # radar, mosdac_ir, mosdac_wv, lightning, nwp
    file_path = Column(String(512), nullable=False)
    quality_flag = Column(String(32), default="nominal")  # nominal, fallback, degraded
    grid_shape = Column(String(32), default="128x128")
    min_val = Column(Float, nullable=True)
    max_val = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)


class ModelInference(Base):
    """Tracking table for automated or manually triggered deep learning inference cycles."""
    __tablename__ = "model_inferences"

    id = Column(String(64), primary_key=True)
    timestamp = Column(DateTime, index=True, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)
    execution_latency_ms = Column(Float, nullable=False)
    model_version = Column(String(64), default="NowcastNet-v1.0")
    device_used = Column(String(32), default="cpu")
    raster_path = Column(String(512), nullable=False)
    horizons_count = Column(Integer, default=6)
    peak_predicted_dbz = Column(Float, default=0.0)
    peak_predicted_lightning = Column(Float, default=0.0)
    alerts_generated = Column(Integer, default=0)
    status = Column(String(32), default="completed")  # completed, failed
    error_message = Column(Text, nullable=True)
