"""Database Query & Spatial Operation Services."""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend_api.app.config import settings
from backend_api.app.db.models import AlertPolygon, ModelInference, Observation

logger = logging.getLogger(__name__)


class DatabaseService:
    """Async database operations for alerts, observations, and inference history."""

    @staticmethod
    async def get_active_alerts(
        session: Optional[AsyncSession],
        severity: Optional[str] = None,
        lead_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves active alert polygons from the database, with JSON file fallback."""
        features: List[Dict[str, Any]] = []

        if session is not None:
            try:
                now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                query = select(AlertPolygon).where(AlertPolygon.expires_at >= now)

                if severity and severity.upper() != "ALL":
                    query = query.where(AlertPolygon.severity == severity.upper())
                if lead_time is not None:
                    query = query.where(AlertPolygon.lead_time_minutes == lead_time)

                query = query.order_by(desc(AlertPolygon.max_dbz))
                result = await session.execute(query)
                rows = result.scalars().all()

                for row in rows:
                    features.append(row.to_geojson_feature())

                if features:
                    return features
            except Exception as e:
                logger.warning(f"Database alert query failed, falling back to disk cache: {e}")

        # Fallback to latest_alerts.json on disk if DB is empty or unavailable
        if settings.LATEST_ALERTS_FILE.exists():
            try:
                with open(settings.LATEST_ALERTS_FILE, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    all_features = cached_data.get("features", [])

                    filtered = []
                    for feat in all_features:
                        props = feat.get("properties", {})
                        if severity and severity.upper() != "ALL" and props.get("severity") != severity.upper():
                            continue
                        if lead_time is not None and props.get("lead_time_minutes") != lead_time:
                            continue
                        filtered.append(feat)
                    return filtered
            except Exception as read_err:
                logger.error(f"Error reading disk alert cache: {read_err}")

        return features

    @staticmethod
    async def save_alert_polygons(
        session: Optional[AsyncSession],
        features: List[Dict[str, Any]],
    ) -> int:
        """Persists new alert features into database."""
        if not session or not features:
            return 0

        saved_count = 0
        try:
            for feat in features:
                props = feat.get("properties", {})
                geom = feat.get("geometry", {})
                alert_id = props.get("alert_id")

                # Parse timestamps
                valid_time = datetime.datetime.fromisoformat(props["valid_time"])
                expires_at = datetime.datetime.fromisoformat(props["expires_at"])

                row = AlertPolygon(
                    alert_id=alert_id,
                    lead_time_minutes=props.get("lead_time_minutes", 15),
                    valid_time=valid_time,
                    severity=props.get("severity", "WARNING"),
                    hazard_type=props.get("hazard_type", "Convective Thunderstorm"),
                    max_dbz=props.get("max_dbz", 0.0),
                    mean_dbz=props.get("mean_dbz", 0.0),
                    max_lightning_prob=props.get("max_lightning_prob", 0.0),
                    area_km2=props.get("area_km2", 0.0),
                    headline=props.get("headline", ""),
                    description=props.get("description", ""),
                    color_hex=props.get("color", "#FFA500"),
                    geometry_wkt=str(geom),
                    geojson_str=json.dumps(geom),
                    created_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
                    expires_at=expires_at,
                )
                session.add(row)
                saved_count += 1
            await session.commit()
        except Exception as e:
            logger.error(f"Failed to persist alerts to DB: {e}")
            await session.rollback()

        return saved_count

    @staticmethod
    async def record_inference_run(
        session: Optional[AsyncSession],
        inference_id: str,
        latency_ms: float,
        raster_path: str,
        peak_dbz: float,
        peak_lightning: float,
        alerts_count: int,
        status: str = "completed",
    ) -> None:
        """Records an inference execution event."""
        if not session:
            return

        try:
            record = ModelInference(
                id=inference_id,
                execution_latency_ms=latency_ms,
                raster_path=raster_path,
                peak_predicted_dbz=peak_dbz,
                peak_predicted_lightning=peak_lightning,
                alerts_generated=alerts_count,
                status=status,
            )
            session.add(record)
            await session.commit()
        except Exception as e:
            logger.error(f"Failed to record inference run: {e}")
            await session.rollback()


db_service = DatabaseService()
