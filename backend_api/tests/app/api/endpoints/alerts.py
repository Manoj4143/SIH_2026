"""Active Severe Weather Alert Polygons Endpoint (/api/v1/alerts/active)."""

from __future__ import annotations

import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend_api.app.api.endpoints.nowcast import load_latest_predictions
from backend_api.app.config import settings
from backend_api.app.db.session import get_db
from backend_api.app.schemas.alert_schema import GeoJSONFeatureCollection
from backend_api.app.services.db_service import db_service
from backend_api.app.services.vector_exporter import vector_exporter

router = APIRouter()


@router.get(
    "/active",
    response_model=GeoJSONFeatureCollection,
    summary="Active Severe Weather Alert Polygons (GeoJSON)",
)
async def get_active_alerts(
    severity: Optional[str] = Query(
        default="ALL",
        description="Filter by hazard severity: ALL, ADVISORY, WARNING, EMERGENCY",
        examples=["ALL"],
    ),
    lead_time: Optional[int] = Query(
        default=None,
        description="Filter by forecast lead time in minutes (15, 30, 45, 60, 120, 180)",
        examples=[30],
    ),
    session: AsyncSession = Depends(get_db),
) -> GeoJSONFeatureCollection:
    """Returns active severe weather warning zones formatted as standard RFC 7946 GeoJSON."""
    # 1. Query database / cache
    features = await db_service.get_active_alerts(
        session=session,
        severity=severity,
        lead_time=lead_time,
    )

    # 2. If no alerts in DB/cache, dynamically extract from latest predictions
    if not features:
        preds, issued_at = load_latest_predictions()
        all_extracted = []

        for t_idx, lead_min in enumerate(settings.LEAD_TIMES_MINUTES):
            # Scale normalized predictions to physical units
            dbz_field = preds[t_idx, 0] * 80.0 - 10.0
            lightning_field = preds[t_idx, 1]

            step_features = vector_exporter.extract_alert_polygons_for_step(
                radar_dbz_2d=dbz_field,
                lightning_2d=lightning_field,
                lead_time_minutes=lead_min,
                base_timestamp=issued_at,
            )
            all_extracted.extend(step_features)

        # Filter by query params if requested
        for feat in all_extracted:
            props = feat.get("properties", {})
            if severity and severity.upper() != "ALL" and props.get("severity") != severity.upper():
                continue
            if lead_time is not None and props.get("lead_time_minutes") != lead_time:
                continue
            features.append(feat)

    collection = vector_exporter.build_feature_collection(features)
    return GeoJSONFeatureCollection(**collection)
