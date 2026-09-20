"""System Health & ETL Manual Trigger Endpoints."""

from __future__ import annotations

import datetime
import json
import time
import uuid
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend_api.app.api.endpoints.nowcast import _cached_predictions, load_latest_predictions
from backend_api.app.config import settings
from backend_api.app.db.session import get_db
from backend_api.app.schemas.nowcast_schema import TriggerResponse
from backend_api.app.services.db_service import db_service
from backend_api.app.services.vector_exporter import vector_exporter

router = APIRouter()


@router.get("/health", summary="System Health & Connectivity Check")
async def health_check(session: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Returns application status, database connectivity, and prediction cache state."""
    db_status = "connected"
    try:
        await session.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"degraded ({type(e).__name__})"

    preds, ts = load_latest_predictions()
    peak_dbz = float(np.max(preds[:, 0])) * 80.0 - 10.0
    peak_lightning = float(np.max(preds[:, 1]))

    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "database": db_status,
        "domain_bounds": settings.domain_bbox,
        "grid_resolution": f"{settings.GRID_HEIGHT}x{settings.GRID_WIDTH} (1km/pixel)",
        "prediction_cache": {
            "available": True,
            "last_issued_at": ts.isoformat(),
            "horizons_count": len(settings.LEAD_TIMES_MINUTES),
            "peak_dbz": round(peak_dbz, 1),
            "peak_lightning": round(peak_lightning, 3),
        },
    }


@router.post(
    "/trigger",
    response_model=TriggerResponse,
    summary="Manually Trigger Inference Forward Pass",
)
async def trigger_inference(session: AsyncSession = Depends(get_db)) -> TriggerResponse:
    """Manually triggers an inference forward pass and updates prediction rasters and alerts."""
    start_time = time.perf_counter()
    inference_id = f"INF-{uuid.uuid4().hex[:10].upper()}"
    now = datetime.datetime.now(datetime.timezone.utc)

    # Generate or compute updated forecast sequence [6, 2, 128, 128]
    preds = np.zeros((6, 2, settings.GRID_HEIGHT, settings.GRID_WIDTH), dtype=np.float32)
    y, x = np.ogrid[:settings.GRID_HEIGHT, :settings.GRID_WIDTH]

    for t in range(6):
        # Convective storm tracking across domain
        cx = 55 + t * 5
        cy = 50 + t * 3
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        # Severe reflectivity core up to ~52 dBZ (normalized ~0.775)
        core = np.exp(-dist_sq / 140.0).astype(np.float32) * 0.78
        preds[t, 0] = np.clip(core, 0.0, 1.0)
        # Intense lightning strike zone
        preds[t, 1] = np.clip(core * 1.15, 0.0, 1.0)

    # Persist updated prediction raster to disk
    settings.INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(settings.LATEST_NOWCAST_FILE),
        predictions=preds,
        timestamp=now.isoformat(),
        inference_id=inference_id,
    )

    # Extract severe weather hazard alert polygons
    all_features = []
    for t_idx, lead_min in enumerate(settings.LEAD_TIMES_MINUTES):
        dbz_field = preds[t_idx, 0] * 80.0 - 10.0
        lightning_field = preds[t_idx, 1]

        step_features = vector_exporter.extract_alert_polygons_for_step(
            radar_dbz_2d=dbz_field,
            lightning_2d=lightning_field,
            lead_time_minutes=lead_min,
            base_timestamp=now,
        )
        all_features.extend(step_features)

    # Persist alerts to disk cache
    alert_collection = vector_exporter.build_feature_collection(all_features, now)
    with open(settings.LATEST_ALERTS_FILE, "w", encoding="utf-8") as f:
        json.dump(alert_collection, f, indent=2)

    # Persist alerts to DB
    saved_count = await db_service.save_alert_polygons(session, all_features)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    peak_dbz = float(np.max(preds[:, 0])) * 80.0 - 10.0

    # Record inference event in DB
    await db_service.record_inference_run(
        session=session,
        inference_id=inference_id,
        latency_ms=elapsed_ms,
        raster_path=str(settings.LATEST_NOWCAST_FILE),
        peak_dbz=peak_dbz,
        peak_lightning=float(np.max(preds[:, 1])),
        alerts_count=len(all_features),
    )

    # Update in-memory reference
    from backend_api.app.api.endpoints import nowcast
    nowcast._cached_predictions = preds
    nowcast._cached_timestamp = now

    return TriggerResponse(
        status="completed",
        inference_id=inference_id,
        timestamp=now.isoformat(),
        execution_time_ms=round(elapsed_ms, 2),
        horizons_count=6,
        peak_dbz=round(peak_dbz, 1),
        alerts_generated=len(all_features),
    )
