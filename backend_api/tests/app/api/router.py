"""Master API v1 Router aggregating all endpoint modules."""

from __future__ import annotations

from fastapi import APIRouter

from backend_api.app.api.endpoints import alerts, nowcast, system, tiles

api_router = APIRouter()

# 1. Point-based Nowcasts (/api/v1/nowcast/latest)
api_router.include_router(
    nowcast.router,
    prefix="/nowcast",
    tags=["Point Forecasts"],
)

# 2. Dynamic XYZ Slippy Map Tiles (/api/v1/radar/tile/{z}/{x}/{y}.png)
api_router.include_router(
    tiles.router,
    prefix="/radar",
    tags=["Map Tiles"],
)

# 3. Active Severe Hazard Alerts (/api/v1/alerts/active)
api_router.include_router(
    alerts.router,
    prefix="/alerts",
    tags=["Severe Weather Alerts"],
)

# 4. System Health (/api/v1/system/health)
api_router.include_router(
    system.router,
    prefix="/system",
    tags=["System Operations"],
)

# 5. Direct top-level Inference Trigger (/api/v1/inference/trigger) to match specification
api_router.add_api_route(
    "/inference/trigger",
    system.trigger_inference,
    methods=["POST"],
    response_model=system.TriggerResponse,
    summary="Manual ETL & Model Inference Trigger",
    tags=["System Operations"],
)
