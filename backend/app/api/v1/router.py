from fastapi import APIRouter
from app.api.v1.endpoints import (
    health,
    nowcast,
    radar,
    observations,
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(nowcast.router, prefix="/nowcast", tags=["Nowcast"])
api_router.include_router(radar.router, prefix="/radar", tags=["Radar"])
api_router.include_router(observations.router, prefix="/observations", tags=["Observations"])
