"""FastAPI Application Entrypoint for SIH 26072 Weather Nowcasting System."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend_api.app.api.router import api_router
from backend_api.app.config import settings
from backend_api.app.db.session import init_db

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle manager."""
    logger.info("Starting SIH 26072 Nowcasting Backend API...")
    settings.ensure_directories()
    try:
        await init_db()
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")
    logger.info("Backend API initialized and ready to serve requests.")
    yield
    logger.info("Shutting down backend API.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "High-performance REST API backend microservice for Smart India Hackathon 26072. "
        "Provides real-time spatial nowcasts, dynamic XYZ slippy map tiles, and GeoJSON "
        "severe thunderstorm alert contours across Chennai and surrounding regions."
    ),
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 Router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["Root"])
async def root_welcome() -> JSONResponse:
    """Root landing endpoint providing API metadata and documentation links."""
    return JSONResponse(
        content={
            "system": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "operational",
            "documentation": f"{settings.API_V1_PREFIX}/docs",
            "endpoints": {
                "point_forecast": f"{settings.API_V1_PREFIX}/nowcast/latest?lat=13.08&lon=80.27",
                "map_tile": f"{settings.API_V1_PREFIX}/radar/tile/10/740/474.png?time_step=0&channel=0",
                "active_alerts": f"{settings.API_V1_PREFIX}/alerts/active",
                "system_health": f"{settings.API_V1_PREFIX}/system/health",
                "inference_trigger": f"{settings.API_V1_PREFIX}/inference/trigger",
            },
        }
    )
