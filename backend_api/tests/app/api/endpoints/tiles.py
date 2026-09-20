"""Dynamic Slippy Map XYZ Tile Streaming Endpoint (/api/v1/radar/tile/{z}/{x}/{y}.png)."""

from __future__ import annotations

from fastapi import APIRouter, Path, Query, Response

from backend_api.app.api.endpoints.nowcast import load_latest_predictions
from backend_api.app.config import settings
from backend_api.app.services.tile_renderer import renderer

router = APIRouter()


@router.get(
    "/tile/{z}/{x}/{y}.png",
    response_class=Response,
    summary="Dynamic XYZ Tile Map Streaming",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Returns a 256x256 RGBA PNG tile for web mapping libraries (Leaflet/OpenLayers).",
        }
    },
)
async def get_radar_tile(
    z: int = Path(..., ge=0, le=20, description="Zoom level", examples=[10]),
    x: int = Path(..., ge=0, description="Tile X coordinate", examples=[740]),
    y: int = Path(..., ge=0, description="Tile Y coordinate", examples=[474]),
    time_step: int = Query(
        default=0,
        ge=0,
        le=5,
        description="Forecast horizon index (0=t+15m, 1=t+30m, 2=t+45m, 3=t+60m, 4=t+120m, 5=t+180m)",
    ),
    channel: int = Query(
        default=0,
        ge=0,
        le=1,
        description="Forecast channel: 0=Radar Reflectivity (dBZ), 1=Lightning Strike Density",
    ),
) -> Response:
    """Generates and streams a 256x256 Web Mercator PNG tile dynamically in < 850 ms."""
    preds, _ = load_latest_predictions()
    raster_slice = preds[time_step, channel]

    png_bytes = renderer.render_tile(
        raster_2d=raster_slice,
        z=z,
        x=x,
        y=y,
        channel=channel,
        domain_bbox=settings.domain_bbox,
        is_normalized=True,
    )

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=300",
            "X-Tile-Zoom": str(z),
            "X-Forecast-Horizon": f"+{settings.LEAD_TIMES_MINUTES[time_step]}m",
            "X-Channel": "dBZ" if channel == 0 else "Lightning",
        },
    )
