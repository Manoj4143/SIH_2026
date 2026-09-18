from fastapi import APIRouter
from datetime import datetime, timezone
from app.core.config import settings

router = APIRouter()


@router.get("/health", summary="Service Health and Status")
def get_health():
    """
    Returns system health, version, and running environment.
    """
    return {
        "status": "online",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
