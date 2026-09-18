from fastapi import APIRouter, HTTPException
from app.schemas.nowcast import NowcastRequest, NowcastResponse
from app.services.ml_service import ml_service
from app.core.logger import setup_logger

logger = setup_logger("nowcast_api")
router = APIRouter()


@router.post("/predict", response_model=NowcastResponse, summary="Run AIML Nowcast Prediction")
def run_nowcast_prediction(request: NowcastRequest):
    """
    Executes thunderstorm and lightning nowcasting inference for specified target coordinates and horizons.
    """
    try:
        response = ml_service.predict_nowcast(request)
        return response
    except Exception as e:
        logger.error(f"Error during nowcast prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Nowcast prediction failed: {str(e)}")


@router.get("/quick", response_model=NowcastResponse, summary="Quick sample nowcast query")
def get_sample_nowcast(lat: float = 28.6139, lon: float = 77.2090):
    """
    Convenience GET endpoint for rapid testing from UI or browser.
    """
    req = NowcastRequest(latitude=lat, longitude=lon, radius_km=50.0, lead_times_min=[15, 30, 60, 120])
    return ml_service.predict_nowcast(req)
