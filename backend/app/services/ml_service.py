import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List
from app.schemas.nowcast import (
    NowcastRequest,
    NowcastResponse,
    GridPointPrediction,
    CellTrackingSummary,
)
from app.core.logger import setup_logger

logger = setup_logger("ml_service")


class MLInferenceService:
    """
    Manages loading of Scikit-learn and XGBoost atmospheric nowcasting models
    and running real-time inferences.
    """

    def __init__(self):
        self.model_version = "xgb_nowcast_v1.0"
        self.thunderstorm_model = None
        self.lightning_model = None
        self._load_models()

    def _load_models(self):
        """
        Loads pre-trained model artifacts from models/ directory if present.
        """
        models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models"
        thunderstorm_path = models_dir / "thunderstorm_xgb_v1.joblib"
        lightning_path = models_dir / "lightning_xgb_v1.joblib"

        if thunderstorm_path.exists():
            try:
                import joblib
                self.thunderstorm_model = joblib.load(thunderstorm_path)
                logger.info(f"Loaded thunderstorm model from {thunderstorm_path}")
            except Exception as e:
                logger.warning(f"Could not load thunderstorm model: {e}")
        else:
            logger.info(f"No trained model artifact at {thunderstorm_path}. Baseline heuristics active.")

    def predict_nowcast(self, request: NowcastRequest) -> NowcastResponse:
        """
        Generates thunderstorm and lightning nowcast for requested horizons (15, 30, 60, 120 mins).
        """
        predictions: List[GridPointPrediction] = []

        # Baseline heuristic calculation (ready to be plugged with live XGBoost model.predict_proba)
        for lead_time in request.lead_times_min:
            # Decay factor over time
            decay = max(0.2, 1.0 - (lead_time / 180.0))
            dbz_val = round(42.5 * decay + 5.0, 1)
            prob_val = round(min(0.95, max(0.1, 0.78 * decay)), 2)

            risk = "Low"
            if prob_val > 0.70:
                risk = "Severe"
            elif prob_val > 0.50:
                risk = "High"
            elif prob_val > 0.30:
                risk = "Moderate"

            predictions.append(
                GridPointPrediction(
                    lead_time_min=lead_time,
                    predicted_max_dbz=dbz_val,
                    thunderstorm_probability=prob_val,
                    lightning_risk=risk,
                    estimated_cape_j_kg=1850.0,
                    vertically_integrated_liquid=28.4,
                )
            )

        # Sample convective storm cell tracks moving north-east
        convective_cells = [
            CellTrackingSummary(
                cell_id="CELL-01",
                current_lat=request.latitude + 0.08,
                current_lon=request.longitude - 0.05,
                centroid_dbz=48.2,
                velocity_km_h=38.5,
                heading_deg=45.0,
                projected_lat_30m=request.latitude + 0.18,
                projected_lon_30m=request.longitude + 0.06,
            ),
            CellTrackingSummary(
                cell_id="CELL-02",
                current_lat=request.latitude - 0.12,
                current_lon=request.longitude + 0.02,
                centroid_dbz=41.0,
                velocity_km_h=28.0,
                heading_deg=35.0,
                projected_lat_30m=request.latitude - 0.04,
                projected_lon_30m=request.longitude + 0.10,
            ),
        ]

        return NowcastResponse(
            generated_at=datetime.now(timezone.utc),
            center_latitude=request.latitude,
            center_longitude=request.longitude,
            forecast_horizons=request.lead_times_min,
            model_version=self.model_version,
            grid_predictions=predictions,
            convective_cells=convective_cells,
        )


ml_service = MLInferenceService()
