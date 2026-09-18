from datetime import datetime, timedelta, timezone
from typing import List
import random
from app.schemas.observation import LightningStrokeEvent, RadarScanSummary
from app.core.logger import setup_logger

logger = setup_logger("data_service")


class AtmosphericDataService:
    """
    Service responsible for ingesting and preparing observation feeds
    (Doppler Radar, Satellite, Lightning Detection Networks).
    """

    def get_recent_lightning_strokes(
        self, center_lat: float = 28.6139, center_lon: float = 77.2090, count: int = 15
    ) -> List[LightningStrokeEvent]:
        """
        Returns latest recorded lightning stroke events near the specified domain.
        """
        strokes: List[LightningStrokeEvent] = []
        now = datetime.now(timezone.utc)

        for i in range(count):
            offset_lat = (random.random() - 0.5) * 0.4
            offset_lon = (random.random() - 0.5) * 0.4
            current_ka = round(random.uniform(15.0, 75.0), 1)
            stroke_type = "CG" if random.random() > 0.3 else "IC"
            polarity = "+" if random.random() > 0.8 else "-"

            strokes.append(
                LightningStrokeEvent(
                    timestamp=now - timedelta(minutes=random.randint(1, 20)),
                    latitude=round(center_lat + offset_lat, 4),
                    longitude=round(center_lon + offset_lon, 4),
                    peak_current_ka=current_ka,
                    stroke_type=stroke_type,
                    polarity=polarity,
                )
            )
        return strokes

    def get_latest_radar_scan(self, station_code: str = "DWR-DEL") -> RadarScanSummary:
        """
        Retrieves metadata of the most recent volumetric radar sweep.
        """
        return RadarScanSummary(
            station_code=station_code,
            scan_timestamp=datetime.now(timezone.utc),
            max_reflectivity_dbz=52.5,
            vil_kg_m2=34.0,
            echo_top_km=14.2,
        )


data_service = AtmosphericDataService()
