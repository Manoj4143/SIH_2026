"""
Database initialization script for AI Weather Nowcast.
Creates SQLite tables and populates baseline reference data.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Ensure parent directory is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from database.connection import engine, Base, SessionLocal
from database.models.db_models import (
    RadarStation,
    HazardAlert,
    NowcastPrediction,
)


def init_database():
    print("Creating SQLite database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

    db = SessionLocal()
    try:
        # Check if stations already exist
        if db.query(RadarStation).count() == 0:
            print("Seeding sample radar stations...")
            stations = [
                RadarStation(
                    station_code="DWR-DEL",
                    name="New Delhi Doppler Radar",
                    latitude=28.6139,
                    longitude=77.2090,
                    elevation_m=216.0,
                    frequency_band="S-band",
                ),
                RadarStation(
                    station_code="DWR-MUM",
                    name="Mumbai Doppler Radar",
                    latitude=18.9220,
                    longitude=72.8347,
                    elevation_m=42.0,
                    frequency_band="S-band",
                ),
                RadarStation(
                    station_code="DWR-BLR",
                    name="Bengaluru Doppler Radar",
                    latitude=12.9716,
                    longitude=77.5946,
                    elevation_m=920.0,
                    frequency_band="C-band",
                ),
                RadarStation(
                    station_code="DWR-CCU",
                    name="Kolkata Doppler Radar",
                    latitude=22.5726,
                    longitude=88.3639,
                    elevation_m=12.0,
                    frequency_band="S-band",
                ),
            ]
            db.add_all(stations)
            db.commit()
            print(f"Added {len(stations)} radar stations.")

        # Seed sample active alert if empty
        if db.query(HazardAlert).count() == 0:
            print("Seeding sample active hazard alert...")
            now = datetime.now(timezone.utc)
            alert = HazardAlert(
                alert_type="Severe Thunderstorm & Lightning",
                severity="Warning",
                region_name="NCR Sub-Division",
                center_latitude=28.6139,
                center_longitude=77.2090,
                radius_km=45.0,
                issued_at=now,
                expires_at=now + timedelta(hours=2),
                description="Intense convective cells developing with high lightning activity (>35 kA) and radar reflectivity exceeding 48 dBZ.",
            )
            db.add(alert)
            db.commit()
            print("Added sample hazard alert.")

        print("Database initialization complete.")
    finally:
        db.close()


if __name__ == "__main__":
    init_database()
