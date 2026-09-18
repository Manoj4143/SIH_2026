# Database Layer: AI Weather Nowcast

This module provides SQLite storage via SQLAlchemy ORM for the AI Weather Nowcast platform.

## Tables Overview

1. **`radar_stations`**: Metadata on deployed Doppler Weather Radars (coordinates, operating frequency band, elevation).
2. **`radar_scans`**: Temporal radar sweeps containing summary indices (max reflectivity in dBZ, VIL, echo tops).
3. **`lightning_observations`**: High-frequency lightning sensor strokes (lat/lon, peak current in kA, CG/IC stroke types).
4. **`nowcast_predictions`**: Persisted model outputs across lead times (15, 30, 60, 120 minutes) for verification.
5. **`hazard_alerts`**: Automated severe thunderstorm and lightning warnings.

## Commands

```bash
# Initialize SQLite schema and seed baseline data
python init_db.py
```
The database will be generated at `weather_nowcast.db`.
