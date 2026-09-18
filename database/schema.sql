-- AI Weather Nowcast - SQLite Database Reference Schema

CREATE TABLE IF NOT EXISTS radar_stations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_code VARCHAR(32) UNIQUE NOT NULL,
    name VARCHAR(128) NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    elevation_m REAL DEFAULT 0.0,
    frequency_band VARCHAR(16) DEFAULT 'S-band',
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS radar_scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id INTEGER NOT NULL,
    scan_timestamp TIMESTAMP NOT NULL,
    max_reflectivity_dbz REAL,
    vil_kg_m2 REAL,
    echo_top_km REAL,
    file_path VARCHAR(256),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(station_id) REFERENCES radar_stations(id)
);

CREATE TABLE IF NOT EXISTS lightning_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    peak_current_ka REAL NOT NULL,
    stroke_type VARCHAR(8) DEFAULT 'CG',
    polarity VARCHAR(4) DEFAULT '+',
    sensors_reporting INTEGER DEFAULT 4,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS nowcast_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_timestamp TIMESTAMP NOT NULL,
    lead_time_minutes INTEGER NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    predicted_max_dbz REAL NOT NULL,
    thunderstorm_probability REAL NOT NULL,
    lightning_risk_level VARCHAR(16) NOT NULL,
    model_version VARCHAR(32) DEFAULT 'xgb_v1.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hazard_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type VARCHAR(32) NOT NULL,
    severity VARCHAR(16) NOT NULL,
    region_name VARCHAR(128) NOT NULL,
    center_latitude REAL NOT NULL,
    center_longitude REAL NOT NULL,
    radius_km REAL DEFAULT 25.0,
    issued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    description TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_radar_scan_timestamp ON radar_scans(scan_timestamp);
CREATE INDEX IF NOT EXISTS idx_lightning_coords ON lightning_observations(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_nowcast_coords ON nowcast_predictions(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_nowcast_lead_time ON nowcast_predictions(lead_time_minutes);
