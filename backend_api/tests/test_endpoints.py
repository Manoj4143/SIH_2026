"""FastAPI TestClient Endpoint Verification Suite."""

from __future__ import annotations

from typing import Generator
import pytest
from fastapi.testclient import TestClient

from backend_api.app.main import app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Provides a synchronous FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client


class TestAPIEndpoints:
    """Test suite verifying all REST API endpoints, response schemas, and error handling."""

    def test_root_endpoint(self, client: TestClient) -> None:
        """Verifies root landing endpoint metadata and docs links."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "operational"
        assert "endpoints" in data
        assert "documentation" in data

    def test_system_health(self, client: TestClient) -> None:
        """Verifies GET /api/v1/system/health status."""
        resp = client.get("/api/v1/system/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "domain_bounds" in data
        assert "prediction_cache" in data
        assert data["prediction_cache"]["horizons_count"] == 6

    def test_nowcast_latest_default_point(self, client: TestClient) -> None:
        """Verifies GET /api/v1/nowcast/latest returns 6 discrete horizons."""
        resp = client.get("/api/v1/nowcast/latest?lat=13.075&lon=80.275")
        assert resp.status_code == 200
        data = resp.json()

        assert data["location"]["inside_domain"] is True
        assert "Chennai" in data["location"]["district"]
        assert len(data["forecast_horizons"]) == 6

        # Check first and last horizon structures
        h0 = data["forecast_horizons"][0]
        assert h0["lead_time_minutes"] == 15
        assert "reflectivity_dbz" in h0
        assert "rain_rate_mmh" in h0
        assert "convective_risk" in h0
        assert "lightning_probability" in h0
        assert 0.0 <= h0["lightning_probability"] <= 1.0

        h5 = data["forecast_horizons"][5]
        assert h5["lead_time_minutes"] == 180

    def test_nowcast_outside_domain(self, client: TestClient) -> None:
        """Verifies point forecast outside domain sets inside_domain=False."""
        resp = client.get("/api/v1/nowcast/latest?lat=28.6139&lon=77.2090")  # New Delhi
        assert resp.status_code == 200
        data = resp.json()
        assert data["location"]["inside_domain"] is False

    def test_nowcast_invalid_parameters(self, client: TestClient) -> None:
        """Verifies parameter validation: latitude > 90 yields HTTP 422."""
        resp = client.get("/api/v1/nowcast/latest?lat=95.0&lon=80.0")
        assert resp.status_code == 422

    def test_radar_tile_streaming(self, client: TestClient) -> None:
        """Verifies GET /api/v1/radar/tile/{z}/{x}/{y}.png returns PNG binary."""
        resp = client.get("/api/v1/radar/tile/10/740/474.png?time_step=0&channel=0")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content.startswith(b"\x89PNG\r\n\x1a\n")

    def test_lightning_tile_streaming(self, client: TestClient) -> None:
        """Verifies lightning channel tile streaming."""
        resp = client.get("/api/v1/radar/tile/10/740/474.png?time_step=2&channel=1")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_tile_invalid_parameters(self, client: TestClient) -> None:
        """Verifies out-of-range horizon index yields HTTP 422."""
        resp = client.get("/api/v1/radar/tile/10/740/474.png?time_step=99&channel=0")
        assert resp.status_code == 422

    def test_active_alerts_geojson(self, client: TestClient) -> None:
        """Verifies GET /api/v1/alerts/active returns valid GeoJSON FeatureCollection."""
        resp = client.get("/api/v1/alerts/active")
        assert resp.status_code == 200
        data = resp.json()

        assert data["type"] == "FeatureCollection"
        assert "features" in data
        assert isinstance(data["features"], list)

        if len(data["features"]) > 0:
            feat = data["features"][0]
            assert feat["type"] == "Feature"
            assert "geometry" in feat
            assert "properties" in feat
            props = feat["properties"]
            assert "severity" in props
            assert props["severity"] in ("ADVISORY", "WARNING", "EMERGENCY")
            assert "headline" in props
            assert "lead_time_minutes" in props

    def test_active_alerts_filtering(self, client: TestClient) -> None:
        """Verifies alert filtering by severity and lead time."""
        resp = client.get("/api/v1/alerts/active?severity=WARNING&lead_time=30")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "FeatureCollection"

    def test_manual_inference_trigger(self, client: TestClient) -> None:
        """Verifies POST /api/v1/inference/trigger triggers forward pass and updates cache."""
        resp = client.post("/api/v1/inference/trigger")
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "completed"
        assert data["inference_id"].startswith("INF-")
        assert data["horizons_count"] == 6
        assert data["execution_time_ms"] > 0
        assert "peak_dbz" in data
        assert "alerts_generated" in data
