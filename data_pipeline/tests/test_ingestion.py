"""Unit tests for multi-source meteorological ingestion modules."""

import json
from datetime import datetime, timezone
import numpy as np
import pytest

from data_pipeline.config import PipelineConfig, BoundingBox
from data_pipeline.ingestion.mosdac_ingest import MOSDACIngestor
from data_pipeline.ingestion.radar_ingest import RadarIngestor
from data_pipeline.ingestion.lightning_ingest import LightningIngestor
from data_pipeline.ingestion.nwp_ingest import NWPIngestor


@pytest.fixture
def test_config():
    return PipelineConfig(
        grid_height=128,
        grid_width=128,
        bbox=BoundingBox(min_lon=79.70, min_lat=12.50, max_lon=80.85, max_lat=13.65),
    )


class TestMOSDACIngestor:
    def test_synthetic_frame_generation(self, test_config):
        ingestor = MOSDACIngestor(test_config)
        frame = ingestor.generate_synthetic_frame(has_convective_storm=True)

        assert frame.ir_10_8.shape == (128, 128)
        assert frame.wv_6_8.shape == (128, 128)
        assert not frame.is_fallback

        # Verify physical boundaries [180.0, 320.0] K and [180.0, 300.0] K
        assert np.all(frame.ir_10_8 >= 180.0)
        assert np.all(frame.ir_10_8 <= 320.0)
        assert np.all(frame.wv_6_8 >= 180.0)
        assert np.all(frame.wv_6_8 <= 300.0)

        # Convective storm cold cloud top should be well below 230 K
        assert np.min(frame.ir_10_8) < 220.0
        assert np.min(frame.wv_6_8) < 235.0

    def test_missing_file_fallback(self, test_config):
        ingestor = MOSDACIngestor(test_config)
        frame = ingestor.ingest_from_file("non_existent_satellite_file.h5")

        assert frame.is_fallback
        assert frame.ir_10_8.shape == (128, 128)
        assert frame.wv_6_8.shape == (128, 128)
        assert np.isclose(np.mean(frame.ir_10_8), 295.0, atol=1.0)


class TestRadarIngestor:
    def test_synthetic_frame_generation(self, test_config):
        ingestor = RadarIngestor(test_config)
        frame = ingestor.generate_synthetic_frame(has_storm=True, peak_dbz=55.0)

        assert frame.reflectivity.shape == (128, 128)
        assert frame.velocity.shape == (128, 128)
        assert not frame.is_fallback

        # Verify physical boundaries
        assert np.all(frame.reflectivity >= -10.0)
        assert np.all(frame.reflectivity <= 70.0)
        assert np.all(frame.velocity >= -50.0)
        assert np.all(frame.velocity <= 50.0)

        # Convective thunderstorm threshold: cell core >= 35 dBZ
        assert np.max(frame.reflectivity) >= 35.0

        # Doppler velocity couplet should have both positive and negative values
        assert np.min(frame.velocity) < -5.0
        assert np.max(frame.velocity) > 5.0

    def test_missing_file_fallback(self, test_config):
        ingestor = RadarIngestor(test_config)
        frame = ingestor.ingest_from_file("non_existent_radar.vol")

        assert frame.is_fallback
        assert frame.reflectivity.shape == (128, 128)
        assert np.all(frame.reflectivity == -10.0)


class TestLightningIngestor:
    def test_synthetic_batch_generation(self, test_config):
        ingestor = LightningIngestor(test_config)
        batch = ingestor.generate_synthetic_batch(num_strikes=100)

        assert batch.total_count > 0
        assert batch.coordinates.shape[1] == 2
        assert not batch.is_fallback

        # Verify all strikes are within bounding box
        coords = batch.coordinates
        bbox = test_config.bbox
        assert np.all(coords[:, 0] >= bbox.min_lat)
        assert np.all(coords[:, 0] <= bbox.max_lat)
        assert np.all(coords[:, 1] >= bbox.min_lon)
        assert np.all(coords[:, 1] <= bbox.max_lon)

    def test_json_payload_ingestion(self, test_config):
        ingestor = LightningIngestor(test_config)
        payload = [
            {"lat": 13.0, "lon": 80.2, "amp": -25.0},
            {"lat": 13.1, "lon": 80.3, "amp": 40.0},
            {"lat": 50.0, "lon": 10.0, "amp": 15.0},  # Out of bounds
        ]
        batch = ingestor.ingest_from_json(payload)
        # Only the 2 in-bounds points should be retained
        assert batch.total_count == 2

    def test_empty_batch(self, test_config):
        ingestor = LightningIngestor(test_config)
        now = datetime.now(timezone.utc)
        batch = ingestor.get_empty_batch(now, now)
        assert batch.total_count == 0
        assert batch.coordinates.shape == (0, 2)


class TestNWPIngestor:
    def test_synthetic_frame_generation(self, test_config):
        ingestor = NWPIngestor(test_config)
        frame = ingestor.generate_synthetic_frame(high_instability=True)

        assert frame.cape.shape == (128, 128)
        assert frame.cin.shape == (128, 128)
        assert frame.wind_mag.shape == (128, 128)
        assert not frame.is_fallback

        # Verify physical boundaries
        assert np.all(frame.cape >= 0.0)
        assert np.all(frame.cape <= 5000.0)
        assert np.all(frame.cin >= 0.0)
        assert np.all(frame.cin <= 500.0)
        assert np.all(frame.wind_mag >= 0.0)
        assert np.all(frame.wind_mag <= 50.0)

        # High instability should have elevated CAPE (> 2000 J/kg)
        assert np.max(frame.cape) >= 2000.0
