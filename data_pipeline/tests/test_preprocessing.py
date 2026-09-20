"""Unit tests for radar preprocessing and spatial reprojection modules."""

import numpy as np
import pytest

from data_pipeline.config import PipelineConfig, BoundingBox
from data_pipeline.preprocessing.radar_processor import RadarProcessor
from data_pipeline.preprocessing.spatial_reprojector import SpatialReprojector


@pytest.fixture
def test_config():
    return PipelineConfig(
        grid_height=128,
        grid_width=128,
        bbox=BoundingBox(min_lon=79.70, min_lat=12.50, max_lon=80.85, max_lat=13.65),
    )


class TestRadarProcessor:
    def test_clutter_mitigation_and_clamping(self, test_config):
        processor = RadarProcessor(test_config)
        # Synthetic polar sweep [360 rays, 120 gates]
        ref_polar = np.full((360, 120), fill_value=-5.0, dtype=np.float32)
        # Add high stationary ground clutter near origin (gates 0-10)
        ref_polar[:, :10] = 45.0
        vel_polar = np.zeros((360, 120), dtype=np.float32)

        clean_ref = processor.mitigate_ground_clutter(ref_polar, vel_polar, clutter_threshold_dbz=0.0)

        # Non-meteorological noise (-5 dBZ) should be clamped to floor (-10 dBZ)
        assert np.all(clean_ref[:, 20:] == -10.0)
        # Stationary clutter should be damped
        assert np.all(clean_ref[:, :10] < 45.0)

    def test_despeckling(self, test_config):
        processor = RadarProcessor(test_config)
        grid = np.full((128, 128), fill_value=-10.0, dtype=np.float32)
        # Insert isolated single-pixel spike
        grid[64, 64] = 60.0

        cleaned = processor.despeckle(grid, kernel_size=3, speckle_diff=20.0)
        # The isolated spike should be smoothed to match its surrounding median
        assert cleaned[64, 64] < 30.0

    def test_polar_to_cartesian_projection(self, test_config):
        processor = RadarProcessor(test_config)
        polar_sweep = np.full((360, 200), fill_value=20.0, dtype=np.float32)

        cartesian = processor.polar_to_cartesian(polar_sweep)
        assert cartesian.shape == (128, 128)
        assert not np.isnan(cartesian).any()
        assert not np.isinf(cartesian).any()
        # Origin and inside radar envelope should have data
        assert cartesian[64, 64] >= 15.0


class TestSpatialReprojector:
    def test_mercator_roundtrip(self, test_config):
        reprojector = SpatialReprojector(test_config)
        test_lon, test_lat = 80.2707, 13.0827  # Chennai

        x_m, y_m = reprojector.to_mercator(test_lon, test_lat)
        lon_back, lat_back = reprojector.from_mercator(x_m, y_m)

        assert np.isclose(test_lon, lon_back, atol=1e-4)
        assert np.isclose(test_lat, lat_back, atol=1e-4)

    def test_center_point_to_pixel(self, test_config):
        reprojector = SpatialReprojector(test_config)
        c_lat, c_lon = test_config.bbox.center

        px, py = reprojector.point_to_pixel(c_lon, c_lat)
        # Center of bounding box should map near pixel (63.5, 63.5)
        assert np.isclose(px, 63.5, atol=2.0)
        assert np.isclose(py, 63.5, atol=2.0)

    def test_reproject_raster_shape(self, test_config):
        reprojector = SpatialReprojector(test_config)
        # Coarse raster e.g. 32x32
        coarse = np.ones((32, 32), dtype=np.float32) * 42.0

        aligned = reprojector.reproject_raster(coarse)
        assert aligned.shape == (128, 128)
        assert np.isclose(np.mean(aligned), 42.0, atol=1.0)
