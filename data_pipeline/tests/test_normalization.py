"""Unit tests for Gaussian KDE lightning transforms and 8-channel MinMax scalers."""

import numpy as np
import pytest

from data_pipeline.config import CHANNEL_NAMES, CHANNEL_SPECS, PipelineConfig
from data_pipeline.normalization.kde_transform import LightningKDE
from data_pipeline.normalization.scalers import ChannelScaler


@pytest.fixture
def test_config():
    return PipelineConfig(grid_height=128, grid_width=128)


class TestLightningKDE:
    def test_empty_strikes_density(self, test_config):
        kde = LightningKDE(test_config)
        density = kde.strikes_to_density_grid([])
        assert density.shape == (128, 128)
        assert np.all(density == 0.0)

    def test_clustered_strikes_density(self, test_config):
        kde = LightningKDE(test_config)
        c_lat, c_lon = test_config.bbox.center

        # Cluster of 50 strikes near center
        rng = np.random.RandomState(42)
        strikes = np.column_stack([
            rng.normal(c_lat, 0.03, size=50),
            rng.normal(c_lon, 0.03, size=50),
        ])

        density = kde.strikes_to_density_grid(strikes)
        assert density.shape == (128, 128)
        assert np.all(density >= 0.0)
        assert np.all(density <= 1.0)
        assert np.max(density) == 1.0  # Max should be normalized to 1.0

        # The peak density should be located near grid center (64, 64)
        peak_y, peak_x = np.unravel_index(np.argmax(density), density.shape)
        assert abs(peak_y - 64) < 10
        assert abs(peak_x - 64) < 10

    def test_out_of_bounds_strikes(self, test_config):
        kde = LightningKDE(test_config)
        out_strikes = np.array([[50.0, 10.0], [-20.0, 100.0]], dtype=np.float32)
        density = kde.strikes_to_density_grid(out_strikes)
        assert density.shape == (128, 128)
        assert np.all(density == 0.0)


class TestChannelScaler:
    def test_single_channel_normalization_bounds(self):
        scaler = ChannelScaler()

        for idx, spec in enumerate(CHANNEL_SPECS):
            # Test at minimum boundary
            norm_min = scaler.normalize_channel(np.array([spec.min_val]), idx)
            assert np.isclose(norm_min[0], 0.0, atol=1e-5), f"Failed for channel {spec.name} min"

            # Test at maximum boundary
            norm_max = scaler.normalize_channel(np.array([spec.max_val]), idx)
            assert np.isclose(norm_max[0], 1.0, atol=1e-5), f"Failed for channel {spec.name} max"

            # Test at midpoint
            mid_val = (spec.min_val + spec.max_val) / 2.0
            norm_mid = scaler.normalize_channel(np.array([mid_val]), idx)
            assert np.isclose(norm_mid[0], 0.5, atol=1e-5), f"Failed for channel {spec.name} mid"

    def test_channel_denormalization_roundtrip(self):
        scaler = ChannelScaler()

        for idx, spec in enumerate(CHANNEL_SPECS):
            original = np.linspace(spec.min_val, spec.max_val, 20, dtype=np.float32)
            norm = scaler.normalize_channel(original, spec.name)
            assert np.all(norm >= 0.0) and np.all(norm <= 1.0)

            restored = scaler.denormalize_channel(norm, spec.name)
            assert np.allclose(original, restored, atol=1e-4)

    def test_multi_channel_tensor_normalization(self):
        scaler = ChannelScaler()
        h, w = 128, 128

        # 3D Tensor: [8, 128, 128]
        raw_3d = np.empty((8, h, w), dtype=np.float32)
        for c, spec in enumerate(CHANNEL_SPECS):
            raw_3d[c] = (spec.min_val + spec.max_val) / 2.0

        norm_3d = scaler.normalize_tensor(raw_3d)
        assert norm_3d.shape == (8, h, w)
        assert np.allclose(norm_3d, 0.5, atol=1e-5)

        # 4D Sequence: [4, 8, 128, 128]
        raw_4d = np.repeat(raw_3d[np.newaxis, ...], 4, axis=0)
        norm_4d = scaler.normalize_tensor(raw_4d)
        assert norm_4d.shape == (4, 8, h, w)
        assert np.allclose(norm_4d, 0.5, atol=1e-5)

        # 5D Mini-batch: [2, 4, 8, 128, 128]
        raw_5d = np.repeat(raw_4d[np.newaxis, ...], 2, axis=0)
        norm_5d = scaler.normalize_tensor(raw_5d)
        assert norm_5d.shape == (2, 4, 8, h, w)
        assert np.allclose(norm_5d, 0.5, atol=1e-5)
