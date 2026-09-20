"""Unit tests for WeatherTensorDataset, DataLoader, and master DataPipeline."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from data_pipeline.config import (
    CHANNEL_NAMES,
    DEFAULT_CONFIG,
    TARGET_CHANNEL_NAMES,
    PipelineConfig,
)
from data_pipeline.loaders.dataset import WeatherTensorDataset, create_dataloader
from data_pipeline.loaders.pipeline import WeatherNowcastDataPipeline


@pytest.fixture
def test_config():
    return PipelineConfig(
        grid_height=128,
        grid_width=128,
        batch_size=2,
        t_in=4,
        t_out=6,
    )


class TestWeatherTensorDataset:
    def test_synthetic_dataset_shapes_and_bounds(self, test_config):
        dataset = WeatherTensorDataset.generate_synthetic(samples=4, config=test_config)

        assert len(dataset) == 4
        x, y = dataset[0]

        # Convert to numpy if torch tensor
        x_arr = x.numpy() if hasattr(x, "numpy") else x
        y_arr = y.numpy() if hasattr(y, "numpy") else y

        # Shape validation: [T_in=4, C=8, H=128, W=128]
        assert x_arr.shape == (4, 8, 128, 128)
        # Shape validation: [T_out=6, C_target=2, H=128, W=128]
        assert y_arr.shape == (6, 2, 128, 128)

        # Values must be strictly bounded in [0.0, 1.0]
        assert np.min(x_arr) >= 0.0
        assert np.max(x_arr) <= 1.0
        assert np.min(y_arr) >= 0.0
        assert np.max(y_arr) <= 1.0

    def test_dataloader_batch_dimensions(self, test_config):
        dataset = WeatherTensorDataset.generate_synthetic(samples=4, config=test_config)
        loader = create_dataloader(dataset, batch_size=2, shuffle=False)

        batches = list(loader)
        assert len(batches) == 2

        batch_x, batch_y = batches[0]
        bx_arr = batch_x.numpy() if hasattr(batch_x, "numpy") else batch_x
        by_arr = batch_y.numpy() if hasattr(batch_y, "numpy") else batch_y

        # Batch 5D Tensor Shape: [B=2, T_in=4, C=8, H=128, W=128]
        assert bx_arr.shape == (2, 4, 8, 128, 128)
        # Target 5D Tensor Shape: [B=2, T_out=6, C_target=2, H=128, W=128]
        assert by_arr.shape == (2, 6, 2, 128, 128)

    def test_h5_serialization_roundtrip(self, test_config):
        dataset = WeatherTensorDataset.generate_synthetic(samples=2, config=test_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_file = Path(tmpdir) / "test_dataset.h5"
            dataset.save_to_h5(h5_file)
            assert h5_file.exists()

            loaded_ds = WeatherTensorDataset(h5_path=h5_file, config=test_config)
            assert len(loaded_ds) == 2
            assert loaded_ds.inputs.shape == dataset.inputs.shape
            assert loaded_ds.targets.shape == dataset.targets.shape
            assert np.allclose(loaded_ds.inputs, dataset.inputs)


class TestWeatherNowcastDataPipeline:
    def test_assemble_current_frame(self, test_config):
        pipeline = WeatherNowcastDataPipeline(test_config)
        frame = pipeline.assemble_current_frame()

        assert frame.tensor_8c.shape == (8, 128, 128)
        assert np.min(frame.tensor_8c) >= 0.0
        assert np.max(frame.tensor_8c) <= 1.0
        assert "radar" in frame.quality_flags
        assert "satellite" in frame.quality_flags
        assert "lightning" in frame.quality_flags
        assert "nwp" in frame.quality_flags

    def test_historical_sequence_assembly(self, test_config):
        pipeline = WeatherNowcastDataPipeline(test_config)
        # Assemble 2 frames
        pipeline.assemble_current_frame()
        pipeline.assemble_current_frame()

        seq = pipeline.get_historical_sequence_tensor()
        # Shape should be [T_in=4, 8, 128, 128]
        assert seq.shape == (4, 8, 128, 128)
        assert np.all(seq >= 0.0)
        assert np.all(seq <= 1.0)
