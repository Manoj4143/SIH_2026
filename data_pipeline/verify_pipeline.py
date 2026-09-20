"""Verification and demonstration script for Phase 1 Data Pipeline.

Generates synthetic multi-modal meteorological observations, processes them through
the end-to-end pipeline, verifies spatial alignment, MinMax scaling bounds, and
tensor shapes [2, 4, 8, 128, 128] for inputs and [2, 6, 2, 128, 128] for targets.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timezone
import numpy as np

from data_pipeline.config import (
    CHANNEL_NAMES,
    DEFAULT_CONFIG,
    TARGET_CHANNEL_NAMES,
    PipelineConfig,
)
from data_pipeline.loaders.dataset import WeatherTensorDataset, create_dataloader
from data_pipeline.loaders.pipeline import WeatherNowcastDataPipeline


def run_pipeline_verification() -> bool:
    print("=" * 70)
    print("SIH 26072 Nowcasting System - Phase 1 Pipeline Verification")
    print("=" * 70)

    config = PipelineConfig(
        grid_height=128,
        grid_width=128,
        batch_size=2,
        t_in=4,
        t_out=6,
    )

    print(f"\n[1/4] Initializing Master Data Pipeline...")
    pipeline = WeatherNowcastDataPipeline(config)
    print(f"  -> Bounding Box: {config.bbox.bounds_4326}")
    print(f"  -> Grid Dimensions: {config.grid_height} x {config.grid_width} (1 km/pixel)")
    print(f"  -> Py-ART available: {pipeline.radar_processor.has_pyart}")
    print(f"  -> pyproj available: {pipeline.reprojector._has_pyproj}")

    print(f"\n[2/4] Assembling Multi-Modal Frame [C=8, H=128, W=128]...")
    frame = pipeline.assemble_current_frame()
    print(f"  -> Output tensor shape: {frame.tensor_8c.shape}")
    print(f"  -> Min value: {np.min(frame.tensor_8c):.4f}, Max value: {np.max(frame.tensor_8c):.4f}")
    print(f"  -> Quality Flags: {frame.quality_flags}")

    for idx, name in enumerate(CHANNEL_NAMES):
        chan = frame.tensor_8c[idx]
        print(f"     Ch {idx} [{name:<14}]: min={np.min(chan):.3f}, max={np.max(chan):.3f}, mean={np.mean(chan):.3f}")

    assert frame.tensor_8c.shape == (8, 128, 128), "Frame shape mismatch"
    assert np.min(frame.tensor_8c) >= 0.0, "Values below 0.0"
    assert np.max(frame.tensor_8c) <= 1.0, "Values above 1.0"
    print("  -> Frame validation: PASS [0.0, 1.0]")

    print(f"\n[3/4] Testing Historical Sequence Tensor [T_in=4, C=8, H=128, W=128]...")
    seq = pipeline.get_historical_sequence_tensor()
    print(f"  -> Sequence shape: {seq.shape}")
    assert seq.shape == (4, 8, 128, 128), "Sequence shape mismatch"
    print("  -> Sequence validation: PASS")

    print(f"\n[4/4] Testing PyTorch Dataset & DataLoader [B=2, T_in=4, C=8, H=128, W=128]...")
    dataset = WeatherTensorDataset.generate_synthetic(samples=4, config=config)
    loader = create_dataloader(dataset, batch_size=2, shuffle=False)

    for b_idx, (batch_x, batch_y) in enumerate(loader):
        bx_shape = tuple(batch_x.shape)
        by_shape = tuple(batch_y.shape)
        print(f"  -> Batch {b_idx + 1}:")
        print(f"     Input  X: shape={bx_shape} (Expected: [2, 4, 8, 128, 128])")
        print(f"     Target Y: shape={by_shape} (Expected: [2, 6, 2, 128, 128])")

        assert bx_shape == (2, 4, 8, 128, 128), f"Input batch shape mismatch: {bx_shape}"
        assert by_shape == (2, 6, 2, 128, 128), f"Target batch shape mismatch: {by_shape}"

        # Tensor boundary checks
        bx_min = float(batch_x.min())
        bx_max = float(batch_x.max())
        by_min = float(batch_y.min())
        by_max = float(batch_y.max())
        print(f"     X value bounds: [{bx_min:.4f}, {bx_max:.4f}]")
        print(f"     Y value bounds: [{by_min:.4f}, {by_max:.4f}]")
        assert bx_min >= 0.0 and bx_max <= 1.0, "Input tensor out of bounds [0, 1]"
        assert by_min >= 0.0 and by_max <= 1.0, "Target tensor out of bounds [0, 1]"

    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("Channel Order: " + " -> ".join(CHANNEL_NAMES))
    print("Target Order:  " + " -> ".join(TARGET_CHANNEL_NAMES))
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_pipeline_verification()
    sys.exit(0 if success else 1)
