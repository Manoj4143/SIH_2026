"""Master End-to-End Data Pipeline Microservice.

Coordinates multi-source data ingestion, Py-ART clutter mitigation,
Web Mercator reprojection, Gaussian KDE density mapping, MinMax scaling,
and emits standardized 8-channel PyTorch input tensors [4, 8, 128, 128].
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

from data_pipeline.config import (
    CHANNEL_NAMES,
    DEFAULT_CONFIG,
    PipelineConfig,
)
from data_pipeline.ingestion.lightning_ingest import LightningBatch, LightningIngestor
from data_pipeline.ingestion.mosdac_ingest import MOSDACIngestor, SatelliteFrame
from data_pipeline.ingestion.nwp_ingest import NWPFrame, NWPIngestor
from data_pipeline.ingestion.radar_ingest import RadarFrame, RadarIngestor
from data_pipeline.normalization.kde_transform import LightningKDE
from data_pipeline.normalization.scalers import ChannelScaler
from data_pipeline.preprocessing.radar_processor import RadarProcessor
from data_pipeline.preprocessing.spatial_reprojector import SpatialReprojector

logger = logging.getLogger(__name__)


@dataclass
class ProcessedFrame:
    """Standardized single-time-slice 8-channel normalized frame."""
    tensor_8c: np.ndarray  # Shape: [8, 128, 128], values in [0.0, 1.0]
    timestamp: datetime
    quality_flags: Dict[str, str]  # Data source status ("nominal", "fallback", etc.)
    metadata: Dict[str, Any]


class WeatherNowcastDataPipeline:
    """Master pipeline orchestrating end-to-end data ingestion and tensor assembly."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.config = config

        # Ingestion sub-modules
        self.mosdac_ingestor = MOSDACIngestor(config)
        self.radar_ingestor = RadarIngestor(config)
        self.lightning_ingestor = LightningIngestor(config)
        self.nwp_ingestor = NWPIngestor(config)

        # Preprocessing & alignment sub-modules
        self.radar_processor = RadarProcessor(config)
        self.reprojector = SpatialReprojector(config)

        # Normalization sub-modules
        self.lightning_kde = LightningKDE(config, reprojector=self.reprojector)
        self.scaler = ChannelScaler()

        # Sliding historical sequence buffer for real-time inference (capacity: T_in = 4)
        self._history_buffer: Deque[ProcessedFrame] = deque(maxlen=config.t_in)

    def assemble_current_frame(
        self,
        radar_frame: Optional[RadarFrame] = None,
        sat_frame: Optional[SatelliteFrame] = None,
        lightning_batch: Optional[LightningBatch] = None,
        nwp_frame: Optional[NWPFrame] = None,
        timestamp: Optional[datetime] = None,
    ) -> ProcessedFrame:
        """Fuses multi-modal observations into an aligned, normalized [8, 128, 128] frame.

        Applies Py-ART clutter mitigation, spatial reprojection, KDE transform,
        and MinMax scaling in the strict canonical channel order:
        [0: dBZ, 1: Radial_Vel, 2: IR_10.8, 3: WV_6.8, 4: Lightning_KDE, 5: CAPE, 6: CIN, 7: Wind_Mag]
        """
        ts = timestamp or datetime.now(timezone.utc)
        quality_flags: Dict[str, str] = {}
        h, w = self.config.grid_height, self.config.grid_width

        # 1. Radar Ingestion & Clutter Processing
        if radar_frame is None:
            radar_frame = self.radar_ingestor.fetch_live_or_fallback(ts)
        quality_flags["radar"] = "fallback" if radar_frame.is_fallback else "nominal"

        # Apply radar clutter filtering and polar-to-cartesian projection
        clean_dbz, clean_vel = self.radar_processor.process_cartesian_grid(
            reflectivity=radar_frame.reflectivity,
            velocity=radar_frame.velocity,
        )

        # 2. Satellite (MOSDAC INSAT-3D) Ingestion & Alignment
        if sat_frame is None:
            sat_frame = self.mosdac_ingestor.fetch_live_or_fallback(ts)
        quality_flags["satellite"] = "fallback" if sat_frame.is_fallback else "nominal"

        reproj_ir = self.reprojector.reproject_raster(sat_frame.ir_10_8, fill_value=295.0)
        reproj_wv = self.reprojector.reproject_raster(sat_frame.wv_6_8, fill_value=265.0)

        # 3. Lightning Flash Ingestion & KDE Transformation
        if lightning_batch is None:
            lightning_batch = self.lightning_ingestor.generate_synthetic_batch(ts)
        quality_flags["lightning"] = "fallback" if lightning_batch.is_fallback else "nominal"

        kde_density = self.lightning_kde.strike_batch_to_density(lightning_batch, normalize_output=True)

        # 4. NWP Atmospheric Reanalysis Ingestion & Upsampling
        if nwp_frame is None:
            nwp_frame = self.nwp_ingestor.generate_synthetic_frame(ts)
        quality_flags["nwp"] = "fallback" if nwp_frame.is_fallback else "nominal"

        reproj_cape = self.reprojector.reproject_raster(nwp_frame.cape, fill_value=1000.0)
        reproj_cin = self.reprojector.reproject_raster(nwp_frame.cin, fill_value=50.0)
        reproj_wind = self.reprojector.reproject_raster(nwp_frame.wind_mag, fill_value=10.0)

        # 5. Assemble Raw Multi-Channel Array [8, 128, 128]
        raw_tensor = np.empty((8, h, w), dtype=np.float32)
        raw_tensor[0] = clean_dbz
        raw_tensor[1] = clean_vel
        raw_tensor[2] = reproj_ir
        raw_tensor[3] = reproj_wv
        raw_tensor[4] = kde_density
        raw_tensor[5] = reproj_cape
        raw_tensor[6] = reproj_cin
        raw_tensor[7] = reproj_wind

        # 6. Apply Standardized MinMax Scaling to [0.0, 1.0]
        normalized_tensor = self.scaler.normalize_tensor(raw_tensor)

        frame = ProcessedFrame(
            tensor_8c=normalized_tensor,
            timestamp=ts,
            quality_flags=quality_flags,
            metadata={
                "channel_order": CHANNEL_NAMES,
                "shape": list(normalized_tensor.shape),
                "grid_height": h,
                "grid_width": w,
            },
        )

        # Add to history buffer
        self._history_buffer.append(frame)
        return frame

    def get_historical_sequence_tensor(self) -> np.ndarray:
        """Returns the current [T_in, 8, 128, 128] sequence tensor from the history buffer.

        If history buffer has fewer than T_in frames, forward-fills or replicates
        earlier frames to ensure fixed shape [4, 8, 128, 128].
        """
        t_in = self.config.t_in
        h, w = self.config.grid_height, self.config.grid_width

        if not self._history_buffer:
            # Generate a fresh frame if empty
            self.assemble_current_frame()

        frames = list(self._history_buffer)
        while len(frames) < t_in:
            # Replicate oldest frame
            frames.insert(0, frames[0])

        seq_tensor = np.stack([f.tensor_8c for f in frames[-t_in:]], axis=0)
        assert seq_tensor.shape == (t_in, 8, h, w), f"Unexpected sequence shape {seq_tensor.shape}"
        return seq_tensor.astype(np.float32)

    def generate_synthetic_nowcast_pair(
        self,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a complete pair of (X_input, Y_target) for inference testing.

        Returns:
            Tuple of:
            - X_input: [T_in=4, C=8, H=128, W=128]
            - Y_target: [T_out=6, C_target=2, H=128, W=128]
        """
        from data_pipeline.loaders.dataset import WeatherTensorDataset
        ds = WeatherTensorDataset.generate_synthetic(samples=1, config=self.config)
        assert ds.inputs is not None and ds.targets is not None
        return ds.inputs[0], ds.targets[0]
