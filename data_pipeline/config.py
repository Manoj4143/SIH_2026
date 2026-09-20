"""Configuration module for SIH 26072 Weather Nowcasting Data Pipeline.

Defines spatial bounds, channel specifications, normalization ranges,
temporal sequence parameters, and severe weather threshold matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class BoundingBox:
    """Geographic bounding box defined in decimal degrees (EPSG:4326)."""
    min_lon: float = 79.70
    min_lat: float = 12.50
    max_lon: float = 80.85
    max_lat: float = 13.65

    @property
    def center(self) -> Tuple[float, float]:
        """Returns (center_lat, center_lon)."""
        return ((self.min_lat + self.max_lat) / 2.0, (self.min_lon + self.max_lon) / 2.0)

    @property
    def bounds_4326(self) -> Tuple[float, float, float, float]:
        """Returns (min_lon, min_lat, max_lon, max_lat)."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


@dataclass(frozen=True)
class ChannelBoundary:
    """Clamping and normalization bounds for a single sensor channel."""
    name: str
    unit: str
    min_val: float
    max_val: float
    description: str


# 8-Channel Standard Normalization Matrix
CHANNEL_SPECS: List[ChannelBoundary] = [
    ChannelBoundary("dBZ", "dBZ", -10.0, 70.0, "Radar Reflectivity Factor"),
    ChannelBoundary("Radial_Vel", "m/s", -50.0, 50.0, "Doppler Radial Velocity"),
    ChannelBoundary("IR_10.8", "K", 180.0, 320.0, "INSAT-3D/3DR IR Brightness Temperature (10.8 µm)"),
    ChannelBoundary("WV_6.8", "K", 180.0, 300.0, "INSAT-3D/3DR Water Vapor Brightness Temperature (6.8 µm)"),
    ChannelBoundary("Lightning_KDE", "density", 0.0, 1.0, "Log-Transformed Lightning Flash Density"),
    ChannelBoundary("CAPE", "J/kg", 0.0, 5000.0, "Convective Available Potential Energy"),
    ChannelBoundary("CIN", "J/kg", 0.0, 500.0, "Convective Inhibition"),
    ChannelBoundary("Wind_Mag", "m/s", 0.0, 50.0, "850 hPa Wind Magnitude"),
]

CHANNEL_NAMES: List[str] = [c.name for c in CHANNEL_SPECS]
NUM_CHANNELS: int = len(CHANNEL_NAMES)  # 8 channels

TARGET_CHANNEL_NAMES: List[str] = ["dBZ", "Lightning_Density"]
NUM_TARGET_CHANNELS: int = len(TARGET_CHANNEL_NAMES)  # 2 channels


@dataclass
class PipelineConfig:
    """Master configuration for the Data Engine & Preprocessing Microservice."""

    # Spatial Canvas
    grid_height: int = 128
    grid_width: int = 128
    resolution_km: float = 1.0
    src_crs: str = "EPSG:4326"
    target_crs: str = "EPSG:3857"
    bbox: BoundingBox = field(default_factory=BoundingBox)

    # Temporal Sequence Parameters
    t_in: int = 4  # Historical frames: t-45m, t-30m, t-15m, t+0m
    t_out: int = 6  # Forecast frames: t+15m to t+180m (15m/30m steps)
    time_interval_minutes: int = 15

    # Batch Specifications (Calibrated for RTX 3050 4GB VRAM limit)
    batch_size: int = 2
    num_channels: int = NUM_CHANNELS
    num_target_channels: int = NUM_TARGET_CHANNELS

    # Radar Preprocessing Parameters
    radar_clutter_threshold_dbz: float = 0.0
    radar_max_range_km: float = 128.0
    radar_elevation_deg: float = 0.5

    # Lightning Gaussian KDE Parameters
    kde_bandwidth_km: float = 3.0
    kde_max_flashes: int = 5000

    # Severe Convection Hazard Thresholds
    threshold_severe_dbz: float = 35.0
    threshold_extreme_dbz: float = 45.0
    threshold_lightning_high_prob: float = 0.80

    def get_channel_bounds_dict(self) -> Dict[str, Tuple[float, float]]:
        """Returns mapping from channel name to (min_val, max_val)."""
        return {c.name: (c.min_val, c.max_val) for c in CHANNEL_SPECS}

    @property
    def input_tensor_shape(self) -> Tuple[int, int, int, int, int]:
        """Expected 5D input tensor shape: [B, T_in, C, H, W] = [2, 4, 8, 128, 128]."""
        return (self.batch_size, self.t_in, self.num_channels, self.grid_height, self.grid_width)

    @property
    def target_tensor_shape(self) -> Tuple[int, int, int, int, int]:
        """Expected 5D target tensor shape: [B, T_out, C_target, H, W] = [2, 6, 2, 128, 128]."""
        return (self.batch_size, self.t_out, self.num_target_channels, self.grid_height, self.grid_width)


DEFAULT_CONFIG = PipelineConfig()
