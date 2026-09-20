"""Doppler Weather Radar (DWR) Ingestion Module.

Streams and ingests radar reflectivity (dBZ) and radial velocity data feeds
from IMD / NOAA NEXRAD archives or local radar volumes, featuring bounds
validation, temporal persistence fallback, and synthetic convective generator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from data_pipeline.config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class RadarFrame:
    """Container for raw or pre-gridded radar observation frames."""
    reflectivity: np.ndarray  # Shape: [128, 128], dBZ clipped to [-10.0, 70.0]
    velocity: np.ndarray      # Shape: [128, 128], m/s clipped to [-50.0, 50.0]
    timestamp: datetime
    is_fallback: bool
    radar_metadata: Dict[str, Any]


class RadarIngestor:
    """Ingestor for Doppler Weather Radar observations."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG, cache_dir: Optional[Union[str, Path]] = None) -> None:
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/radar")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_valid_frame: Optional[RadarFrame] = None

    def ingest_from_file(self, file_path: Union[str, Path], timestamp: Optional[datetime] = None) -> RadarFrame:
        """Reads radar volume file (UF, NEXRAD, NetCDF, or HDF5) via Py-ART or native reader."""
        path = Path(file_path)
        ts = timestamp or datetime.now(timezone.utc)

        if not path.exists():
            logger.warning("Radar file not found: %s. Reverting to fallback.", path)
            return self.get_fallback_frame(ts, reason="file_not_found")

        try:
            # Try importing arm_pyart
            try:
                import pyart
                radar = pyart.io.read(str(path))
                # Extract first sweep reflectivity and velocity
                ref_field = next((k for k in ["reflectivity", "DBZ", "CZ", "reflectivity_horizontal"] if k in radar.fields), None)
                vel_field = next((k for k in ["velocity", "VEL", "VR", "radial_velocity"] if k in radar.fields), None)

                if ref_field is None:
                    raise KeyError(f"No reflectivity field found in radar fields: {list(radar.fields.keys())}")

                raw_ref = radar.fields[ref_field]["data"]
                raw_vel = radar.fields[vel_field]["data"] if vel_field else np.zeros_like(raw_ref)

                # Convert polar sweep to cartesian or pass to processor
                ref_grid = self._quick_polar_to_cartesian(raw_ref)
                vel_grid = self._quick_polar_to_cartesian(raw_vel)

            except ImportError:
                # Fallback to direct numpy/h5py reading if NetCDF/HDF5
                import h5py
                with h5py.File(path, "r") as h5:
                    ref_grid = self._resample_to_grid(np.array(h5.get("reflectivity", np.zeros((128, 128))), dtype=np.float32))
                    vel_grid = self._resample_to_grid(np.array(h5.get("velocity", np.zeros((128, 128))), dtype=np.float32))

            frame = self._build_frame(ref_grid, vel_grid, ts, is_fallback=False, metadata={"source": str(path)})
            self._last_valid_frame = frame
            return frame

        except Exception as err:
            logger.error("Failed to parse radar file %s: %s", path, err)
            return self.get_fallback_frame(ts, reason=str(err))

    def fetch_live_or_fallback(self, timestamp: Optional[datetime] = None) -> RadarFrame:
        """Retrieves radar frame from buffer, with temporal persistence fallback."""
        ts = timestamp or datetime.now(timezone.utc)
        if self._last_valid_frame is not None:
            logger.info("Radar stream offline. Using last valid radar frame persistence.")
            return RadarFrame(
                reflectivity=self._last_valid_frame.reflectivity.copy(),
                velocity=self._last_valid_frame.velocity.copy(),
                timestamp=ts,
                is_fallback=True,
                radar_metadata={"fallback_source": "persistence"},
            )
        return self.get_fallback_frame(ts, reason="no_radar_stream")

    def generate_synthetic_frame(
        self,
        timestamp: Optional[datetime] = None,
        has_storm: bool = True,
        center: Optional[Tuple[int, int]] = None,
        peak_dbz: float = 58.0,
    ) -> RadarFrame:
        """Generates realistic synthetic radar reflectivity and Doppler velocity couplet.

        Models a severe thunderstorm cell with high reflectivity core (up to 55-65 dBZ)
        surrounded by stratiform rain, and a rotating Doppler velocity signature (+/- 30 m/s).
        """
        ts = timestamp or datetime.now(timezone.utc)
        h, w = self.config.grid_height, self.config.grid_width

        # Baseline noise floor (-10.0 dBZ)
        ref = np.full((h, w), fill_value=-10.0, dtype=np.float32)
        vel = np.zeros((h, w), dtype=np.float32)

        if has_storm:
            cy, cx = center if center is not None else (h // 2, w // 2)
            y, x = np.mgrid[0:h, 0:w]
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2

            # Core storm cell radius ~ 14 pixels (~14 km)
            core_mask = np.exp(-dist_sq / (2.0 * (12.0 ** 2)))
            ref = -10.0 + core_mask * (peak_dbz - (-10.0))

            # Add minor atmospheric speckle
            noise = np.random.RandomState(1337).normal(0.0, 1.2, size=(h, w))
            ref = np.where(ref > 0.0, ref + noise, ref)

            # Doppler velocity couplet: inflow (negative) on one side, outflow (positive) on the other
            dx = (x - cx) / 12.0
            dy = (y - cy) / 12.0
            # Tangential / rotational shear velocity
            v_rot = (-dy * core_mask) * 28.0
            # Radial divergence
            v_div = (dx * core_mask) * 15.0
            vel = (v_rot + v_div).astype(np.float32)

        frame = self._build_frame(ref, vel, ts, is_fallback=False, metadata={"synthetic": True, "peak_dbz": peak_dbz})
        self._last_valid_frame = frame
        return frame

    def get_fallback_frame(self, timestamp: datetime, reason: str = "unknown") -> RadarFrame:
        """Generates clear-air neutral fallback radar frame."""
        h, w = self.config.grid_height, self.config.grid_width
        ref = np.full((h, w), fill_value=-10.0, dtype=np.float32)
        vel = np.zeros((h, w), dtype=np.float32)
        return self._build_frame(ref, vel, timestamp, is_fallback=True, metadata={"reason": reason, "type": "clear_air"})

    def _build_frame(
        self,
        ref: np.ndarray,
        vel: np.ndarray,
        timestamp: datetime,
        is_fallback: bool,
        metadata: Dict[str, Any],
    ) -> RadarFrame:
        """Clamps radar metrics strictly within physical boundaries."""
        # Reflectivity clamped to [-10.0, 70.0] dBZ
        ref_clamped = np.clip(ref.astype(np.float32), -10.0, 70.0)
        # Radial velocity clamped to [-50.0, 50.0] m/s
        vel_clamped = np.clip(vel.astype(np.float32), -50.0, 50.0)

        return RadarFrame(
            reflectivity=ref_clamped,
            velocity=vel_clamped,
            timestamp=timestamp,
            is_fallback=is_fallback,
            radar_metadata=metadata,
        )

    def _quick_polar_to_cartesian(self, sweep_data: np.ndarray) -> np.ndarray:
        """Maps polar sweep [azimuth, range] to [128, 128] cartesian grid."""
        h, w = self.config.grid_height, self.config.grid_width
        from scipy.ndimage import map_coordinates

        n_rays, n_gates = sweep_data.shape
        # Center of grid
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        y, x = np.mgrid[0:h, 0:w]

        # Convert Cartesian (x, y) to range and azimuth
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx ** 2 + dy ** 2) / (max(h, w) / 2.0) * n_gates
        theta = (np.arctan2(dx, -dy) % (2.0 * np.pi)) / (2.0 * np.pi) * n_rays

        coords = np.array([theta, r])
        cartesian = map_coordinates(sweep_data, coords, order=1, mode="nearest")
        return cartesian.astype(np.float32)

    def _resample_to_grid(self, arr: np.ndarray) -> np.ndarray:
        """Resamples 2D raster to [128, 128]."""
        from scipy.ndimage import zoom
        h, w = self.config.grid_height, self.config.grid_width
        if arr.shape == (h, w):
            return arr.astype(np.float32)
        scale_y = h / arr.shape[0]
        scale_x = w / arr.shape[1]
        return zoom(arr, (scale_y, scale_x), order=1).astype(np.float32)
