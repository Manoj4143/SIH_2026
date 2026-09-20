"""Radar Preprocessing Module with Clutter Mitigation and Polar-to-Cartesian Gridding.

Implements Py-ART integration with robust fallback for ground clutter removal,
despeckling, velocity gate filtering, and polar-to-Cartesian coordinate projection
onto a uniform 128x128 grid (1 km/pixel).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, median_filter

from data_pipeline.config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)


class RadarProcessor:
    """Processes Doppler radar polar sweeps into clean, clutter-free Cartesian grids."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._has_pyart = False
        try:
            import pyart  # noqa: F401
            self._has_pyart = True
            logger.info("Py-ART library successfully detected for radar processing.")
        except ImportError:
            logger.info("Py-ART not installed in runtime. Running in high-performance native SciPy/NumPy mode.")

    @property
    def has_pyart(self) -> bool:
        """Returns True if Py-ART is available."""
        return self._has_pyart

    def process_raw_sweep(
        self,
        reflectivity_polar: np.ndarray,
        velocity_polar: Optional[np.ndarray] = None,
        azimuth_deg: Optional[np.ndarray] = None,
        range_km: Optional[np.ndarray] = None,
        clutter_threshold_dbz: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Converts raw polar sweep [N_azimuth, N_gates] into clean [128, 128] Cartesian grids.

        Applies clutter mitigation, despeckling, and polar-to-Cartesian interpolation.
        Returns:
            Tuple of (reflectivity_cartesian, velocity_cartesian) each of shape [128, 128].
        """
        thresh = clutter_threshold_dbz if clutter_threshold_dbz is not None else self.config.radar_clutter_threshold_dbz

        # 1. Clutter mitigation on polar domain
        clean_ref_polar = self.mitigate_ground_clutter(reflectivity_polar, velocity_polar, thresh)

        # 2. Despeckle polar reflectivity
        despeckled_ref_polar = self.despeckle(clean_ref_polar)

        # 3. Process velocity polar (mask where reflectivity is below noise floor)
        if velocity_polar is not None:
            clean_vel_polar = np.where(despeckled_ref_polar <= thresh, 0.0, velocity_polar)
            clean_vel_polar = np.clip(clean_vel_polar, -50.0, 50.0)
        else:
            clean_vel_polar = np.zeros_like(reflectivity_polar)

        # 4. Project polar sweep to 128x128 Cartesian grid
        ref_cartesian = self.polar_to_cartesian(despeckled_ref_polar, azimuth_deg, range_km)
        vel_cartesian = self.polar_to_cartesian(clean_vel_polar, azimuth_deg, range_km)

        # 5. Post-grid cleanup (final range clamping)
        ref_cartesian = np.clip(ref_cartesian, -10.0, 70.0)
        vel_cartesian = np.clip(vel_cartesian, -50.0, 50.0)

        return ref_cartesian.astype(np.float32), vel_cartesian.astype(np.float32)

    def process_cartesian_grid(
        self,
        reflectivity: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        clutter_threshold_dbz: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocesses already gridded reflectivity and velocity rasters."""
        h, w = self.config.grid_height, self.config.grid_width
        thresh = clutter_threshold_dbz if clutter_threshold_dbz is not None else self.config.radar_clutter_threshold_dbz

        ref = reflectivity.copy()
        vel = velocity.copy() if velocity is not None else np.zeros((h, w), dtype=np.float32)

        # Clutter mitigation and despeckling
        ref_clean = self.mitigate_ground_clutter(ref, vel, thresh)
        ref_clean = self.despeckle(ref_clean)

        # Velocity mask
        vel_clean = np.where(ref_clean <= thresh, 0.0, vel)

        ref_clamped = np.clip(ref_clean, -10.0, 70.0)
        vel_clamped = np.clip(vel_clean, -50.0, 50.0)

        return ref_clamped.astype(np.float32), vel_clamped.astype(np.float32)

    def mitigate_ground_clutter(
        self,
        reflectivity: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        clutter_threshold_dbz: float = 0.0,
    ) -> np.ndarray:
        """Applies ground clutter mitigation.

        Identifies stationary ground clutter (high reflectivity with near-zero Doppler velocity
        and low spatial texture near radar origin) and clamps noise echoes below threshold.
        """
        clean_ref = reflectivity.copy()

        # Rule 1: Clamp sub-meteorological noise below threshold to floor (-10.0 dBZ)
        clean_ref = np.where(clean_ref < clutter_threshold_dbz, -10.0, clean_ref)

        # Rule 2: If velocity is provided, detect stationary clutter (high ref + near zero velocity)
        if velocity is not None and velocity.shape == reflectivity.shape:
            # Persistent stationary ground echoes: |vel| < 0.5 m/s while ref > 25 dBZ
            stationary_mask = (clean_ref > 25.0) & (np.abs(velocity) < 0.5)
            # Damping stationary ground returns
            clean_ref = np.where(stationary_mask, clean_ref - 15.0, clean_ref)
            clean_ref = np.where(clean_ref < clutter_threshold_dbz, -10.0, clean_ref)

        return clean_ref

    def despeckle(self, grid: np.ndarray, kernel_size: int = 3, speckle_diff: float = 22.0) -> np.ndarray:
        """Removes isolated single-pixel radar spikes (noise speckles)."""
        # Apply 3x3 median filter
        median = median_filter(grid, size=kernel_size, mode="nearest")
        # An isolated spike differs dramatically from its surrounding median
        is_speckle = (grid - median) > speckle_diff
        cleaned = np.where(is_speckle, median, grid)
        return cleaned

    def polar_to_cartesian(
        self,
        polar_data: np.ndarray,
        azimuth_deg: Optional[np.ndarray] = None,
        range_km: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Projects a 2D polar sweep [N_azimuth, N_gates] to [128, 128] Cartesian grid.

        Uses bilinear sub-pixel coordinate mapping via scipy.ndimage.map_coordinates.
        """
        h, w = self.config.grid_height, self.config.grid_width
        n_azimuth, n_gates = polar_data.shape

        # Create target Cartesian grid coordinates centered at radar origin
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        y_grid, x_grid = np.mgrid[0:h, 0:w]

        # Normalized coordinates relative to radar center
        dx = (x_grid - cx) / (w / 2.0)
        dy = (y_grid - cy) / (h / 2.0)

        # Radial distance in gate units (0 to n_gates - 1)
        r_norm = np.sqrt(dx ** 2 + dy ** 2)
        gate_coords = r_norm * (n_gates - 1)

        # Azimuth angle in ray units (0 to n_azimuth - 1)
        # Radar standard: 0° is North, 90° is East (clockwise)
        theta_rad = (np.arctan2(dx, -dy) + 2.0 * np.pi) % (2.0 * np.pi)
        az_coords = (theta_rad / (2.0 * np.pi)) * (n_azimuth - 1)

        # Interpolate polar sweep onto Cartesian grid
        coords = np.array([az_coords, gate_coords])
        cartesian = map_coordinates(polar_data, coords, order=1, mode="constant", cval=-10.0)

        # Mask regions beyond radar maximum range
        mask_outside = r_norm > 1.0
        cartesian[mask_outside] = -10.0

        return cartesian.astype(np.float32)

    def process_pyart_radar(self, pyart_radar: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Utilizes Py-ART GateFilter and Grid routines if available."""
        if not self._has_pyart:
            raise RuntimeError("Py-ART is not installed in the current environment.")

        import pyart

        # Build Py-ART GateFilter
        gatefilter = pyart.filters.GateFilter(pyart_radar)
        ref_field = next((k for k in ["reflectivity", "DBZ", "CZ"] if k in pyart_radar.fields), "reflectivity")
        gatefilter.exclude_below(ref_field, self.config.radar_clutter_threshold_dbz)
        gatefilter.exclude_masked(ref_field)

        # Py-ART Grid object creation
        h, w = self.config.grid_height, self.config.grid_width
        grid = pyart.map.grid_from_radars(
            (pyart_radar,),
            grid_shape=(1, h, w),
            grid_limits=((0, 1000), (-self.config.radar_max_range_km * 1000, self.config.radar_max_range_km * 1000),
                         (-self.config.radar_max_range_km * 1000, self.config.radar_max_range_km * 1000)),
            fields=[ref_field],
            gatefilters=(gatefilter,),
        )

        ref_cart = grid.fields[ref_field]["data"][0]
        vel_cart = np.zeros_like(ref_cart)

        return np.clip(ref_cart, -10.0, 70.0).astype(np.float32), np.clip(vel_cart, -50.0, 50.0).astype(np.float32)
