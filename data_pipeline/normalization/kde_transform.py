"""Gaussian Kernel Density Estimation (KDE) and Log Transformation for Lightning.

Converts discrete lightning flash point coordinates (latitude, longitude) into
a smooth continuous 128x128 spatial density field using Gaussian convolution
and logarithmic compression into [0.0, 1.0].
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter

from data_pipeline.config import DEFAULT_CONFIG, PipelineConfig
from data_pipeline.preprocessing.spatial_reprojector import SpatialReprojector

logger = logging.getLogger(__name__)


class LightningKDE:
    """Transforms point lightning observations into continuous spatial density fields."""

    def __init__(
        self,
        config: PipelineConfig = DEFAULT_CONFIG,
        reprojector: Optional[SpatialReprojector] = None,
    ) -> None:
        self.config = config
        self.h = config.grid_height
        self.w = config.grid_width
        self.bandwidth_pixels = max(1.0, config.kde_bandwidth_km / config.resolution_km)
        self.reprojector = reprojector or SpatialReprojector(config)

    def strikes_to_density_grid(
        self,
        strikes_lat_lon: Union[np.ndarray, List[Tuple[float, float]]],
        weights: Optional[np.ndarray] = None,
        normalize_output: bool = True,
    ) -> np.ndarray:
        """Converts an [N, 2] array of (lat, lon) coordinates into a [128, 128] density field.

        Process:
        1. Reproject (lon, lat) to (pixel_x, pixel_y) in Web Mercator space.
        2. Bin into 2D histogram accumulator.
        3. Apply 2D Gaussian kernel convolution with bandwidth sigma.
        4. Apply non-linear log compression: log(1 + density).
        5. Scale density to [0.0, 1.0].

        Returns:
            np.ndarray of shape [128, 128] with values in [0.0, 1.0].
        """
        arr = np.asarray(strikes_lat_lon, dtype=np.float32)

        # Empty strike set returns all zeros
        if arr.size == 0 or len(arr.shape) < 2 or arr.shape[0] == 0:
            return np.zeros((self.h, self.w), dtype=np.float32)

        lats = arr[:, 0]
        lons = arr[:, 1]

        # Convert to pixel coordinates
        px, py = self.reprojector.points_to_pixel_array(lons, lats)

        # Filter out-of-bounds strikes
        valid = (px >= 0) & (px < self.w) & (py >= 0) & (py < self.h)
        if not np.any(valid):
            return np.zeros((self.h, self.w), dtype=np.float32)

        px_valid = px[valid]
        py_valid = py[valid]
        w_valid = weights[valid] if weights is not None else None

        # 2D histogram accumulator over [128, 128] pixels
        # y is row (0 to h), x is col (0 to w)
        bins_y = np.arange(self.h + 1)
        bins_x = np.arange(self.w + 1)
        hist, _, _ = np.histogram2d(
            py_valid, px_valid,
            bins=[bins_y, bins_x],
            weights=w_valid,
        )

        # 2D Gaussian convolution
        density = gaussian_filter(hist.astype(np.float32), sigma=self.bandwidth_pixels, mode="constant", cval=0.0)

        # Non-linear log transform: log(1 + rho)
        # Prevents super-dense convective clusters from suppressing moderate surrounding lightning
        log_density = np.log1p(density * 10.0)

        if normalize_output:
            max_val = np.max(log_density)
            if max_val > 1e-6:
                log_density = log_density / max_val
            else:
                log_density = np.zeros_like(log_density)

        return np.clip(log_density, 0.0, 1.0).astype(np.float32)

    def strike_batch_to_density(self, lightning_batch: Any, normalize_output: bool = True) -> np.ndarray:
        """Helper to process LightningBatch dataclass."""
        coords = lightning_batch.coordinates
        if coords.shape[0] == 0:
            return np.zeros((self.h, self.w), dtype=np.float32)

        # Use strike amplitudes as optional energy weights if available
        amps = np.array([abs(s.amplitude_ka) for s in lightning_batch.strikes], dtype=np.float32)
        # Soften weights with square root to prevent single extreme strike from dominating
        weights = np.sqrt(np.clip(amps, 1.0, 200.0))

        return self.strikes_to_density_grid(coords, weights=weights, normalize_output=normalize_output)
