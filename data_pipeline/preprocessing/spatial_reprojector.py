"""Spatial Reprojection and Alignment Module.

Reprojects spatial data layers from native geographic coordinates (EPSG:4326)
to Web Mercator (EPSG:3857) and ensures pixel-perfect alignment onto a
standardized 128x128 Cartesian grid (1 km/pixel resolution).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from data_pipeline.config import DEFAULT_CONFIG, BoundingBox, PipelineConfig

logger = logging.getLogger(__name__)


class SpatialReprojector:
    """Handles spatial coordinate transformations and grid alignment."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.bbox = config.bbox
        self.h = config.grid_height
        self.w = config.grid_width

        self._has_pyproj = False
        self._transformer = None
        try:
            import pyproj
            self._transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            self._inv_transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
            self._has_pyproj = True
            logger.info("pyproj Transformer initialized for EPSG:4326 -> EPSG:3857.")
        except ImportError:
            logger.info("pyproj not detected. Using mathematical Mercator projection equations.")

        # Compute Web Mercator bounding box in meters
        self.x_min, self.y_min = self.to_mercator(self.bbox.min_lon, self.bbox.min_lat)
        self.x_max, self.y_max = self.to_mercator(self.bbox.max_lon, self.bbox.max_lat)

    def to_mercator(self, lon: float, lat: float) -> Tuple[float, float]:
        """Converts (lon, lat) degrees to Web Mercator (x, y) meters (EPSG:3857)."""
        if self._has_pyproj and self._transformer is not None:
            x, y = self._transformer.transform(lon, lat)
            return float(x), float(y)

        # Standard mathematical Spherical Mercator formulas
        r_major = 6378137.0
        x = r_major * np.radians(lon)
        lat_clamped = np.clip(lat, -85.05112878, 85.05112878)
        y = r_major * np.log(np.tan(np.pi / 4.0 + np.radians(lat_clamped) / 2.0))
        return float(x), float(y)

    def from_mercator(self, x: float, y: float) -> Tuple[float, float]:
        """Converts Web Mercator (x, y) meters to (lon, lat) degrees (EPSG:4326)."""
        if self._has_pyproj and self._inv_transformer is not None:
            lon, lat = self._inv_transformer.transform(x, y)
            return float(lon), float(lat)

        r_major = 6378137.0
        lon = np.degrees(x / r_major)
        lat = np.degrees(2.0 * np.arctan(np.exp(y / r_major)) - np.pi / 2.0)
        return float(lon), float(lat)

    def point_to_pixel(self, lon: float, lat: float) -> Tuple[float, float]:
        """Maps (lon, lat) coordinate to grid pixel coordinates (pixel_x, pixel_y).

        Returns:
            Tuple of (pixel_x, pixel_y) where (0, 0) is top-left of the 128x128 grid.
        """
        x_m, y_m = self.to_mercator(lon, lat)
        # Normalize into [0, 1] relative to domain bounding box
        x_norm = (x_m - self.x_min) / (self.x_max - self.x_min) if self.x_max != self.x_min else 0.5
        # Invert y because image row 0 is top (North, maximum y)
        y_norm = (self.y_max - y_m) / (self.y_max - self.y_min) if self.y_max != self.y_min else 0.5

        px = x_norm * (self.w - 1)
        py = y_norm * (self.h - 1)
        return float(px), float(py)

    def points_to_pixel_array(self, lons: np.ndarray, lats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized conversion of (lons, lats) to pixel coordinates (px, py)."""
        if self._has_pyproj and self._transformer is not None:
            x_m, y_m = self._transformer.transform(lons, lats)
        else:
            r_major = 6378137.0
            x_m = r_major * np.radians(lons)
            lats_c = np.clip(lats, -85.05112878, 85.05112878)
            y_m = r_major * np.log(np.tan(np.pi / 4.0 + np.radians(lats_c) / 2.0))

        x_norm = (x_m - self.x_min) / (self.x_max - self.x_min)
        y_norm = (self.y_max - y_m) / (self.y_max - self.y_min)

        px = x_norm * (self.w - 1)
        py = y_norm * (self.h - 1)
        return px, py

    def reproject_raster(
        self,
        raster: np.ndarray,
        src_bbox: Optional[BoundingBox] = None,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """Reprojects and aligns a 2D geographic raster (EPSG:4326) onto target 128x128 EPSG:3857 grid.

        Uses bilinear interpolation to compute accurate continuous resampling.
        """
        if raster.shape == (self.h, self.w) and src_bbox is None:
            # Already matching grid dimensions
            return raster.astype(np.float32)

        source_box = src_bbox or self.bbox
        src_h, src_w = raster.shape

        # Create target pixel grid in Web Mercator space
        y_target, x_target = np.mgrid[0:self.h, 0:self.w]

        # Map target grid pixels back to Web Mercator meters
        x_m = self.x_min + (x_target / (self.w - 1)) * (self.x_max - self.x_min)
        y_m = self.y_max - (y_target / (self.h - 1)) * (self.y_max - self.y_min)

        # Convert meters back to source (lon, lat) degrees
        if self._has_pyproj and self._inv_transformer is not None:
            lons, lats = self._inv_transformer.transform(x_m, y_m)
        else:
            r_major = 6378137.0
            lons = np.degrees(x_m / r_major)
            lats = np.degrees(2.0 * np.arctan(np.exp(y_m / r_major)) - np.pi / 2.0)

        # Map (lon, lat) to source raster fractional coordinates
        src_x_norm = (lons - source_box.min_lon) / (source_box.max_lon - source_box.min_lon)
        src_y_norm = (source_box.max_lat - lats) / (source_box.max_lat - source_box.min_lat)

        src_coords_x = src_x_norm * (src_w - 1)
        src_coords_y = src_y_norm * (src_h - 1)

        coords = np.array([src_coords_y, src_coords_x])
        reprojected = map_coordinates(raster, coords, order=1, mode="constant", cval=fill_value)

        return reprojected.astype(np.float32)
