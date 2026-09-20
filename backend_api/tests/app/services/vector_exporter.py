"""Vector Contouring & Severe Storm Alert Polygon Generator.

Extracts continuous hazard contours for severe reflectivity (>= 35 dBZ) and lightning strikes,
emitting standardized RFC 7946 GeoJSON FeatureCollections for map dashboard rendering.
"""

from __future__ import annotations

import datetime
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon, box, mapping
from shapely.ops import unary_union

from backend_api.app.config import settings


class VectorExporter:
    """Extracts hazard polygons and formats them into GeoJSON features."""

    def __init__(
        self,
        domain_bbox: Optional[Tuple[float, float, float, float]] = None,
        grid_shape: Tuple[int, int] = (128, 128),
    ) -> None:
        self.domain = domain_bbox or settings.domain_bbox
        self.H, self.W = grid_shape
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = self.domain

        # Pixel resolution in degrees
        self.d_lon = (self.max_lon - self.min_lon) / float(self.W)
        self.d_lat = (self.max_lat - self.min_lat) / float(self.H)

    def pixel_to_lon_lat(self, col: float, row: float) -> Tuple[float, float]:
        """Converts raster pixel (col, row) to geographic (lon, lat)."""
        lon = self.min_lon + col * self.d_lon
        lat = self.max_lat - row * self.d_lat
        return (lon, lat)

    def extract_alert_polygons_for_step(
        self,
        radar_dbz_2d: np.ndarray,
        lightning_2d: np.ndarray,
        lead_time_minutes: int,
        base_timestamp: Optional[datetime.datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Extracts severe hazard polygons for a single forecast horizon.

        Args:
            radar_dbz_2d: 2D array of radar reflectivity in dBZ [-10, 70].
            lightning_2d: 2D array of lightning strike density [0.0, 1.0].
            lead_time_minutes: Forecast horizon (e.g. 15, 30, ...).
            base_timestamp: Base forecast issuance timestamp.

        Returns:
            List of GeoJSON Feature dictionaries.
        """
        now = base_timestamp or datetime.datetime.now(datetime.timezone.utc)
        valid_time = now + datetime.timedelta(minutes=lead_time_minutes)
        expires_time = valid_time + datetime.timedelta(minutes=15)

        features: List[Dict[str, Any]] = []

        # 1. Evaluate Radar Reflectivity Hazard (> 35 dBZ)
        features.extend(
            self._extract_radar_hazards(
                radar_dbz_2d,
                lead_time_minutes,
                valid_time,
                expires_time,
            )
        )

        # 2. Evaluate Lightning Density Hazard (> 0.30)
        features.extend(
            self._extract_lightning_hazards(
                lightning_2d,
                lead_time_minutes,
                valid_time,
                expires_time,
            )
        )

        return features

    def _extract_radar_hazards(
        self,
        dbz_field: np.ndarray,
        lead_time_min: int,
        valid_time: datetime.datetime,
        expires_time: datetime.datetime,
    ) -> List[Dict[str, Any]]:
        """Finds connected severe storm components (>= 35 dBZ)."""
        features: List[Dict[str, Any]] = []
        mask_35 = dbz_field >= settings.DBZ_SEVERE

        if not np.any(mask_35):
            return features

        # Label connected components
        labeled_array, num_features = ndimage.label(mask_35)

        for comp_id in range(1, num_features + 1):
            comp_mask = labeled_array == comp_id
            num_pixels = int(np.sum(comp_mask))

            # Filter tiny noise specks (< 3 pixels, ~3 km2)
            if num_pixels < 3:
                continue

            max_dbz = float(np.max(dbz_field[comp_mask]))
            mean_dbz = float(np.mean(dbz_field[comp_mask]))

            # Classify severity
            if max_dbz >= settings.DBZ_EXTREME:
                severity = "EMERGENCY"
                color = "#8B0000"
                headline = f"EMERGENCY: Extreme Hail / Convective Storm Core ({max_dbz:.1f} dBZ)"
            elif max_dbz >= settings.DBZ_SEVERE:
                severity = "WARNING"
                color = "#FF4500"
                headline = f"WARNING: Severe Thunderstorm ({max_dbz:.1f} dBZ)"
            else:
                severity = "ADVISORY"
                color = "#FFA500"
                headline = f"ADVISORY: Moderate Convection ({max_dbz:.1f} dBZ)"

            poly = self._mask_to_simplified_polygon(comp_mask)
            if poly is None or poly.is_empty:
                continue

            # Calculate area approx (1 pixel ~ 1 km^2 in EPSG:3857)
            area_km2 = float(num_pixels * 1.0)
            alert_id = f"ALT-RADAR-{lead_time_min}M-{uuid.uuid4().hex[:8].upper()}"

            feature = {
                "type": "Feature",
                "id": alert_id,
                "geometry": mapping(poly),
                "properties": {
                    "alert_id": alert_id,
                    "lead_time_minutes": lead_time_min,
                    "valid_time": valid_time.isoformat(),
                    "severity": severity,
                    "hazard_type": "Severe Convective Radar Core",
                    "max_dbz": round(max_dbz, 1),
                    "mean_dbz": round(mean_dbz, 1),
                    "max_lightning_prob": 0.0,
                    "area_km2": round(area_km2, 2),
                    "headline": headline,
                    "description": (
                        f"Radar cell projected at +{lead_time_min}m. "
                        f"Peak reflectivity: {max_dbz:.1f} dBZ across ~{area_km2:.0f} km²."
                    ),
                    "color": color,
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "expires_at": expires_time.isoformat(),
                },
            }
            features.append(feature)

        return features

    def _extract_lightning_hazards(
        self,
        lightning_field: np.ndarray,
        lead_time_min: int,
        valid_time: datetime.datetime,
        expires_time: datetime.datetime,
    ) -> List[Dict[str, Any]]:
        """Finds connected clusters of high lightning activity."""
        features: List[Dict[str, Any]] = []
        mask_light = lightning_field >= settings.LIGHTNING_MODERATE

        if not np.any(mask_light):
            return features

        labeled_array, num_features = ndimage.label(mask_light)

        for comp_id in range(1, num_features + 1):
            comp_mask = labeled_array == comp_id
            num_pixels = int(np.sum(comp_mask))

            if num_pixels < 3:
                continue

            peak_prob = float(np.max(lightning_field[comp_mask]))
            mean_prob = float(np.mean(lightning_field[comp_mask]))

            if peak_prob >= settings.LIGHTNING_HIGH:
                severity = "WARNING"
                color = "#990099"
                headline = f"WARNING: High-Frequency Lightning Danger ({peak_prob*100:.0f}%)"
            else:
                severity = "ADVISORY"
                color = "#CC6600"
                headline = f"ADVISORY: Active Lightning Risk ({peak_prob*100:.0f}%)"

            poly = self._mask_to_simplified_polygon(comp_mask)
            if poly is None or poly.is_empty:
                continue

            area_km2 = float(num_pixels * 1.0)
            alert_id = f"ALT-LIGHT-{lead_time_min}M-{uuid.uuid4().hex[:8].upper()}"

            feature = {
                "type": "Feature",
                "id": alert_id,
                "geometry": mapping(poly),
                "properties": {
                    "alert_id": alert_id,
                    "lead_time_minutes": lead_time_min,
                    "valid_time": valid_time.isoformat(),
                    "severity": severity,
                    "hazard_type": "Lightning Flash Density Hazard",
                    "max_dbz": 0.0,
                    "mean_dbz": 0.0,
                    "max_lightning_prob": round(peak_prob, 3),
                    "area_km2": round(area_km2, 2),
                    "headline": headline,
                    "description": (
                        f"Frequent lightning activity predicted at +{lead_time_min}m. "
                        f"Peak flash probability: {peak_prob*100:.1f}%."
                    ),
                    "color": color,
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "expires_at": expires_time.isoformat(),
                },
            }
            features.append(feature)

        return features

    def _mask_to_simplified_polygon(self, mask: np.ndarray) -> Optional[Polygon | MultiPolygon]:
        """Converts a binary pixel mask to a geographic Polygon with smoothed topology."""
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return None

        boxes = []
        for r, c in zip(rows, cols):
            lon_min, lat_max = self.pixel_to_lon_lat(c, r)
            lon_max, lat_min = self.pixel_to_lon_lat(c + 1, r + 1)
            boxes.append(box(lon_min, lat_min, lon_max, lat_max))

        merged = unary_union(boxes)
        # Clean potential invalid geometries and simplify coordinate count
        clean_geom = merged.buffer(0).simplify(tolerance=0.002, preserve_topology=True)
        return clean_geom

    def build_feature_collection(
        self,
        features: List[Dict[str, Any]],
        query_timestamp: Optional[datetime.datetime] = None,
    ) -> Dict[str, Any]:
        """Wraps features in a standard GeoJSON FeatureCollection."""
        ts = query_timestamp or datetime.datetime.now(datetime.timezone.utc)
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "generated_at": ts.isoformat(),
                "total_alerts": len(features),
                "domain_bounds": list(self.domain),
            },
        }


vector_exporter = VectorExporter()
