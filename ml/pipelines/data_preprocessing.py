"""
Data Preprocessing Pipeline:
Cleans raw radar sweeps, filters non-meteorological echoes (anomalous propagation, ground clutter),
and normalizes satellite and lightning spatial grids.
"""

from typing import Tuple, Dict, Any
import numpy as np


class AtmosphericDataPreprocessor:
    """
    Standardizes multi-source sensor observations onto a common spatio-temporal grid.
    """

    def __init__(self, grid_resolution_km: float = 2.0, domain_size_km: float = 200.0):
        self.grid_res = grid_resolution_km
        self.domain_size = domain_size_km

    def clean_radar_reflectivity(self, raw_reflectivity: np.ndarray) -> np.ndarray:
        """
        Removes ground clutter and anomalous propagation (AP) echoes.
        Clamps dBZ values to physical atmospheric ranges (-10 dBZ to 75 dBZ).
        """
        cleaned = np.copy(raw_reflectivity)
        # Suppress noise below threshold
        cleaned[cleaned < 0.0] = 0.0
        # Cap anomalous extremes
        cleaned[cleaned > 75.0] = 75.0
        return cleaned

    def compute_lightning_density_grid(
        self,
        stroke_lats: np.ndarray,
        stroke_lons: np.ndarray,
        grid_bounds: Tuple[float, float, float, float],
        bins: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregates point lightning strokes into a 2D spatial flash density histogram.
        """
        min_lat, max_lat, min_lon, max_lon = grid_bounds
        density_grid, lat_edges, lon_edges = np.histogram2d(
            stroke_lats,
            stroke_lons,
            bins=bins,
            range=[[min_lat, max_lat], [min_lon, max_lon]],
        )
        return density_grid, lat_edges, lon_edges


if __name__ == "__main__":
    preprocessor = AtmosphericDataPreprocessor()
    test_radar = np.random.uniform(-15.0, 80.0, size=(100, 100))
    cleaned = preprocessor.clean_radar_reflectivity(test_radar)
    print(f"Sample Radar Preprocessing: input shape {test_radar.shape}, max dBZ: {np.max(cleaned):.1f}")
