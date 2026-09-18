"""
Feature Engineering Pipeline:
Extracts meteorological predictor features from radar, satellite, lightning, and NWP fields.
"""

from typing import Dict
import numpy as np
import pandas as pd


class AtmosphericFeatureExtractor:
    """
    Computes convective storm features for Scikit-learn and XGBoost models.
    """

    def extract_cell_features(
        self,
        radar_dbz_history: np.ndarray,
        satellite_ir_temp: float,
        lightning_counts_15m: int,
        cape_j_kg: float,
        shear_0_6km_m_s: float,
    ) -> Dict[str, float]:
        """
        Extracts tabular features for a given candidate storm cell or grid cell.

        :param radar_dbz_history: Array of dBZ values over the past [t-20, t-10, t] minutes
        :param satellite_ir_temp: Infrared brightness temp in Kelvin (lower = higher cloud tops)
        :param lightning_counts_15m: Stroke counts in past 15 min
        :param cape_j_kg: Convective Available Potential Energy (NWP)
        :param shear_0_6km_m_s: Deep layer vertical wind shear
        """
        current_dbz = float(radar_dbz_history[-1])
        dbz_tendency_10m = float(radar_dbz_history[-1] - radar_dbz_history[-2]) if len(radar_dbz_history) >= 2 else 0.0
        dbz_tendency_20m = float(radar_dbz_history[-1] - radar_dbz_history[0]) if len(radar_dbz_history) >= 3 else 0.0

        # Estimated Vertically Integrated Liquid (heuristic empirical relation)
        vil_approx = 0.00344 * (10 ** (0.055 * current_dbz)) if current_dbz > 15.0 else 0.0

        return {
            "current_max_dbz": current_dbz,
            "dbz_tendency_10m": dbz_tendency_10m,
            "dbz_tendency_20m": dbz_tendency_20m,
            "vil_kg_m2": round(vil_approx, 2),
            "satellite_ir_kelvin": satellite_ir_temp,
            "is_overshooting_top": 1.0 if satellite_ir_temp < 210.0 else 0.0,
            "lightning_flash_rate_15m": float(lightning_counts_15m),
            "cape_j_kg": cape_j_kg,
            "deep_layer_shear": shear_0_6km_m_s,
            "energy_shear_index": round((cape_j_kg * shear_0_6km_m_s) / 1000.0, 3),
        }

    def create_synthetic_training_dataset(self, num_samples: int = 1200) -> pd.DataFrame:
        """
        Generates realistic synthetic atmospheric feature rows for initializing and testing ML models.
        """
        np.random.seed(42)
        records = []

        for _ in range(num_samples):
            current_dbz = np.random.uniform(15.0, 65.0)
            tendency_10 = np.random.normal(1.2, 3.5)
            tendency_20 = tendency_10 + np.random.normal(0.8, 4.0)
            ir_temp = np.random.uniform(200.0, 285.0)
            lightning_count = int(max(0, np.random.poisson(lam=max(0.1, (current_dbz - 30.0) * 0.8))))
            cape = np.random.uniform(500.0, 4200.0)
            shear = np.random.uniform(5.0, 35.0)

            # Heuristic target: Will cell produce severe thunderstorm (dBZ >= 40 at t+30m)?
            prob_thunderstorm = (
                0.35 * (current_dbz / 60.0)
                + 0.20 * (max(0, tendency_10) / 10.0)
                + 0.20 * (cape / 3500.0)
                + 0.15 * (1.0 if ir_temp < 225.0 else 0.0)
                + 0.10 * (min(30, lightning_count) / 30.0)
            )
            target_thunderstorm = 1 if prob_thunderstorm + np.random.normal(0, 0.1) > 0.55 else 0
            target_lightning = 1 if (lightning_count > 3 or (current_dbz > 42 and cape > 2000)) else 0

            records.append({
                "current_max_dbz": current_dbz,
                "dbz_tendency_10m": tendency_10,
                "dbz_tendency_20m": tendency_20,
                "vil_kg_m2": 0.00344 * (10 ** (0.055 * current_dbz)),
                "satellite_ir_kelvin": ir_temp,
                "is_overshooting_top": 1.0 if ir_temp < 210.0 else 0.0,
                "lightning_flash_rate_15m": lightning_count,
                "cape_j_kg": cape,
                "deep_layer_shear": shear,
                "energy_shear_index": (cape * shear) / 1000.0,
                "target_thunderstorm_30m": target_thunderstorm,
                "target_lightning_30m": target_lightning,
            })

        return pd.DataFrame(records)


if __name__ == "__main__":
    fe = AtmosphericFeatureExtractor()
    sample_df = fe.create_synthetic_training_dataset(5)
    print("Sample feature rows:")
    print(sample_df[["current_max_dbz", "cape_j_kg", "target_thunderstorm_30m", "target_lightning_30m"]])
