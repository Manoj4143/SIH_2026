"""
Meteorological Evaluation Metrics:
Calculates standard severe weather nowcasting verification metrics:
- CSI (Critical Success Index / Threat Score)
- POD (Probability of Detection)
- FAR (False Alarm Ratio)
- Brier Score
- HSS (Heidke Skill Score)
"""

from typing import Dict
import numpy as np


class NowcastEvaluator:
    """
    Computes meteorological contingency table verification scores.
    """

    @staticmethod
    def calculate_contingency_table(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """
        Calculates 2x2 contingency elements:
        - Hits (a): true=1, pred=1
        - False alarms (b): true=0, pred=1
        - Misses (c): true=1, pred=0
        - Correct negatives (d): true=0, pred=0
        """
        y_t = np.array(y_true, dtype=bool)
        y_p = np.array(y_pred, dtype=bool)

        hits = int(np.sum(y_t & y_p))
        false_alarms = int(np.sum((~y_t) & y_p))
        misses = int(np.sum(y_t & (~y_p)))
        correct_negatives = int(np.sum((~y_t) & (~y_p)))

        return {
            "hits": hits,
            "false_alarms": false_alarms,
            "misses": misses,
            "correct_negatives": correct_negatives,
        }

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
        table = NowcastEvaluator.calculate_contingency_table(y_true, y_pred)
        a = table["hits"]
        b = table["false_alarms"]
        c = table["misses"]
        d = table["correct_negatives"]

        # Probability of Detection (POD) = a / (a + c)
        pod = a / (a + c) if (a + c) > 0 else 0.0

        # False Alarm Ratio (FAR) = b / (a + b)
        far = b / (a + b) if (a + b) > 0 else 0.0

        # Critical Success Index (CSI) = a / (a + b + c)
        csi = a / (a + b + c) if (a + b + c) > 0 else 0.0

        # Heidke Skill Score (HSS)
        num = 2 * (a * d - b * c)
        den = ((a + c) * (c + d) + (a + b) * (b + d))
        hss = num / den if den > 0 else 0.0

        metrics = {
            "POD": round(pod, 4),
            "FAR": round(far, 4),
            "CSI": round(csi, 4),
            "HSS": round(hss, 4),
            **table,
        }

        if y_prob is not None:
            # Brier Score = 1/N * sum((prob - true)^2)
            brier = float(np.mean((np.array(y_prob) - np.array(y_true)) ** 2))
            metrics["Brier_Score"] = round(brier, 4)

        return metrics


if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    scores = NowcastEvaluator.compute_metrics(y_true, y_pred)
    print("Sample Verification Metrics:")
    for k, v in scores.items():
        print(f"  {k}: {v}")
