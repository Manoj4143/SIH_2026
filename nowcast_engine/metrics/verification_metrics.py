"""Meteorological Verification Metrics Module.

Implements standard atmospheric verification metrics:
- Critical Success Index (CSI / Threat Score)
- Probability of Detection (POD / Hit Rate)
- False Alarm Ratio (FAR)
- Structural Similarity Index (SSIM)
Evaluated at 20 dBZ, 35 dBZ (convective hazard threshold), and 45 dBZ (severe storm).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class MeteorologicalMetrics:
    """Evaluates nowcasting forecast performance against meteorological benchmarks."""

    DEFAULT_THRESHOLDS_DBZ = [20.0, 35.0, 45.0]

    @staticmethod
    def norm_to_dbz(norm_val: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Converts normalized [0, 1] reflectivity back to physical dBZ [-10, 70]."""
        return norm_val * 80.0 - 10.0

    @staticmethod
    def dbz_to_norm(dbz: float) -> float:
        """Converts physical dBZ to normalized [0, 1]."""
        return (dbz - (-10.0)) / 80.0

    @classmethod
    def compute_contingency_table(
        cls,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        threshold_dbz: float = 35.0,
    ) -> Tuple[int, int, int, int]:
        """Computes Hits, Misses, False Alarms, and Correct Negatives.

        Args:
            y_pred: Predicted reflectivity array in physical dBZ or normalized [0, 1]
            y_true: Ground truth reflectivity array
            threshold_dbz: Decision threshold in dBZ (e.g. 35.0 dBZ)

        Returns:
            Tuple of (hits, misses, false_alarms, correct_negatives)
        """
        # If arrays are normalized in [0, 1], convert threshold to normalized
        if np.max(y_true) <= 1.05 and threshold_dbz > 1.0:
            thresh = cls.dbz_to_norm(threshold_dbz)
        else:
            thresh = threshold_dbz

        pred_mask = (y_pred >= thresh)
        true_mask = (y_true >= thresh)

        hits = int(np.sum(pred_mask & true_mask))
        misses = int(np.sum((~pred_mask) & true_mask))
        false_alarms = int(np.sum(pred_mask & (~true_mask)))
        correct_negatives = int(np.sum((~pred_mask) & (~true_mask)))

        return hits, misses, false_alarms, correct_negatives

    @classmethod
    def compute_csi(
        cls,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        threshold_dbz: float = 35.0,
    ) -> float:
        """Critical Success Index: Hits / (Hits + Misses + False Alarms)."""
        hits, misses, fas, _ = cls.compute_contingency_table(y_pred, y_true, threshold_dbz)
        den = hits + misses + fas
        if den == 0:
            return 1.0  # Perfect agreement on clear air
        return float(hits / den)

    @classmethod
    def compute_pod(
        cls,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        threshold_dbz: float = 35.0,
    ) -> float:
        """Probability of Detection: Hits / (Hits + Misses)."""
        hits, misses, _, _ = cls.compute_contingency_table(y_pred, y_true, threshold_dbz)
        den = hits + misses
        if den == 0:
            return 1.0
        return float(hits / den)

    @classmethod
    def compute_far(
        cls,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        threshold_dbz: float = 35.0,
    ) -> float:
        """False Alarm Ratio: False Alarms / (Hits + False Alarms)."""
        hits, _, fas, _ = cls.compute_contingency_table(y_pred, y_true, threshold_dbz)
        den = hits + fas
        if den == 0:
            return 0.0
        return float(fas / den)

    @staticmethod
    def compute_ssim(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Computes structural similarity index for 2D rasters in [0, 1]."""
        from scipy.ndimage import uniform_filter

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        img1 = y_pred.astype(np.float64)
        img2 = y_true.astype(np.float64)

        mu1 = uniform_filter(img1, size=11)
        mu2 = uniform_filter(img2, size=11)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = uniform_filter(img1 * img1, size=11) - mu1_sq
        sigma2_sq = uniform_filter(img2 * img2, size=11) - mu2_sq
        sigma12 = uniform_filter(img1 * img2, size=11) - mu1_mu2

        num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / (den + 1e-7)

        return float(np.mean(ssim_map))

    @classmethod
    def evaluate_forecast_sequence(
        cls,
        pred_tensor: Union[torch.Tensor, np.ndarray],
        true_tensor: Union[torch.Tensor, np.ndarray],
        thresholds_dbz: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Evaluates complete multi-horizon forecast tensors.

        Args:
            pred_tensor: [B, T=6, C=2, H=128, W=128] or [T=6, C=2, H=128, W=128]
            true_tensor: Ground truth tensor

        Returns:
            Dictionary with CSI, POD, FAR across 20, 35, 45 dBZ and SSIM.
        """
        # Convert torch tensors to numpy if needed
        if isinstance(pred_tensor, torch.Tensor):
            y_p = pred_tensor.detach().cpu().numpy()
        else:
            y_p = np.asarray(pred_tensor)

        if isinstance(true_tensor, torch.Tensor):
            y_t = true_tensor.detach().cpu().numpy()
        else:
            y_t = np.asarray(true_tensor)

        # Reflectivity is channel 0
        ref_pred = y_p[..., 0, :, :]
        ref_true = y_t[..., 0, :, :]

        thresh_list = thresholds_dbz or cls.DEFAULT_THRESHOLDS_DBZ
        results: Dict[str, Any] = {}

        for thresh in thresh_list:
            tag = f"{int(thresh)}dBZ"
            csi = cls.compute_csi(ref_pred, ref_true, threshold_dbz=thresh)
            pod = cls.compute_pod(ref_pred, ref_true, threshold_dbz=thresh)
            far = cls.compute_far(ref_pred, ref_true, threshold_dbz=thresh)

            results[f"csi_{tag}"] = round(csi, 4)
            results[f"pod_{tag}"] = round(pod, 4)
            results[f"far_{tag}"] = round(far, 4)

        # Average SSIM across time frames
        t_steps = ref_pred.shape[-3] if len(ref_pred.shape) >= 3 else 1
        ssims = []
        if len(ref_pred.shape) == 4:  # [B, T, H, W]
            for b in range(ref_pred.shape[0]):
                for t in range(t_steps):
                    ssims.append(cls.compute_ssim(ref_pred[b, t], ref_true[b, t]))
        elif len(ref_pred.shape) == 3:  # [T, H, W]
            for t in range(t_steps):
                ssims.append(cls.compute_ssim(ref_pred[t], ref_true[t]))
        else:
            ssims.append(cls.compute_ssim(ref_pred, ref_true))

        results["mean_ssim"] = round(float(np.mean(ssims)), 4)
        return results
