"""Threshold-Weighted Mean Squared Error (WMSE) Loss.

Meteorological weighted loss heavily penalizing forecasting errors on
convective storm cells (>= 35 dBZ) and extreme storm cores (>= 45 dBZ)
to prevent blurring and ensure severe hazard detection.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from nowcast_engine.config import DEFAULT_LOSS_CONFIG, LossConfig


class ThresholdWeightedMSELoss(nn.Module):
    """Computes meteorological threshold-weighted MSE on normalized radar reflectivity [0, 1].

    Normalized thresholds:
    - 20 dBZ: (20 - (-10)) / 80 = 0.375
    - 35 dBZ: (35 - (-10)) / 80 = 0.5625
    - 45 dBZ: (45 - (-10)) / 80 = 0.6875
    """

    def __init__(self, config: LossConfig = DEFAULT_LOSS_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.t_mod = (config.thresh_moderate_dbz - (-10.0)) / 80.0
        self.t_sev = (config.thresh_severe_dbz - (-10.0)) / 80.0
        self.t_ext = (config.thresh_extreme_dbz - (-10.0)) / 80.0

        self.w_light = config.weight_light
        self.w_mod = config.weight_moderate
        self.w_sev = config.weight_severe
        self.w_ext = config.weight_extreme

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes weighted MSE: (1/N) * sum(W(target) * (pred - target)^2).

        Args:
            pred: Predicted reflectivity [B, T, 1, H, W] or [B, T, H, W] in [0, 1]
            target: Ground truth reflectivity of matching shape

        Returns:
            Scalar tensor loss
        """
        # Weight matrix derived from target values
        weights = torch.full_like(target, fill_value=self.w_light)
        weights = torch.where(target >= self.t_mod, self.w_mod, weights)
        weights = torch.where(target >= self.t_sev, self.w_sev, weights)
        weights = torch.where(target >= self.t_ext, self.w_ext, weights)

        squared_errors = (pred - target) ** 2
        weighted_loss = weights * squared_errors

        return weighted_loss.mean()
