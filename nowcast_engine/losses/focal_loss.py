"""Focal Loss Module for Sparse Lightning Flash Density Maps.

Addresses severe class imbalance where non-striking regions (>95%)
dominate the spatial canvas, focusing learning gradients on active
convective flash bursts.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nowcast_engine.config import DEFAULT_LOSS_CONFIG, LossConfig


class FocalLoss2D(nn.Module):
    """2D Spatial Focal Loss for continuous/binarized density distributions in [0, 1].

    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            pred: Predicted probabilities/density in [0, 1]
            target: Ground truth density in [0, 1]

        Returns:
            Scalar tensor loss
        """
        pred_clamped = torch.clamp(pred, min=self.eps, max=1.0 - self.eps)

        # Cross-entropy components
        pt_pos = pred_clamped
        pt_neg = 1.0 - pred_clamped

        # Modulating factors
        mod_pos = (1.0 - pt_pos) ** self.gamma
        mod_neg = (1.0 - pt_neg) ** self.gamma

        loss_pos = -self.alpha * mod_pos * torch.log(pt_pos) * target
        loss_neg = -(1.0 - self.alpha) * mod_neg * torch.log(pt_neg) * (1.0 - target)

        focal_loss = loss_pos + loss_neg
        return focal_loss.mean()
