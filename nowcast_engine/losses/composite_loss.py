"""Multi-Task Composite Nowcasting Loss Module.

Combines Threshold-Weighted MSE for radar reflectivity, 2D Focal Loss for
sparse lightning strike density, and differentiable Structural Similarity (SSIM) loss.
"""

from __future__ import annotations

from typing import Dict, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from nowcast_engine.config import DEFAULT_LOSS_CONFIG, LossConfig
from nowcast_engine.losses.focal_loss import FocalLoss2D
from nowcast_engine.losses.weighted_mse import ThresholdWeightedMSELoss


class SSIMLoss2D(nn.Module):
    """Differentiable 2D Structural Similarity Index (SSIM) Loss.

    Loss = 1.0 - SSIM(pred, target)
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.window_size = window_size
        self.channel = 1
        # Create 1D Gaussian kernel
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2.0 * (sigma ** 2)))
        g = g / g.sum()

        # 2D Gaussian kernel: [1, 1, window_size, window_size]
        g2d = g.unsqueeze(1) * g.unsqueeze(0)
        self.register_buffer("window", g2d.unsqueeze(0).unsqueeze(0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            pred: [N, 1, H, W] in [0, 1]
            target: [N, 1, H, W] in [0, 1]
        """
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        raw_window = cast(torch.Tensor, self.window)
        window = raw_window.to(dtype=pred.dtype, device=pred.device)

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2) - mu1_mu2

        num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / (den + 1e-7)

        return 1.0 - ssim_map.mean()


class CompositeNowcastLoss(nn.Module):
    """Multi-task combined loss across radar reflectivity and lightning strike predictions."""

    def __init__(self, config: LossConfig = DEFAULT_LOSS_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.wmse = ThresholdWeightedMSELoss(config)
        self.focal = FocalLoss2D(gamma=config.focal_gamma, alpha=config.focal_alpha)
        self.ssim_loss = SSIMLoss2D()

        self.alpha = config.alpha_wmse
        self.beta = config.beta_focal
        self.lambd = config.lambda_ssim

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Args:
            pred: Predicted sequence [B, T=6, C=2, H=128, W=128]
            target: Ground truth sequence of matching shape

        Returns:
            Tuple of:
            - total_loss: Differentiable scalar tensor
            - loss_dict: Component losses as floats for logging
        """
        # Split channels: Ch 0 = Radar dBZ, Ch 1 = Lightning Density
        pred_ref = pred[:, :, 0:1]    # [B, T, 1, H, W]
        target_ref = target[:, :, 0:1]

        pred_light = pred[:, :, 1:2]  # [B, T, 1, H, W]
        target_light = target[:, :, 1:2]

        # 1. Threshold-Weighted MSE for Reflectivity
        loss_wmse = self.wmse(pred_ref, target_ref)

        # 2. Focal Loss for Sparse Lightning Density
        loss_focal = self.focal(pred_light, target_light)

        # 3. SSIM Loss for Reflectivity Spatial Structure
        # Flatten B and T dimensions for 2D SSIM convolution: [B*T, 1, H, W]
        b, t, c, h, w = pred_ref.shape
        flat_pred_ref = pred_ref.reshape(b * t, c, h, w)
        flat_target_ref = target_ref.reshape(b * t, c, h, w)
        loss_ssim = self.ssim_loss(flat_pred_ref, flat_target_ref)

        # Multi-task total loss
        total_loss = (self.alpha * loss_wmse) + (self.beta * loss_focal) + (self.lambd * loss_ssim)

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "wmse_loss": float(loss_wmse.item()),
            "focal_loss": float(loss_focal.item()),
            "ssim_loss": float(loss_ssim.item()),
        }

        return total_loss, loss_dict
