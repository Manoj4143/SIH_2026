"""Unit tests for meteorological loss functions, gradients, and penalty weights."""

import torch
import pytest

from nowcast_engine.config import LossConfig
from nowcast_engine.losses.composite_loss import CompositeNowcastLoss, SSIMLoss2D
from nowcast_engine.losses.focal_loss import FocalLoss2D
from nowcast_engine.losses.weighted_mse import ThresholdWeightedMSELoss


class TestLossFunctions:
    def test_weighted_mse_penalties(self):
        loss_fn = ThresholdWeightedMSELoss()

        # Fixed error delta of 0.2
        # Target 1: Clear air (0.1 normalized, ~ -2 dBZ) -> weight = 1.0
        target_light = torch.full((1, 1, 1, 32, 32), fill_value=0.1, requires_grad=False)
        pred_light = target_light + 0.2
        loss_light = loss_fn(pred_light, target_light)

        # Target 2: Severe convective storm (0.60 normalized, ~ 38 dBZ) -> weight = 5.0
        target_severe = torch.full((1, 1, 1, 32, 32), fill_value=0.60, requires_grad=False)
        pred_severe = target_severe + 0.2
        loss_severe = loss_fn(pred_severe, target_severe)

        # Target 3: Extreme convective hail core (0.75 normalized, ~ 50 dBZ) -> weight = 10.0
        target_extreme = torch.full((1, 1, 1, 32, 32), fill_value=0.75, requires_grad=False)
        pred_extreme = target_extreme + 0.2
        loss_extreme = loss_fn(pred_extreme, target_extreme)

        # Verify severe errors are penalized strictly higher than light errors
        assert loss_severe > loss_light * 4.0
        assert loss_extreme > loss_severe * 1.5

    def test_weighted_mse_backprop(self):
        loss_fn = ThresholdWeightedMSELoss()
        pred = torch.randn(2, 6, 1, 64, 64, requires_grad=True)
        target = torch.clamp(torch.randn(2, 6, 1, 64, 64), 0.0, 1.0)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_focal_loss_sparse_lightning(self):
        focal_fn = FocalLoss2D(gamma=2.0, alpha=0.75)

        # Mostly empty target with single flash burst cluster
        target = torch.zeros(2, 6, 1, 64, 64)
        target[:, :, :, 30:35, 30:35] = 1.0

        pred = torch.full((2, 6, 1, 64, 64), fill_value=0.01, requires_grad=True)
        loss = focal_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert loss.item() > 0.0
        assert not torch.isnan(pred.grad).any()

    def test_ssim_loss(self):
        ssim_fn = SSIMLoss2D()

        # Identical images should have SSIM ~ 1.0 (loss ~ 0.0)
        img = torch.rand(2, 1, 64, 64)
        loss_ident = ssim_fn(img, img)
        assert loss_ident.item() < 1e-3

        # Inverted image should have high loss
        loss_inv = ssim_fn(img, 1.0 - img)
        assert loss_inv.item() > 0.5

    def test_composite_loss(self):
        comp_loss = CompositeNowcastLoss()
        pred = torch.rand(2, 6, 2, 64, 64, requires_grad=True)
        target = torch.rand(2, 6, 2, 64, 64)

        total_loss, loss_dict = comp_loss(pred, target)
        total_loss.backward()

        assert pred.grad is not None
        assert "total_loss" in loss_dict
        assert "wmse_loss" in loss_dict
        assert "focal_loss" in loss_dict
        assert "ssim_loss" in loss_dict
        assert loss_dict["total_loss"] > 0.0
