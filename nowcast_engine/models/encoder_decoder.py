"""Spatial Encoder, Skip-Connected Decoder, and Dual Output Head Modules.

Implements Depthwise Separable Conv2D blocks to minimize parameters (<3.5M)
and VRAM footprint (<2.48 GB on RTX 3050), coupled with a U-Net style skip-connected
decoder and dual forecasting output heads.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise Separable 2D Convolution block for high computational efficiency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # Depthwise convolution: applies a single convolutional filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        # Pointwise convolution: 1x1 convolution computing linear combinations across channels
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.norm(out)
        return self.act(out)


class EncoderStage(nn.Module):
    """Single stage of spatial feature extraction and downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns:
            Tuple of (features_for_skip_connection, downsampled_features)
        """
        feat = self.conv2(self.conv1(x))
        down = self.downsample(feat)
        return feat, down


class SpatialEncoder(nn.Module):
    """Hierarchical Multi-Scale Spatial Encoder with Depthwise Separable Convolutions.

    Downsamples from 128x128 -> 64x64 -> 32x32.
    """

    def __init__(
        self,
        in_channels: int = 8,
        dims: Tuple[int, ...] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.dims = dims
        self.stage1 = EncoderStage(in_channels, dims[0], downsample=True)   # 128x128 -> 64x64
        self.stage2 = EncoderStage(dims[0], dims[1], downsample=True)       # 64x64 -> 32x32
        self.stage3 = EncoderStage(dims[1], dims[2], downsample=False)      # 32x32 bottleneck

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Args:
            x: Input frame [B, C=8, H=128, W=128]

        Returns:
            Tuple of:
            - bottleneck_features: [B, 128, 32, 32]
            - skip_features: List of [feat_128, feat_64]
        """
        skip1, down1 = self.stage1(x)       # skip1: [B, 32, 128, 128], down1: [B, 32, 64, 64]
        skip2, down2 = self.stage2(down1)   # skip2: [B, 64, 64, 64],   down2: [B, 64, 32, 32]
        bottleneck, _ = self.stage3(down2)  # bottleneck: [B, 128, 32, 32]

        return bottleneck, [skip1, skip2]


class DecoderStage(nn.Module):
    """Upsampling and skip-fusion stage."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv1 = DepthwiseSeparableConv2d((in_channels // 2) + skip_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.upsample(x)
        # Handle odd dimension padding if needed
        if x_up.shape[-2:] != skip.shape[-2:]:
            x_up = F.interpolate(x_up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([x_up, skip], dim=1)
        return self.conv2(self.conv1(fused))


class SkipConnectionDecoder(nn.Module):
    """Hierarchical U-Net style Decoder reconstructing fine 128x128 spatial resolution."""

    def __init__(
        self,
        bottleneck_dim: int = 128,
        decoder_dims: Tuple[int, ...] = (128, 64, 32),
    ) -> None:
        super().__init__()
        # stage_up1: 32x32 -> 64x64, fusing skip2 (dim 64)
        self.stage_up1 = DecoderStage(
            in_channels=bottleneck_dim,
            skip_channels=64,
            out_channels=decoder_dims[1],  # 64
        )
        # stage_up2: 64x64 -> 128x128, fusing skip1 (dim 32)
        self.stage_up2 = DecoderStage(
            in_channels=decoder_dims[1],
            skip_channels=32,
            out_channels=decoder_dims[2],  # 32
        )

    def forward(self, bottleneck: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Args:
            bottleneck: [B, 128, 32, 32]
            skips: [skip1 (128x128), skip2 (64x64)]

        Returns:
            Reconstructed spatial features: [B, 32, 128, 128]
        """
        skip1, skip2 = skips
        feat_64 = self.stage_up1(bottleneck, skip2)
        feat_128 = self.stage_up2(feat_64, skip1)
        return feat_128


class DualOutputHead(nn.Module):
    """Dual-branch forecasting head predicting radar reflectivity and lightning density."""

    def __init__(self, in_channels: int = 32) -> None:
        super().__init__()
        # Channel 0 Head: Radar Reflectivity field [0.0, 1.0]
        self.radar_head = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        # Channel 1 Head: Total Lightning Strike Density [0.0, 1.0]
        self.lightning_head = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, in_channels, 128, 128]

        Returns:
            Dual hazard predictions: [B, 2, 128, 128]
            - Ch 0: Reflectivity (dBZ)
            - Ch 1: Lightning Density
        """
        ref = self.radar_head(x)
        light = self.lightning_head(x)
        return torch.cat([ref, light], dim=1)
