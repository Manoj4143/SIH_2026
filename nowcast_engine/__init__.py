"""Nowcast Engine Microservice.

SIH Problem Statement 26072: Multi-Source Deep Learning Weather Nowcasting System.
Team ASTRO (Disaster Management).
"""

from nowcast_engine.config import (
    DEFAULT_LOSS_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)
from nowcast_engine.losses.composite_loss import CompositeNowcastLoss, SSIMLoss2D
from nowcast_engine.losses.focal_loss import FocalLoss2D
from nowcast_engine.losses.weighted_mse import ThresholdWeightedMSELoss
from nowcast_engine.metrics.verification_metrics import MeteorologicalMetrics
from nowcast_engine.models.conv_lstm_cell import ConvLSTMCell, ConvLSTMBottleneck, ConvLSTMLayer
from nowcast_engine.models.encoder_decoder import (
    DecoderStage,
    DepthwiseSeparableConv2d,
    DualOutputHead,
    EncoderStage,
    SkipConnectionDecoder,
    SpatialEncoder,
)
from nowcast_engine.models.nowcast_net import NowcastNet
from nowcast_engine.training.lr_scheduler import CosineWarmupScheduler
from nowcast_engine.training.trainer import NowcastTrainer

__all__ = [
    "ModelConfig",
    "LossConfig",
    "TrainingConfig",
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_LOSS_CONFIG",
    "DEFAULT_TRAIN_CONFIG",
    "NowcastNet",
    "ConvLSTMCell",
    "ConvLSTMLayer",
    "ConvLSTMBottleneck",
    "DepthwiseSeparableConv2d",
    "EncoderStage",
    "SpatialEncoder",
    "DecoderStage",
    "SkipConnectionDecoder",
    "DualOutputHead",
    "ThresholdWeightedMSELoss",
    "FocalLoss2D",
    "SSIMLoss2D",
    "CompositeNowcastLoss",
    "MeteorologicalMetrics",
    "CosineWarmupScheduler",
    "NowcastTrainer",
]
