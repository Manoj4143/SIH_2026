"""Configuration parameters for Nowcast Engine deep learning models, losses, and training.

Calibrated for NVIDIA RTX 3050 (4GB VRAM ceiling), enforcing FP16 mixed precision,
low parameter footprint (< 3.5M parameters), and severe convective thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class ModelConfig:
    """Hyperparameters for the spatiotemporal NowcastNet architecture."""
    # Sequence & Spatial Canvas
    t_in: int = 4                # Historical input frames (t-45m to t+0m)
    t_out: int = 6               # Future forecast horizons (t+15m to t+180m)
    in_channels: int = 8         # [dBZ, Radial_Vel, IR_10.8, WV_6.8, Lightning_KDE, CAPE, CIN, Wind_Mag]
    out_channels: int = 2        # [dBZ (Reflectivity), Lightning_Density]
    img_height: int = 128
    img_width: int = 128

    # Depthwise Separable CNN Encoder
    encoder_dims: Tuple[int, ...] = (32, 64, 128)

    # 2-Layer ConvLSTM Bottleneck
    conv_lstm_hidden_dim: int = 128
    conv_lstm_num_layers: int = 2
    conv_lstm_kernel_size: int = 3

    # Skip-Connected Decoder
    decoder_dims: Tuple[int, ...] = (128, 64, 32)

    # Hardware Optimization
    use_separable_convs: bool = True
    dropout_rate: float = 0.1


@dataclass
class LossConfig:
    """Weights and parameters for multi-task meteorological loss formulation."""
    alpha_wmse: float = 1.0      # Weight for dBZ threshold-weighted MSE
    beta_focal: float = 0.5      # Weight for lightning focal loss
    lambda_ssim: float = 0.5     # Weight for structural similarity loss

    # dBZ Threshold Boundaries (in physical dBZ units)
    # Raw range [-10.0, 70.0] maps to [0.0, 1.0] via: norm = (dBZ - (-10)) / 80
    thresh_moderate_dbz: float = 20.0     # norm = 30/80 = 0.375
    thresh_severe_dbz: float = 35.0       # norm = 45/80 = 0.5625
    thresh_extreme_dbz: float = 45.0      # norm = 55/80 = 0.6875

    # Threshold penalty multipliers for Weighted MSE
    weight_light: float = 1.0
    weight_moderate: float = 2.0
    weight_severe: float = 5.0
    weight_extreme: float = 10.0

    # Focal Loss Parameters
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75


@dataclass
class TrainingConfig:
    """Hyperparameters for model training, optimization, and checkpointing."""
    batch_size: int = 2                   # Calibrated for RTX 3050 4GB VRAM
    grad_accum_steps: int = 2             # Effective batch size = 4
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    epochs: int = 30
    warmup_epochs: int = 3

    use_amp: bool = True                  # Automatic Mixed Precision (torch.cuda.amp / torch.amp)
    clip_grad_norm: float = 1.0

    checkpoint_dir: str = "nowcast_engine/weights"
    save_top_k: int = 3
    early_stopping_patience: int = 8

    # Target Benchmark Evaluation
    csi_target_threshold_dbz: float = 35.0
    csi_benchmark_target: float = 0.60


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_LOSS_CONFIG = LossConfig()
DEFAULT_TRAIN_CONFIG = TrainingConfig()
