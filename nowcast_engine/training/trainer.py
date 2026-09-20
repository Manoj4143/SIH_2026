"""PyTorch Automatic Mixed Precision (AMP) Trainer for NowcastNet.

Calibrated for NVIDIA RTX 3050 (4GB VRAM limit), employing torch.amp.autocast,
gradient accumulation, learning rate warmup, and validation CSI-based checkpointing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from nowcast_engine.config import (
    DEFAULT_LOSS_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    LossConfig,
    ModelConfig,
    TrainingConfig,
)
from nowcast_engine.losses.composite_loss import CompositeNowcastLoss
from nowcast_engine.metrics.verification_metrics import MeteorologicalMetrics
from nowcast_engine.models.nowcast_net import NowcastNet
from nowcast_engine.training.lr_scheduler import CosineWarmupScheduler

logger = logging.getLogger(__name__)


class NowcastTrainer:
    """End-to-end training orchestrator for spatiotemporal convective nowcasting."""

    def __init__(
        self,
        model: Optional[NowcastNet] = None,
        model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
        loss_config: LossConfig = DEFAULT_LOSS_CONFIG,
        train_config: TrainingConfig = DEFAULT_TRAIN_CONFIG,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_config = model_config
        self.loss_config = loss_config
        self.train_config = train_config

        # Device assignment: CUDA if available else CPU
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Initializing NowcastTrainer on device: %s", self.device)

        # Model instantiation
        self.model = model or NowcastNet(model_config)
        self.model.to(self.device)

        # Multi-task loss
        self.criterion = CompositeNowcastLoss(loss_config).to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=train_config.warmup_epochs,
            total_epochs=train_config.epochs,
            min_lr=train_config.min_lr,
        )

        # Automatic Mixed Precision GradScaler
        self.use_amp = train_config.use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Checkpoints
        self.checkpoint_dir = Path(train_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_csi_35 = -1.0
        self.saved_checkpoints: List[Tuple[float, Path]] = []

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Executes a single training epoch with AMP and gradient accumulation."""
        self.model.train()
        accum_steps = self.train_config.grad_accum_steps
        running_losses: Dict[str, float] = {
            "total_loss": 0.0,
            "wmse_loss": 0.0,
            "focal_loss": 0.0,
            "ssim_loss": 0.0,
        }
        num_batches = len(train_loader)
        self.optimizer.zero_grad()

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                predictions = self.model(batch_x)
                loss, loss_dict = self.criterion(predictions, batch_y)
                scaled_loss = loss / accum_steps

            # Backward pass
            self.scaler.scale(scaled_loss).backward()

            # Gradient accumulation step
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
                if self.train_config.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.train_config.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            for k, v in loss_dict.items():
                running_losses[k] += v

        # Average losses over batches
        avg_losses = {k: round(v / max(1, num_batches), 4) for k, v in running_losses.items()}
        return avg_losses

    def validate(self, val_loader: DataLoader) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Evaluates model over validation dataset computing losses and meteorological metrics."""
        self.model.eval()
        running_losses: Dict[str, float] = {
            "total_loss": 0.0,
            "wmse_loss": 0.0,
            "focal_loss": 0.0,
            "ssim_loss": 0.0,
        }
        all_preds = []
        all_targets = []
        num_batches = len(val_loader)

        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    preds = self.model(batch_x)
                    _, loss_dict = self.criterion(preds, batch_y)

                for k, v in loss_dict.items():
                    running_losses[k] += v

                all_preds.append(preds.cpu())
                all_targets.append(batch_y.cpu())

        avg_losses = {k: round(v / max(1, num_batches), 4) for k, v in running_losses.items()}

        if all_preds:
            cat_preds = torch.cat(all_preds, dim=0)
            cat_targets = torch.cat(all_targets, dim=0)
            metrics = MeteorologicalMetrics.evaluate_forecast_sequence(cat_preds, cat_targets)
        else:
            metrics = {"csi_35dBZ": 0.0, "pod_35dBZ": 0.0, "far_35dBZ": 0.0, "mean_ssim": 0.0}

        return avg_losses, metrics

    def save_checkpoint(
        self,
        epoch: int,
        val_csi_35: float,
        val_losses: Dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """Saves model weights and training metadata."""
        ckpt_name = f"nowcast_best_csi35_{val_csi_35:.4f}_epoch_{epoch}.pt" if is_best else f"nowcast_epoch_{epoch}.pt"
        ckpt_path = self.checkpoint_dir / ckpt_name

        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_csi_35": val_csi_35,
            "val_losses": val_losses,
            "model_config": self.model_config,
            "train_config": self.train_config,
        }
        torch.save(payload, ckpt_path)
        logger.info("Saved model checkpoint: %s", ckpt_path)

        if is_best:
            best_link = self.checkpoint_dir / "best_model.pt"
            torch.save(payload, best_link)

        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Loads weights and optimizer state from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        self.best_csi_35 = checkpoint.get("val_csi_35", 0.0)
        logger.info("Loaded checkpoint from %s (Epoch %d, CSI_35: %.4f)", checkpoint_path, epoch, self.best_csi_35)
        return int(epoch)
