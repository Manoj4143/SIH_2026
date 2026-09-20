"""Cosine Annealing with Linear Warmup Learning Rate Scheduler."""

from __future__ import annotations

import math
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """Linearly warms up learning rate over warmup_epochs then applies cosine decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 3,
        total_epochs: int = 30,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore[override]
        if not self._get_lr_called_within_step:  # type: ignore[attr-defined]
            pass

        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Linear warmup: scale from min_lr to base_lr
            warmup_factor = float(epoch + 1) / float(self.warmup_epochs)
            return [float(self.min_lr + (float(base_lr) - self.min_lr) * warmup_factor) for base_lr in self.base_lrs]

        # Cosine decay phase
        progress = float(epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
        progress = min(1.0, max(0.0, progress))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [float(self.min_lr + (float(base_lr) - self.min_lr) * cosine_factor) for base_lr in self.base_lrs]
