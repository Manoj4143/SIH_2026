"""Multi-Channel MinMax Normalization Module.

Applies standardized MinMax scaling across all 8 meteorological channels to
map raw physical units strictly into [0.0, 1.0] floating-point representations,
with reversible inverse transformations for inference visualization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from data_pipeline.config import CHANNEL_NAMES, CHANNEL_SPECS, ChannelBoundary

logger = logging.getLogger(__name__)


class ChannelScaler:
    """Standardized MinMax scaler for 8-channel weather input tensors."""

    def __init__(self, channel_specs: Optional[List[ChannelBoundary]] = None) -> None:
        self.specs = channel_specs or CHANNEL_SPECS
        self.channel_names = [s.name for s in self.specs]
        self.min_vals = np.array([s.min_val for s in self.specs], dtype=np.float32)
        self.max_vals = np.array([s.max_val for s in self.specs], dtype=np.float32)
        self.ranges = self.max_vals - self.min_vals

        # Mapping for fast lookup
        self._name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.channel_names)}

    def normalize_channel(
        self,
        data: np.ndarray,
        channel: Union[str, int],
    ) -> np.ndarray:
        """Scales a single channel raster from raw physical units into [0.0, 1.0]."""
        idx = self._name_to_idx[channel] if isinstance(channel, str) else channel
        min_v = self.min_vals[idx]
        max_v = self.max_vals[idx]
        rng = self.ranges[idx]

        clamped = np.clip(data.astype(np.float32), min_v, max_v)
        normalized = (clamped - min_v) / rng
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def denormalize_channel(
        self,
        data_norm: np.ndarray,
        channel: Union[str, int],
    ) -> np.ndarray:
        """Restores a normalized [0.0, 1.0] channel raster back to physical units."""
        idx = self._name_to_idx[channel] if isinstance(channel, str) else channel
        min_v = self.min_vals[idx]
        rng = self.ranges[idx]

        denorm = data_norm.astype(np.float32) * rng + min_v
        return denorm.astype(np.float32)

    def normalize_tensor(self, tensor: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """Normalizes an 8-channel tensor where channel dimension is C=8.

        Supports shapes:
        - [C, H, W] = [8, 128, 128]
        - [T, C, H, W] = [4, 8, 128, 128]
        - [B, T, C, H, W] = [2, 4, 8, 128, 128]
        """
        is_torch = False
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                is_torch = True
                device = tensor.device
                dtype = tensor.dtype
                arr = tensor.detach().cpu().numpy()
            else:
                arr = np.asarray(tensor)
        except ImportError:
            arr = np.asarray(tensor)

        norm_arr = np.empty_like(arr, dtype=np.float32)

        # Detect channel axis (dimension of size 8)
        if arr.shape[0] == 8 and len(arr.shape) == 3:
            # [C, H, W]
            for c in range(8):
                norm_arr[c] = self.normalize_channel(arr[c], c)
        elif len(arr.shape) == 4 and arr.shape[1] == 8:
            # [T, C, H, W]
            for c in range(8):
                norm_arr[:, c] = self.normalize_channel(arr[:, c], c)
        elif len(arr.shape) == 5 and arr.shape[2] == 8:
            # [B, T, C, H, W]
            for c in range(8):
                norm_arr[:, :, c] = self.normalize_channel(arr[:, :, c], c)
        else:
            raise ValueError(
                f"Unsupported tensor shape {arr.shape}. Expected channel dimension of size 8 "
                f"at axis 0 (3D), axis 1 (4D), or axis 2 (5D)."
            )

        if is_torch:
            import torch
            return torch.from_numpy(norm_arr).to(device=device, dtype=dtype)
        return norm_arr

    def denormalize_tensor(self, tensor: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """Denormalizes an 8-channel tensor back to physical meteorological units."""
        is_torch = False
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                is_torch = True
                device = tensor.device
                dtype = tensor.dtype
                arr = tensor.detach().cpu().numpy()
            else:
                arr = np.asarray(tensor)
        except ImportError:
            arr = np.asarray(tensor)

        denorm_arr = np.empty_like(arr, dtype=np.float32)

        if arr.shape[0] == 8 and len(arr.shape) == 3:
            for c in range(8):
                denorm_arr[c] = self.denormalize_channel(arr[c], c)
        elif len(arr.shape) == 4 and arr.shape[1] == 8:
            for c in range(8):
                denorm_arr[:, c] = self.denormalize_channel(arr[:, c], c)
        elif len(arr.shape) == 5 and arr.shape[2] == 8:
            for c in range(8):
                denorm_arr[:, :, c] = self.denormalize_channel(arr[:, :, c], c)
        else:
            raise ValueError(f"Unsupported tensor shape {arr.shape} for denormalization.")

        if is_torch:
            import torch
            return torch.from_numpy(denorm_arr).to(device=device, dtype=dtype)
        return denorm_arr
