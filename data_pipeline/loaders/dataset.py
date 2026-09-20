"""PyTorch WeatherTensorDataset and DataLoader Module.

Constructs sequence pairs of historical inputs [T_in=4, C=8, H=128, W=128]
and future forecast targets [T_out=6, C_target=2, H=128, W=128], supporting
both HDF5 disk archives and live in-memory streaming tensors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np

from data_pipeline.config import (
    CHANNEL_NAMES,
    DEFAULT_CONFIG,
    TARGET_CHANNEL_NAMES,
    PipelineConfig,
)

logger = logging.getLogger(__name__)

# Try importing torch; if not available yet (in process of installing), fallback to base object
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:  # type: ignore
        pass
    class DataLoader:  # type: ignore
        pass


class WeatherTensorDataset(Dataset):
    """Custom PyTorch Dataset emitting standardized 8-channel spatiotemporal tensors.

    Input sequence:
        X: [T_in=4, C=8, H=128, W=128]
        Channels: [dBZ, Radial_Vel, IR_10.8, WV_6.8, Lightning_KDE, CAPE, CIN, Wind_Mag]
    Target sequence:
        Y: [T_out=6, C_target=2, H=128, W=128]
        Channels: [dBZ (reflectivity), Lightning_Density]
    """

    def __init__(
        self,
        inputs: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        h5_path: Optional[Union[str, Path]] = None,
        config: PipelineConfig = DEFAULT_CONFIG,
        transform: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.transform = transform
        self.t_in = config.t_in
        self.t_out = config.t_out
        self.c_in = config.num_channels
        self.c_out = config.num_target_channels
        self.h = config.grid_height
        self.w = config.grid_width

        self.inputs = inputs
        self.targets = targets
        self.h5_path = Path(h5_path) if h5_path else None

        if self.h5_path and self.h5_path.exists():
            self._init_from_h5(self.h5_path)
        elif self.inputs is not None and self.targets is not None:
            self._validate_shapes(self.inputs, self.targets)
        elif self.inputs is None and self.targets is None:
            # Empty dataset placeholder
            self.inputs = np.empty((0, self.t_in, self.c_in, self.h, self.w), dtype=np.float32)
            self.targets = np.empty((0, self.t_out, self.c_out, self.h, self.w), dtype=np.float32)

    def _validate_shapes(self, x: np.ndarray, y: np.ndarray) -> None:
        """Validates exact tensor dimensional constraints."""
        n_samples = x.shape[0]
        if y.shape[0] != n_samples:
            raise ValueError(f"Sample count mismatch: inputs {n_samples} vs targets {y.shape[0]}")

        expected_x = (self.t_in, self.c_in, self.h, self.w)
        if x.shape[1:] != expected_x:
            raise ValueError(f"Invalid input tensor shape {x.shape}. Expected [N, {expected_x}]")

        expected_y = (self.t_out, self.c_out, self.h, self.w)
        if y.shape[1:] != expected_y:
            raise ValueError(f"Invalid target tensor shape {y.shape}. Expected [N, {expected_y}]")

    def _init_from_h5(self, path: Path) -> None:
        """Loads dataset from an HDF5 archive file."""
        import h5py
        with h5py.File(path, "r") as h5:
            self.inputs = np.array(h5["inputs"], dtype=np.float32)
            self.targets = np.array(h5["targets"], dtype=np.float32)
        self._validate_shapes(self.inputs, self.targets)
        logger.info("Loaded %d sequence samples from %s", len(self), path)

    def save_to_h5(self, path: Union[str, Path]) -> None:
        """Persists dataset to standardized HDF5 archive file."""
        if self.inputs is None or self.targets is None:
            raise ValueError("Cannot save empty dataset to HDF5.")
        import h5py
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as h5:
            h5.create_dataset("inputs", data=self.inputs, compression="gzip")
            h5.create_dataset("targets", data=self.targets, compression="gzip")
            h5.attrs["channel_order"] = ",".join(CHANNEL_NAMES)
            h5.attrs["target_channel_order"] = ",".join(TARGET_CHANNEL_NAMES)
            h5.attrs["t_in"] = self.t_in
            h5.attrs["t_out"] = self.t_out
            h5.attrs["grid_size"] = f"{self.h}x{self.w}"
        logger.info("Saved %d samples to %s", len(self), out_path)

    def __len__(self) -> int:
        return self.inputs.shape[0] if self.inputs is not None else 0

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, Any], Union[np.ndarray, Any]]:
        """Returns single sample pair: (x_seq, y_seq).

        Returns PyTorch Tensors if torch is installed, else NumPy arrays.
        """
        if self.inputs is None or self.targets is None:
            raise IndexError("Dataset is empty.")

        x = self.inputs[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        if HAS_TORCH:
            import torch
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        return x.astype(np.float32), y.astype(np.float32)

    @classmethod
    def generate_synthetic(
        cls,
        samples: int = 4,
        config: PipelineConfig = DEFAULT_CONFIG,
        random_seed: int = 42,
    ) -> WeatherTensorDataset:
        """Generates realistic synthetic multi-timestep convective storm sequences.

        Models moving thunderstorm cells across 10 continuous 15-minute steps
        (4 historical frames + 6 forecast frames = 10 frames total).
        """
        rng = np.random.RandomState(random_seed)
        t_total = config.t_in + config.t_out  # 4 + 6 = 10 frames
        h, w = config.grid_height, config.grid_width

        all_inputs = np.empty((samples, config.t_in, config.num_channels, h, w), dtype=np.float32)
        all_targets = np.empty((samples, config.t_out, config.num_target_channels, h, w), dtype=np.float32)

        for s in range(samples):
            # Storm track parameters: start position and velocity vector (dx, dy)
            start_x = rng.uniform(20.0, 50.0)
            start_y = rng.uniform(30.0, 60.0)
            vel_x = rng.uniform(3.0, 7.0)  # ~5 km/15min cell propagation
            vel_y = rng.uniform(2.0, 5.0)

            # Generate full 10-step sequence
            seq_10 = np.empty((t_total, config.num_channels, h, w), dtype=np.float32)
            y_grid, x_grid = np.mgrid[0:h, 0:w]

            for t in range(t_total):
                cx = start_x + vel_x * t
                cy = start_y + vel_y * t
                dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2

                # Convective core mask
                storm_mask = np.exp(-dist_sq / (2.0 * (12.0 ** 2)))

                # Channel 0: dBZ (normalized [0, 1] from [-10, 70])
                # Core peaks at ~58 dBZ -> (58 - (-10)) / 80 = 68/80 = 0.85
                raw_dbz = -10.0 + storm_mask * 68.0
                norm_dbz = np.clip((raw_dbz - (-10.0)) / 80.0, 0.0, 1.0)

                # Channel 1: Radial Velocity (normalized [0, 1] from [-50, 50])
                # Couplet with 0.5 as zero velocity
                dx = (x_grid - cx) / 12.0
                dy = (y_grid - cy) / 12.0
                v_rot = (-dy * storm_mask) * 25.0
                norm_vel = np.clip((v_rot - (-50.0)) / 100.0, 0.0, 1.0)

                # Channel 2: IR_10.8 (cold anvil ~ 195 K -> (195 - 180)/(320 - 180) = 15/140 ~ 0.10)
                raw_ir = 300.0 - storm_mask * 105.0
                norm_ir = np.clip((raw_ir - 180.0) / 140.0, 0.0, 1.0)

                # Channel 3: WV_6.8 (cold core ~ 205 K -> (205 - 180)/(300 - 180) = 25/120 ~ 0.20)
                raw_wv = 270.0 - storm_mask * 65.0
                norm_wv = np.clip((raw_wv - 180.0) / 120.0, 0.0, 1.0)

                # Channel 4: Lightning_KDE [0, 1]
                norm_lightning = np.clip(storm_mask ** 1.5, 0.0, 1.0)

                # Channel 5: CAPE (plume ~ 3200 J/kg -> 3200/5000 = 0.64)
                raw_cape = 1000.0 + storm_mask * 2200.0
                norm_cape = np.clip(raw_cape / 5000.0, 0.0, 1.0)

                # Channel 6: CIN (10-120 J/kg -> 120/500 = 0.24)
                raw_cin = np.clip(120.0 - storm_mask * 100.0, 10.0, 300.0)
                norm_cin = np.clip(raw_cin / 500.0, 0.0, 1.0)

                # Channel 7: Wind_Mag (8-22 m/s -> 22/50 = 0.44)
                raw_wind = 8.0 + storm_mask * 14.0
                norm_wind = np.clip(raw_wind / 50.0, 0.0, 1.0)

                seq_10[t, 0] = norm_dbz
                seq_10[t, 1] = norm_vel
                seq_10[t, 2] = norm_ir
                seq_10[t, 3] = norm_wv
                seq_10[t, 4] = norm_lightning
                seq_10[t, 5] = norm_cape
                seq_10[t, 6] = norm_cin
                seq_10[t, 7] = norm_wind

            # Historical inputs: [0, 1, 2, 3] (t-45m to t+0m)
            all_inputs[s] = seq_10[:config.t_in]

            # Forecast targets: [4, 5, 6, 7, 8, 9] (t+15m to t+180m)
            # Target Channel 0: Reflectivity (norm_dbz)
            # Target Channel 1: Lightning Density (norm_lightning)
            future_targets = seq_10[config.t_in:, [0, 4]]
            all_targets[s] = future_targets

        return cls(inputs=all_inputs, targets=all_targets, config=config)


def create_dataloader(
    dataset: WeatherTensorDataset,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Any:
    """Creates a PyTorch DataLoader yielding batches of [B, T_in, C, H, W] tensors.

    Default batch_size is 2, optimized for RTX 3050 4GB VRAM limit.
    """
    bs = batch_size or dataset.config.batch_size
    if HAS_TORCH:
        import torch
        from torch.utils.data import DataLoader
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=False,
        )
    else:
        # Simple iterator fallback if torch is not yet imported
        class SimpleBatchIterator:
            def __init__(self, ds: WeatherTensorDataset, b_size: int, shuf: bool) -> None:
                self.ds = ds
                self.b_size = b_size
                self.indices = list(range(len(ds)))
                if shuf:
                    np.random.shuffle(self.indices)

            def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
                assert self.ds.inputs is not None and self.ds.targets is not None
                for i in range(0, len(self.indices), self.b_size):
                    batch_idx = self.indices[i:i + self.b_size]
                    batch_x = np.stack([self.ds.inputs[idx] for idx in batch_idx])
                    batch_y = np.stack([self.ds.targets[idx] for idx in batch_idx])
                    yield batch_x, batch_y

        return SimpleBatchIterator(dataset, bs, shuffle)
