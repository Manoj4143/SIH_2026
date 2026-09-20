"""Standalone Training Entrypoint for NowcastNet.

Usage:
    python nowcast_engine/train.py --epochs 5 --batch-size 2 --samples 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from data_pipeline.loaders.dataset import WeatherTensorDataset, create_dataloader
from nowcast_engine.config import ModelConfig, TrainingConfig
from nowcast_engine.models.nowcast_net import NowcastNet
from nowcast_engine.training.trainer import NowcastTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SIH 26072 NowcastNet Model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2 for RTX 3050)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--samples", type=int, default=8, help="Number of synthetic sequence samples")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu (auto-detected if empty)")
    parser.add_argument("--weights-dir", type=str, default="nowcast_engine/weights", help="Checkpoint directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("SIH 26072 Nowcasting System - Phase 2 Model Training Engine")
    print("=" * 70)

    device_str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Target Compute Device: {device} (VRAM optimization active: True)")

    # Model & Training configs
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.weights_dir,
    )

    print("\n[1/4] Initializing NowcastNet Architecture...")
    model = NowcastNet(model_cfg)
    params = model.get_parameter_count()
    print(f"  -> Total Parameters:     {params['total_parameters']:,}")
    print(f"  -> Trainable Parameters: {params['trainable_parameters']:,}")
    print(f"  -> Memory budget target: < 2.48 GB VRAM on RTX 3050")

    print(f"\n[2/4] Generating Training and Validation Sequences (N={args.samples})...")
    train_ds = WeatherTensorDataset.generate_synthetic(samples=args.samples, random_seed=42)
    val_ds = WeatherTensorDataset.generate_synthetic(samples=max(2, args.samples // 2), random_seed=1337)

    train_loader = create_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"  -> Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("\n[3/4] Starting Spatiotemporal Training Loop...")
    trainer = NowcastTrainer(
        model=model,
        model_config=model_cfg,
        train_config=train_cfg,
        device=device,
    )

    for epoch in range(1, args.epochs + 1):
        train_losses = trainer.train_epoch(train_loader)
        val_losses, val_metrics = trainer.validate(val_loader)
        trainer.scheduler.step()

        cur_lr = trainer.optimizer.param_groups[0]["lr"]
        csi_35 = val_metrics.get("csi_35dBZ", 0.0)

        print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] (LR: {cur_lr:.6f}):")
        print(f"    Train Loss: Total={train_losses['total_loss']:.4f}, WMSE={train_losses['wmse_loss']:.4f}, "
              f"Focal={train_losses['focal_loss']:.4f}, SSIM={train_losses['ssim_loss']:.4f}")
        print(f"    Val Loss:   Total={val_losses['total_loss']:.4f}, WMSE={val_losses['wmse_loss']:.4f}")
        print(f"    Metrics:    CSI@35dBZ={csi_35:.4f}, POD={val_metrics.get('pod_35dBZ', 0.0):.4f}, "
              f"FAR={val_metrics.get('far_35dBZ', 0.0):.4f}, SSIM={val_metrics.get('mean_ssim', 0.0):.4f}")

        # Checkpoint if best CSI
        if csi_35 > trainer.best_csi_35:
            trainer.best_csi_35 = csi_35
            ckpt = trainer.save_checkpoint(epoch, csi_35, val_losses, is_best=True)
            print(f"    *** New Best Model Saved: {ckpt.name} (CSI@35: {csi_35:.4f}) ***")

    print("\n[4/4] Finalizing Checkpoints...")
    final_path = trainer.save_checkpoint(args.epochs, trainer.best_csi_35, val_losses, is_best=False)
    print(f"  -> Final weights saved to: {final_path}")
    print("\n" + "=" * 70)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
