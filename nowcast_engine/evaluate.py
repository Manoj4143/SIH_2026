"""Standalone Evaluation and Verification Scorecard Script for NowcastNet.

Usage:
    python nowcast_engine/evaluate.py --checkpoint nowcast_engine/weights/best_model.pt
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
from nowcast_engine.config import ModelConfig
from nowcast_engine.metrics.verification_metrics import MeteorologicalMetrics
from nowcast_engine.models.nowcast_net import NowcastNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SIH 26072 NowcastNet Model")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to .pt checkpoint file")
    parser.add_argument("--samples", type=int, default=4, help="Number of test evaluation sequences")
    parser.add_argument("--batch-size", type=int, default=2, help="Evaluation batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("SIH 26072 Nowcasting System - Meteorological Verification Scorecard")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = NowcastNet(ModelConfig()).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  -> Checkpoint trained at Epoch {ckpt.get('epoch', 'N/A')}")
    else:
        print("No checkpoint provided or file missing. Evaluating in demonstration inference mode.")

    model.eval()

    # Generate test dataset
    print(f"\nGenerating {args.samples} test spatiotemporal sequences...")
    test_ds = WeatherTensorDataset.generate_synthetic(samples=args.samples, random_seed=999)
    loader = create_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            preds = model(bx)
            all_preds.append(preds.cpu())
            all_targets.append(by)

    cat_preds = torch.cat(all_preds, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)

    print("\n" + "-" * 70)
    print("OVERALL METEOROLOGICAL PERFORMANCE SCORECARD")
    print("-" * 70)
    metrics = MeteorologicalMetrics.evaluate_forecast_sequence(cat_preds, cat_targets)

    print(f"{'Metric':<25} | {'20 dBZ':<10} | {'35 dBZ (Severe)':<15} | {'45 dBZ (Extreme)':<15}")
    print("-" * 70)
    print(f"{'Critical Success Index (CSI)':<25} | {metrics.get('csi_20dBZ', 0.0):<10.4f} | "
          f"{metrics.get('csi_35dBZ', 0.0):<15.4f} | {metrics.get('csi_45dBZ', 0.0):<15.4f}")
    print(f"{'Probability of Detection (POD)':<25} | {metrics.get('pod_20dBZ', 0.0):<10.4f} | "
          f"{metrics.get('pod_35dBZ', 0.0):<15.4f} | {metrics.get('pod_45dBZ', 0.0):<15.4f}")
    print(f"{'False Alarm Ratio (FAR)':<25} | {metrics.get('far_20dBZ', 0.0):<10.4f} | "
          f"{metrics.get('far_35dBZ', 0.0):<15.4f} | {metrics.get('far_45dBZ', 0.0):<15.4f}")
    print("-" * 70)
    print(f"Mean Structural Similarity (SSIM): {metrics.get('mean_ssim', 0.0):.4f}")

    # Per-timestep CSI breakdown (t+15m to t+180m)
    lead_times = ["t+15m", "t+30m", "t+45m", "t+60m", "t+120m", "t+180m"]
    print("\n" + "-" * 70)
    print("LEAD-TIME BREAKDOWN (CSI @ 35 dBZ Convective Hazard)")
    print("-" * 70)
    for t_idx in range(min(6, cat_preds.shape[1])):
        lead = lead_times[t_idx] if t_idx < len(lead_times) else f"Step {t_idx + 1}"
        csi_t = MeteorologicalMetrics.compute_csi(
            cat_preds[:, t_idx, 0].numpy(),
            cat_targets[:, t_idx, 0].numpy(),
            threshold_dbz=35.0,
        )
        pod_t = MeteorologicalMetrics.compute_pod(
            cat_preds[:, t_idx, 0].numpy(),
            cat_targets[:, t_idx, 0].numpy(),
            threshold_dbz=35.0,
        )
        far_t = MeteorologicalMetrics.compute_far(
            cat_preds[:, t_idx, 0].numpy(),
            cat_targets[:, t_idx, 0].numpy(),
            threshold_dbz=35.0,
        )
        print(f"  Horizon {lead:<8}: CSI={csi_t:.4f} | POD={pod_t:.4f} | FAR={far_t:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
