"""Automated 15-Minute Background Inference Worker Daemon.

Orchestrates live ingestion from data_pipeline, runs NowcastNet forward passes,
extracts severe storm alert polygons, and publishes prediction rasters for backend_api.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from data_pipeline.loaders.dataset import WeatherTensorDataset
from data_pipeline.loaders.pipeline import WeatherNowcastDataPipeline
from nowcast_engine.config import ModelConfig
from nowcast_engine.models.nowcast_net import NowcastNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [InferenceWorker]: %(message)s",
)
logger = logging.getLogger("InferenceWorker")


class InferenceWorker:
    """Automated operational nowcasting worker daemon."""

    def __init__(
        self,
        weights_path: str = "nowcast_engine/weights/best_model.pt",
        output_dir: str = "data/inferences",
        device: str = "",
        synthetic: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic = synthetic

        # Determine compute device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Target compute device: {self.device}")

        # Initialize data pipeline
        self.pipeline = WeatherNowcastDataPipeline()

        # Initialize Model
        self.model_config = ModelConfig()
        self.model = NowcastNet(self.model_config).to(self.device)
        self.model.eval()

        # Load weights if available
        w_path = Path(weights_path)
        if w_path.exists():
            try:
                checkpoint = torch.load(w_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    logger.info(f"Loaded trained checkpoint: {w_path} (epoch {checkpoint.get('epoch', '?')})")
                else:
                    self.model.load_state_dict(checkpoint)
                    logger.info(f"Loaded state dict: {w_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint ({e}). Running with initial weights.")
        else:
            logger.warning(f"No checkpoint found at {weights_path}. Running with initial weights.")

    def run_inference_cycle(self) -> dict:
        """Executes a single end-to-end inference step:

        Ingest -> Format [1, 4, 8, 128, 128] -> NowcastNet -> Export Rasters & Alerts.
        """
        start_time = time.perf_counter()
        now = datetime.datetime.utcnow()
        inference_id = f"INF-{int(now.timestamp())}"
        logger.info(f"--- Starting Nowcasting Inference Cycle ({inference_id}) ---")

        # 1. Fetch historical frames [T_in=4, C=8, H=128, W=128]
        if self.synthetic:
            # Use synthetic moving storm sequence
            dataset = WeatherTensorDataset.generate_synthetic(samples=1)
            x_raw, _ = dataset[0]  # [4, 8, 128, 128]
            x_tensor = x_raw if isinstance(x_raw, torch.Tensor) else torch.from_numpy(x_raw).float()
        else:
            # Assemble frames from data pipeline
            x_np = self.pipeline.get_historical_sequence_tensor()
            x_tensor = torch.from_numpy(x_np).float()
            logger.info("Assembled historical sequence tensor from data pipeline.")

        # Add batch dimension: [1, 4, 8, 128, 128]
        x_batch = x_tensor.unsqueeze(0).to(self.device)

        # 2. Run forward pass with torch.no_grad
        amp_enabled = self.device.type == "cuda"
        with torch.no_grad():
            if amp_enabled:
                with torch.amp.autocast(device_type="cuda"):
                    y_pred = self.model(x_batch)  # [1, 6, 2, 128, 128]
            else:
                y_pred = self.model(x_batch)

        # Extract predictions array [6, 2, 128, 128]
        preds_np = y_pred.squeeze(0).float().cpu().numpy().astype(np.float32)
        preds_np = np.clip(preds_np, 0.0, 1.0)

        # 3. Export rasters to shared storage
        npz_target = self.output_dir / "latest_nowcast.npz"
        np.savez_compressed(
            str(npz_target),
            predictions=preds_np,
            timestamp=now.isoformat(),
            inference_id=inference_id,
        )
        logger.info(f"Saved prediction rasters: {npz_target}")

        # 4. Generate Alert Polygons
        try:
            from backend_api.app.services.vector_exporter import vector_exporter
            all_features = []
            lead_times = [15, 30, 45, 60, 120, 180]
            for t_idx, lead_min in enumerate(lead_times):
                dbz_field = preds_np[t_idx, 0] * 80.0 - 10.0
                light_field = preds_np[t_idx, 1]
                step_features = vector_exporter.extract_alert_polygons_for_step(
                    radar_dbz_2d=dbz_field,
                    lightning_2d=light_field,
                    lead_time_minutes=lead_min,
                    base_timestamp=now,
                )
                all_features.extend(step_features)

            alert_collection = vector_exporter.build_feature_collection(all_features, now)
            alerts_file = self.output_dir / "latest_alerts.json"
            with open(alerts_file, "w", encoding="utf-8") as f:
                json.dump(alert_collection, f, indent=2)
            logger.info(f"Published {len(all_features)} hazard alerts to: {alerts_file}")
            alerts_count = len(all_features)
        except Exception as alert_err:
            logger.warning(f"Alert export error: {alert_err}")
            alerts_count = 0

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        peak_dbz = float(np.max(preds_np[:, 0])) * 80.0 - 10.0
        peak_light = float(np.max(preds_np[:, 1]))

        logger.info(
            f"Cycle completed in {elapsed_ms:.1f} ms | "
            f"Peak dBZ: {peak_dbz:.1f} | Peak Lightning: {peak_light:.3f} | Alerts: {alerts_count}"
        )

        return {
            "inference_id": inference_id,
            "timestamp": now.isoformat(),
            "latency_ms": elapsed_ms,
            "peak_dbz": peak_dbz,
            "peak_lightning": peak_light,
            "alerts_count": alerts_count,
        }

    def start_loop(self, interval_seconds: int = 900) -> None:
        """Runs the continuous daemon loop executing every interval_seconds (default: 15 min)."""
        logger.info(f"Starting automated inference loop (Interval: {interval_seconds} seconds)...")
        while True:
            try:
                self.run_inference_cycle()
            except Exception as e:
                logger.error(f"Error during scheduled inference cycle: {e}", exc_info=True)

            logger.info(f"Sleeping for {interval_seconds} seconds until next scheduled nowcast...")
            time.sleep(interval_seconds)


def main() -> None:
    """CLI entrypoint for operational nowcasting inference worker."""
    parser = argparse.ArgumentParser(description="Automated 15-minute operational nowcasting worker")
    parser.add_argument("--interval", type=int, default=900, help="Inference cadence in seconds (default: 900)")
    parser.add_argument("--once", action="store_true", help="Execute a single inference pass and exit")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic test ingestion")
    parser.add_argument("--weights", type=str, default="nowcast_engine/weights/best_model.pt", help="Model weights path")
    parser.add_argument("--device", type=str, default="", help="Compute device: cuda or cpu")
    args = parser.parse_args()

    worker = InferenceWorker(
        weights_path=args.weights,
        device=args.device,
        synthetic=args.synthetic,
    )

    if args.once:
        result = worker.run_inference_cycle()
        print(json.dumps(result, indent=2))
    else:
        worker.start_loop(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
