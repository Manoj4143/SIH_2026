"""Unit tests for meteorological verification metrics (CSI, POD, FAR, SSIM)."""

import numpy as np
import pytest
import torch

from nowcast_engine.metrics.verification_metrics import MeteorologicalMetrics


class TestMeteorologicalMetrics:
    def test_unit_conversions(self):
        # -10 dBZ maps to 0.0
        assert np.isclose(MeteorologicalMetrics.dbz_to_norm(-10.0), 0.0)
        # 70 dBZ maps to 1.0
        assert np.isclose(MeteorologicalMetrics.dbz_to_norm(70.0), 1.0)
        # 30 dBZ maps to 0.5
        assert np.isclose(MeteorologicalMetrics.dbz_to_norm(30.0), 0.5)
        # Round trip
        test_dbz = np.array([20.0, 35.0, 45.0])
        norm = MeteorologicalMetrics.dbz_to_norm(test_dbz)
        restored = MeteorologicalMetrics.norm_to_dbz(norm)
        assert np.allclose(test_dbz, restored)

    def test_csi_corner_cases(self):
        # 1. Perfect prediction (100% hits, 0 misses, 0 false alarms)
        y_true = np.zeros((100, 100))
        y_true[30:70, 30:70] = 40.0  # > 35 dBZ storm cell
        y_pred = y_true.copy()

        csi_perfect = MeteorologicalMetrics.compute_csi(y_pred, y_true, threshold_dbz=35.0)
        pod_perfect = MeteorologicalMetrics.compute_pod(y_pred, y_true, threshold_dbz=35.0)
        far_perfect = MeteorologicalMetrics.compute_far(y_pred, y_true, threshold_dbz=35.0)

        assert csi_perfect == 1.0
        assert pod_perfect == 1.0
        assert far_perfect == 0.0

        # 2. Complete miss (0 hits, all misses)
        y_pred_miss = np.zeros_like(y_true)
        csi_miss = MeteorologicalMetrics.compute_csi(y_pred_miss, y_true, threshold_dbz=35.0)
        pod_miss = MeteorologicalMetrics.compute_pod(y_pred_miss, y_true, threshold_dbz=35.0)
        assert csi_miss == 0.0
        assert pod_miss == 0.0

        # 3. All false alarms (storm predicted where none occurred)
        y_pred_fa = np.zeros((100, 100))
        y_true_clear = np.zeros((100, 100))
        y_pred_fa[20:40, 20:40] = 50.0

        csi_fa = MeteorologicalMetrics.compute_csi(y_pred_fa, y_true_clear, threshold_dbz=35.0)
        far_fa = MeteorologicalMetrics.compute_far(y_pred_fa, y_true_clear, threshold_dbz=35.0)
        assert csi_fa == 0.0
        assert far_fa == 1.0

        # 4. Clear-air agreement (both target and pred clear of storms)
        csi_clear = MeteorologicalMetrics.compute_csi(y_true_clear, y_true_clear, threshold_dbz=35.0)
        assert csi_clear == 1.0

    def test_ssim_computation(self):
        img1 = np.random.RandomState(42).uniform(0.0, 1.0, size=(128, 128))
        ssim_ident = MeteorologicalMetrics.compute_ssim(img1, img1)
        assert np.isclose(ssim_ident, 1.0, atol=1e-4)

        img2 = np.random.RandomState(99).uniform(0.0, 1.0, size=(128, 128))
        ssim_diff = MeteorologicalMetrics.compute_ssim(img1, img2)
        assert ssim_diff < 0.5

    def test_evaluate_forecast_sequence(self):
        pred = torch.rand(2, 6, 2, 128, 128)
        true = torch.rand(2, 6, 2, 128, 128)

        scorecard = MeteorologicalMetrics.evaluate_forecast_sequence(pred, true)
        assert "csi_20dBZ" in scorecard
        assert "csi_35dBZ" in scorecard
        assert "csi_45dBZ" in scorecard
        assert "pod_35dBZ" in scorecard
        assert "far_35dBZ" in scorecard
        assert "mean_ssim" in scorecard
