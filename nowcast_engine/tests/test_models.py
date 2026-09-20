"""Unit tests for NowcastNet architecture and spatiotemporal model components."""

import torch
import pytest

from nowcast_engine.config import ModelConfig
from nowcast_engine.models.conv_lstm_cell import ConvLSTMCell, ConvLSTMBottleneck
from nowcast_engine.models.encoder_decoder import (
    DepthwiseSeparableConv2d,
    DualOutputHead,
    SkipConnectionDecoder,
    SpatialEncoder,
)
from nowcast_engine.models.nowcast_net import NowcastNet


@pytest.fixture
def model_config():
    return ModelConfig(
        t_in=4,
        t_out=6,
        in_channels=8,
        out_channels=2,
        img_height=128,
        img_width=128,
    )


class TestModelComponents:
    def test_depthwise_separable_conv2d(self):
        conv = DepthwiseSeparableConv2d(in_channels=8, out_channels=32)
        x = torch.randn(2, 8, 128, 128)
        out = conv(x)
        assert out.shape == (2, 32, 128, 128)

    def test_spatial_encoder_scales(self):
        encoder = SpatialEncoder(in_channels=8, dims=(32, 64, 128))
        x = torch.randn(2, 8, 128, 128)
        bottleneck, skips = encoder(x)

        # Bottleneck downsampled to 32x32
        assert bottleneck.shape == (2, 128, 32, 32)
        # Skip connections
        assert skips[0].shape == (2, 32, 128, 128)
        assert skips[1].shape == (2, 64, 64, 64)

    def test_conv_lstm_cell(self):
        cell = ConvLSTMCell(input_dim=128, hidden_dim=128, kernel_size=3)
        x = torch.randn(2, 128, 32, 32)
        h, c = cell(x)
        assert h.shape == (2, 128, 32, 32)
        assert c.shape == (2, 128, 32, 32)

    def test_conv_lstm_bottleneck(self):
        bottleneck = ConvLSTMBottleneck(input_dim=128, hidden_dim=128, num_layers=2)
        # Sequence of 4 frames at 32x32
        seq = torch.randn(2, 4, 128, 32, 32)
        out_seq, states = bottleneck(seq)

        assert out_seq.shape == (2, 4, 128, 32, 32)
        assert len(states) == 2

    def test_skip_decoder_and_dual_heads(self):
        decoder = SkipConnectionDecoder(bottleneck_dim=128, decoder_dims=(128, 64, 32))
        head = DualOutputHead(in_channels=32)

        bottleneck = torch.randn(2, 128, 32, 32)
        skips = [torch.randn(2, 32, 128, 128), torch.randn(2, 64, 64, 64)]

        features = decoder(bottleneck, skips)
        assert features.shape == (2, 32, 128, 128)

        dual_preds = head(features)
        assert dual_preds.shape == (2, 2, 128, 128)
        assert torch.all(dual_preds >= 0.0) and torch.all(dual_preds <= 1.0)


class TestNowcastNetMaster:
    def test_full_model_forward_dimensions(self, model_config):
        model = NowcastNet(model_config)
        # Input 5D Tensor: [B=2, T_in=4, C=8, H=128, W=128]
        inputs = torch.randn(2, 4, 8, 128, 128)
        outputs = model(inputs)

        # Output 5D Tensor: [B=2, T_out=6, C_target=2, H=128, W=128]
        assert outputs.shape == (2, 6, 2, 128, 128)
        # Values must be bounded in [0.0, 1.0]
        assert torch.all(outputs >= 0.0)
        assert torch.all(outputs <= 1.0)

    def test_parameter_count_and_vram_limit(self, model_config):
        model = NowcastNet(model_config)
        params = model.get_parameter_count()
        total_p = params["total_parameters"]

        # Low-VRAM policy: Must stay under 3.5M parameters
        assert total_p < 3_500_000, f"Parameter count {total_p} exceeds 3.5M limit"
        print(f"Verified NowcastNet parameter count: {total_p:,}")
