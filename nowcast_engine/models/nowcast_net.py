"""Master NowcastNet Spatiotemporal Architecture.

Fuses a Depthwise Separable Conv2D Spatial Encoder, a 2-Layer ConvLSTM Bottleneck,
and a Skip-Connected U-Net Decoder with Dual Output Heads.
Processes [B, 4, 8, 128, 128] inputs and emits [B, 6, 2, 128, 128] predictions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from nowcast_engine.config import DEFAULT_MODEL_CONFIG, ModelConfig
from nowcast_engine.models.conv_lstm_cell import ConvLSTMCell, ConvLSTMBottleneck
from nowcast_engine.models.encoder_decoder import DualOutputHead, SkipConnectionDecoder, SpatialEncoder


class NowcastNet(nn.Module):
    """Spatiotemporal Sequence-to-Sequence Deep Learning Model for Convective Nowcasting.

    Input tensor shape:
        [B, T_in=4, C=8, H=128, W=128]
    Output tensor shape:
        [B, T_out=6, C_target=2, H=128, W=128]
        (Ch 0: Radar Reflectivity factor in [0, 1], Ch 1: Total Lightning Probability in [0, 1])
    """

    def __init__(self, config: ModelConfig = DEFAULT_MODEL_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.t_in = config.t_in
        self.t_out = config.t_out
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.hidden_dim = config.conv_lstm_hidden_dim

        # 1. Spatial Encoder (Depthwise Separable Conv2D)
        self.encoder = SpatialEncoder(
            in_channels=self.in_channels,
            dims=config.encoder_dims,
        )

        # 2. Spatiotemporal Bottleneck (2-Layer ConvLSTM)
        self.bottleneck = ConvLSTMBottleneck(
            input_dim=config.encoder_dims[-1],
            hidden_dim=self.hidden_dim,
            num_layers=config.conv_lstm_num_layers,
            kernel_size=config.conv_lstm_kernel_size,
            separable=config.use_separable_convs,
        )

        # 3. Future Forecasting Recurrent Cell
        self.forecast_cell = ConvLSTMCell(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=config.conv_lstm_kernel_size,
            separable=config.use_separable_convs,
        )

        # 4. Skip-Connected Decoder
        self.decoder = SkipConnectionDecoder(
            bottleneck_dim=self.hidden_dim,
            decoder_dims=config.decoder_dims,
        )

        # 5. Dual Hazard Output Heads (Reflectivity & Lightning)
        self.output_head = DualOutputHead(in_channels=config.decoder_dims[-1])

    def forward(
        self,
        x_seq: torch.Tensor,
        future_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward execution through encoder, ConvLSTM bottleneck, and dual-head decoder.

        Args:
            x_seq: Input sequence tensor of shape [B, T_in=4, C=8, H=128, W=128]
            future_steps: Number of forecast horizons (default: config.t_out = 6)

        Returns:
            Forecast tensor of shape [B, T_out=6, C_target=2, H=128, W=128]
        """
        b, t_in, c, h, w = x_seq.shape
        steps_out = future_steps or self.t_out

        # Step 1: Spatial Encoding across all historical timesteps
        encoded_latents = []
        last_skips = None

        for step in range(t_in):
            x_t = x_seq[:, step]  # [B, 8, 128, 128]
            latent_t, skips_t = self.encoder(x_t)  # latent_t: [B, 128, 32, 32]
            encoded_latents.append(latent_t)
            if step == t_in - 1:
                last_skips = skips_t

        # Latent historical sequence: [B, T_in=4, 128, 32, 32]
        latent_seq = torch.stack(encoded_latents, dim=1)

        # Step 2: Spatiotemporal Bottleneck processing
        # output_seq: [B, 4, 128, 32, 32], states: List of (h, c) for each layer
        output_seq, states = self.bottleneck(latent_seq)

        # Take last layer's final state for autoregressive forecasting
        h_t, c_t = states[-1]

        # Step 3: Autoregressive Future Forecasting Horizons
        forecasts = []
        cur_h, cur_c = h_t, c_t

        for _ in range(steps_out):
            # Recurrent forecast step
            cur_h, cur_c = self.forecast_cell(cur_h, (cur_h, cur_c))

            # Step 4 & 5: Spatial Decoding + Dual Head Prediction
            assert last_skips is not None
            decoded_features = self.decoder(cur_h, last_skips)  # [B, 32, 128, 128]
            pred_dual = self.output_head(decoded_features)       # [B, 2, 128, 128]
            forecasts.append(pred_dual)

        # Concatenate into 5D forecast tensor [B, T_out=6, 2, 128, 128]
        out_tensor = torch.stack(forecasts, dim=1)
        return out_tensor

    def get_parameter_count(self) -> Dict[str, int]:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}
