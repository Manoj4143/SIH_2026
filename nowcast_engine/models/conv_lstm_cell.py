"""Spatiotemporal ConvLSTM Cell and Bottleneck Module.

Implements 2D Convolutional LSTM recurrent cells that track thunderstorm
advection trajectories, convective cell deformation, and spatial movement
over time while preserving 2D spatial feature topology.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single 2D Convolutional LSTM Cell with fused convolution gates."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
        separable: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        in_c = self.input_dim + self.hidden_dim
        out_c = 4 * self.hidden_dim

        self.conv: nn.Module
        if separable:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=self.kernel_size, padding=self.padding, groups=in_c, bias=bias),
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=bias),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=bias,
            )

    def forward(
        self,
        x: torch.Tensor,
        cur_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single time step.

        Args:
            x: Input tensor of shape [B, input_dim, H, W]
            cur_state: Tuple of (h_cur, c_cur) each of shape [B, hidden_dim, H, W]

        Returns:
            Tuple of (h_next, c_next) each of shape [B, hidden_dim, H, W]
        """
        b, _, h, w = x.shape

        if cur_state is None:
            h_cur = torch.zeros(b, self.hidden_dim, h, w, device=x.device, dtype=x.dtype)
            c_cur = torch.zeros(b, self.hidden_dim, h, w, device=x.device, dtype=x.dtype)
        else:
            h_cur, c_cur = cur_state

        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMLayer(nn.Module):
    """Processes an entire temporal sequence through a ConvLSTMCell."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        return_all_layers: bool = False,
        separable: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size=kernel_size, separable=separable)
        self.return_all_layers = return_all_layers

    def forward(
        self,
        x_seq: torch.Tensor,
        init_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Processes sequence [B, T, C, H, W].

        Returns:
            output_seq: [B, T, hidden_dim, H, W]
            last_state: (h_last, c_last)
        """
        b, t, _, h, w = x_seq.shape
        state = init_state
        output_inner = []

        for step in range(t):
            x_t = x_seq[:, step]
            h, c = self.cell(x_t, state)
            state = (h, c)
            output_inner.append(h)

        output_seq = torch.stack(output_inner, dim=1)
        assert state is not None
        return output_seq, state


class ConvLSTMBottleneck(nn.Module):
    """2-Layer Stacked ConvLSTM Bottleneck for deep spatiotemporal trajectory tracking."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        kernel_size: int = 3,
        separable: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            cur_input_dim = input_dim if layer_idx == 0 else hidden_dim
            self.layers.append(
                ConvLSTMLayer(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    separable=separable,
                )
            )

    def forward(
        self,
        x_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through stacked ConvLSTM layers.

        Args:
            x_seq: Input tensor of shape [B, T, C, H, W]

        Returns:
            Tuple of:
            - output_seq: [B, T, hidden_dim, H, W]
            - states: List of (h, c) for each layer at final timestep
        """
        cur_input = x_seq
        states = []

        for layer in self.layers:
            output_seq, state = layer(cur_input)
            states.append(state)
            # Add residual skip connection if dimensions match
            if cur_input.shape == output_seq.shape:
                cur_input = cur_input + output_seq
            else:
                cur_input = output_seq

        return cur_input, states
