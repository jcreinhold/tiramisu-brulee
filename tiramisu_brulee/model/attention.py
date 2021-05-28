#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.model.attention

grid attention blocks for gated attention networks

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jul 12, 2020
"""

__all__ = [
    "GridAttentionBlock2d",
    "GridAttentionBlock3d",
    "AttentionTiramisu2d",
    "AttentionTiramisu3d",
]

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from tiramisu_brulee.model.dense import (
    ACTIVATION,
    Bottleneck2d,
    Bottleneck3d,
    DenseBlock2d,
    DenseBlock3d,
    TransitionDown2d,
    TransitionDown3d,
    TransitionUp2d,
    TransitionUp3d,
)


class GridAttentionBlock(nn.Module):
    _conv = None
    _norm = None
    _upsample = None

    def __init__(
        self,
        in_channels: int,
        gating_channels: int,
        inter_channels: Optional[int] = None,
    ):
        super().__init__()
        if inter_channels is None:
            inter_channels = in_channels

        self.W = nn.Sequential(
            self._conv(in_channels, in_channels, 1),
            self._norm(in_channels),
            ACTIVATION(),
        )

        self.theta = self._conv(in_channels, inter_channels, 2, stride=2, bias=False)
        self.phi = self._conv(gating_channels, inter_channels, 1)
        self.psi = self._conv(inter_channels, 1, 1)

    def _interp(self, x: Tensor, size: List[int]) -> Tensor:
        return F.interpolate(x, size=size, mode=self._upsample, align_corners=True)

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        input_size = x.shape[2:]

        theta_x = self.theta(x)
        theta_x_size = theta_x.shape[2:]

        phi_g = self.phi(g)
        phi_g = self._interp(phi_g, theta_x_size)
        theta_phi_sum = theta_x + phi_g
        f = F.relu(theta_phi_sum, inplace=True)

        psi_f = self.psi(f)
        psi_f = torch.sigmoid(psi_f)
        psi_f = self._interp(psi_f, input_size)

        y = psi_f * x
        W_y = self.W(y)
        return W_y


class GridAttentionBlock3d(GridAttentionBlock):
    _conv = nn.Conv3d
    _norm = nn.BatchNorm3d
    _upsample = "trilinear"


class GridAttentionBlock2d(GridAttentionBlock):
    _conv = nn.Conv2d
    _norm = nn.BatchNorm2d
    _upsample = "bilinear"


class AttentionTiramisu(nn.Module):
    _attention = None
    _bottleneck = None
    _conv = None
    _denseblock = None
    _pad = None
    _trans_down = None
    _trans_up = None
    _upsample = None

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        down_blocks: List[int] = (5, 5, 5, 5, 5),
        up_blocks: List[int] = (5, 5, 5, 5, 5),
        bottleneck_layers: int = 5,
        growth_rate: int = 16,
        first_conv_out_channels: int = 48,
        dropout_rate: float = 0.2,
    ):
        """Base class for Tiramisu convolutional neural network with attention

        See Also:
            Schlemper, Jo, et al. "Attention gated networks: Learning to leverage
            salient regions in medical images." Medical image analysis 53 (2019):
            197-207.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            down_blocks (List[int]): number of layers in each block in down path
            up_blocks (List[int]): number of layers in each block in up path
            bottleneck_layers (int): number of layers in the bottleneck
            growth_rate (int): number of channels to grow by in each layer
            first_conv_out_channels (int): number of output channels in first conv
            dropout_rate (float): dropout rate/probability
        """
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        first_kernel_size = 3
        final_kernel_size = 1
        skip_connection_channel_counts = []

        self.first_conv = nn.Sequential(
            self._pad(first_kernel_size // 2),
            self._conv(
                in_channels, first_conv_out_channels, first_kernel_size, bias=False,
            ),
        )
        cur_channels_count = first_conv_out_channels

        # Downsampling path
        self.dense_down = nn.ModuleList([])
        self.trans_down = nn.ModuleList([])
        for n_layers in down_blocks:
            block = self._denseblock(
                cur_channels_count,
                growth_rate,
                n_layers,
                upsample=False,
                dropout_rate=dropout_rate,
            )
            self.dense_down.append(block)
            cur_channels_count += growth_rate * n_layers
            skip_connection_channel_counts.insert(0, cur_channels_count)
            block = self._trans_down(
                cur_channels_count, cur_channels_count, dropout_rate=dropout_rate,
            )
            self.trans_down.append(block)

        # Bottleneck
        self.bottleneck = self._bottleneck(
            cur_channels_count,
            growth_rate,
            bottleneck_layers,
            dropout_rate=dropout_rate,
        )
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        # Upsampling path
        self.trans_up = nn.ModuleList([])
        self.dense_up = nn.ModuleList([])
        self.attention_gates = nn.ModuleList([])
        self.deep_supervision = nn.ModuleList([])
        up_info = zip(up_blocks, skip_connection_channel_counts)
        for i, (n_layers, sccc) in enumerate(up_info, 1):
            block = self._attention(sccc, prev_block_channels)
            self.attention_gates.append(block)
            block = self._trans_up(prev_block_channels, prev_block_channels)
            self.trans_up.append(block)
            cur_channels_count = prev_block_channels + sccc
            not_last_block = i < len(up_blocks)  # don't upsample on last blk
            block = self._denseblock(
                cur_channels_count,
                growth_rate,
                n_layers,
                upsample=not_last_block,
                dropout_rate=dropout_rate,
            )
            self.dense_up.append(block)
            prev_block_channels = growth_rate * n_layers
            cur_channels_count += prev_block_channels
            if not_last_block:
                dsv_channel_count = prev_block_channels
            else:
                dsv_channel_count = cur_channels_count
            self.deep_supervision.append(
                self._conv(dsv_channel_count, out_channels, final_kernel_size)
            )

    @property
    def _down_blocks(self):
        return zip(self.dense_down, self.trans_down)

    @property
    def _up_blocks(self):
        return zip(
            self.dense_up, self.trans_up, self.attention_gates, self.deep_supervision
        )

    def _interp(self, tensor: Tensor, size: Tuple[int]) -> Tensor:
        return F.interpolate(tensor, size, mode=self._upsample, align_corners=True)

    def forward(self, tensor: Tensor) -> List[Tensor]:
        input_size = tensor.shape[2:]
        out = self.first_conv(tensor)
        skip_connections = []
        for dbd, tdb in self._down_blocks:
            out = dbd(out)
            skip_connections.append(out)
            out = tdb(out)
        out = self.bottleneck(out)
        dsvs = []
        for ubd, tub, atg, dsl in self._up_blocks:
            skip = skip_connections.pop()
            skip = atg(skip, out)
            out = tub(out, skip)
            out = ubd(out)
            dsv = dsl(out)
            dsv = self._interp(dsv, input_size)
            dsvs.append(dsv)
        return dsvs

    def predict(self, tensor: Tensor) -> Tensor:
        out = self.first_conv(tensor)
        skip_connections = []
        for dbd, tdb in self._down_blocks:
            out = dbd(out)
            skip_connections.append(out)
            out = tdb(out)
        out = self.bottleneck(out)
        for ubd, tub, atg, _ in self._up_blocks:
            skip = skip_connections.pop()
            skip = atg(skip, out)
            out = tub(out, skip)
            out = ubd(out)
        out = self.deep_supervision[-1](out)
        return out


class AttentionTiramisu2d(AttentionTiramisu):
    _attention = GridAttentionBlock2d
    _bottleneck = Bottleneck2d
    _conv = nn.Conv2d
    _denseblock = DenseBlock2d
    _pad = nn.ReplicationPad2d
    _trans_down = TransitionDown2d
    _trans_up = TransitionUp2d
    _upsample = "bilinear"


class AttentionTiramisu3d(AttentionTiramisu):
    _attention = GridAttentionBlock3d
    _bottleneck = Bottleneck3d
    _conv = nn.Conv3d
    _denseblock = DenseBlock3d
    _pad = nn.ReplicationPad3d
    _trans_down = TransitionDown3d
    _trans_up = TransitionUp3d
    _upsample = "trilinear"
