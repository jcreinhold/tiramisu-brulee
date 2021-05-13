#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.model.attention

grid attention blocks for gated attention networks

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 12, 2020
"""

__all__ = ['GridAttentionBlock2d',
           'GridAttentionBlock3d',
           'AttentionTiramisu2d',
           'AttentionTiramisu3d']

from typing import *

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from tiramisu_brulee.model.dense import *
from tiramisu_brulee.model.dense import ACTIVATION


class GridAttentionBlock(nn.Module):
    _conv = None
    _norm = None
    _upsample = None

    def __init__(self, in_channels: int, gating_channels: int, inter_channels: Optional[int] = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = in_channels

        self.W = nn.Sequential(
            self._conv(in_channels, in_channels, 1),
            self._norm(in_channels),
            ACTIVATION()
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

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1,
                 down_blocks: List[int] = (5, 5, 5, 5, 5),
                 up_blocks: List[int] = (5, 5, 5, 5, 5),
                 bottleneck_layers: int = 5,
                 growth_rate: int = 16,
                 out_chans_first_conv: int = 48,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        first_kernel_size = 3
        final_kernel_size = 1
        skip_connection_channel_counts = []

        self.first_conv = nn.Sequential(
            self._pad(first_kernel_size // 2),
            self._conv(in_channels, out_chans_first_conv,
                       first_kernel_size, bias=False))
        cur_channels_count = out_chans_first_conv

        ## Downsampling path ##
        self.dense_down = nn.ModuleList([])
        self.trans_down = nn.ModuleList([])
        for n_layers in down_blocks:
            self.dense_down.append(self._denseblock(
                cur_channels_count, growth_rate, n_layers,
                upsample=False, dropout_rate=dropout_rate))
            cur_channels_count += (growth_rate * n_layers)
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.trans_down.append(self._trans_down(
                cur_channels_count, cur_channels_count,
                dropout_rate=dropout_rate))

        ## Bottleneck ##
        self.bottleneck = self._bottleneck(
            cur_channels_count, growth_rate, bottleneck_layers,
            dropout_rate=dropout_rate)
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        ## Upsampling path ##
        self.trans_up = nn.ModuleList([])
        self.dense_up = nn.ModuleList([])
        self.attention_gates = nn.ModuleList([])
        self.deep_supervision = nn.ModuleList([])
        up_info = zip(up_blocks, skip_connection_channel_counts)
        for i, (n_layers, sccc) in enumerate(up_info, 1):
            self.attention_gates.append(self._attention(
                sccc, prev_block_channels))
            self.trans_up.append(self._trans_up(
                prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + sccc
            not_last_block = i < len(up_blocks)  # do not upsample on last block
            self.dense_up.append(self._denseblock(
                cur_channels_count, growth_rate, n_layers,
                upsample=not_last_block, dropout_rate=dropout_rate))
            prev_block_channels = growth_rate * n_layers
            cur_channels_count += prev_block_channels
            dsv_channel_count = prev_block_channels if not_last_block else \
                cur_channels_count
            self.deep_supervision.append(
                self._conv(dsv_channel_count, out_channels,
                           final_kernel_size))

    @property
    def _down_blocks(self):
        return zip(self.dense_down, self.trans_down)

    @property
    def _up_blocks(self):
        return zip(self.dense_up, self.trans_up,
                   self.attention_gates, self.deep_supervision)

    def _interp(self, x: Tensor, size: Tuple[int]) -> Tensor:
        return F.interpolate(x, size, mode=self._upsample, align_corners=True)

    def forward(self, x: Tensor) -> List[Tensor]:
        x_size = x.shape[2:]
        out = self.first_conv(x)
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
            dsv = self._interp(dsv, x_size)
            dsvs.append(dsv)
        return dsvs

    def predict(self, x: Tensor) -> Tensor:
        out = self.first_conv(x)
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
    _upsample = 'bilinear'


class AttentionTiramisu3d(AttentionTiramisu):
    _attention = GridAttentionBlock3d
    _bottleneck = Bottleneck3d
    _conv = nn.Conv3d
    _denseblock = DenseBlock3d
    _pad = nn.ReplicationPad3d
    _trans_down = TransitionDown3d
    _trans_up = TransitionUp3d
    _upsample = 'trilinear'


if __name__ == "__main__":
    attention_block = GridAttentionBlock3d(1, 1)
    x = torch.randn(2, 1, 32, 32, 32)
    g = torch.randn(2, 1, 16, 16, 16)
    y = attention_block(x, g)
    assert x.shape == y.shape
    net_kwargs = dict(in_channels=1, out_channels=1,
                      down_blocks=[2, 2], up_blocks=[2, 2],
                      bottleneck_layers=2)
    net = AttentionTiramisu3d(**net_kwargs)
    y = net(x)
    assert all([x.shape == yi.shape for yi in y])
    y = net.predict(x)
    assert x.shape == y.shape
