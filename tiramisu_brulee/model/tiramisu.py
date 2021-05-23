#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.model.tiramisu

PyTorch implementation of the Tiramisu network architecture [1]
(2D) Implementation based on [2].

Changes from [2] include:
  1) removal of bias from conv layers,
  2) change zero padding to replication padding,
  3) cosmetic changes for brevity, clarity, consistency

References:
  [1] Jégou, Simon, et al. "The one hundred layers tiramisu:
      Fully convolutional densenets for semantic segmentation."
      CVPR. 2017.
  [2] https://github.com/bfortuner/pytorch_tiramisu

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jul 01, 2020
"""

__all__ = [
    "Tiramisu2d",
    "Tiramisu3d",
]

from typing import List

from torch import Tensor
from torch import nn

from tiramisu_brulee.model.dense import (
    Bottleneck2d,
    Bottleneck3d,
    DenseBlock2d,
    DenseBlock3d,
    TransitionDown2d,
    TransitionDown3d,
    TransitionUp2d,
    TransitionUp3d,
)


class Tiramisu(nn.Module):
    _bottleneck = None
    _conv = None
    _denseblock = None
    _pad = None
    _trans_down = None
    _trans_up = None

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
        super().__init__()
        assert len(down_blocks) == len(up_blocks)
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        first_kernel_size = 3
        final_kernel_size = 1
        skip_connection_channel_counts = []

        self.first_conv = nn.Sequential(
            self._pad(first_kernel_size // 2),
            self._conv(
                in_channels, first_conv_out_channels, first_kernel_size, bias=False
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
                cur_channels_count, cur_channels_count, dropout_rate=dropout_rate
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
        self.dense_up = nn.ModuleList([])
        self.trans_up = nn.ModuleList([])
        up_info = zip(up_blocks, skip_connection_channel_counts)
        for i, (n_layers, sccc) in enumerate(up_info, 1):
            block = self._trans_up(prev_block_channels, prev_block_channels)
            self.trans_up.append(block)
            cur_channels_count = prev_block_channels + sccc
            upsample = i < len(up_blocks)  # do not upsample on last block
            block = self._denseblock(
                cur_channels_count,
                growth_rate,
                n_layers,
                upsample=upsample,
                dropout_rate=dropout_rate,
            )
            self.dense_up.append(block)
            prev_block_channels = growth_rate * n_layers
            cur_channels_count += prev_block_channels

        self.final_conv = self._conv(
            cur_channels_count, out_channels, final_kernel_size, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.first_conv(x)
        skip_connections = []
        for dbd, tdb in zip(self.dense_down, self.trans_down):
            out = dbd(out)
            skip_connections.append(out)
            out = tdb(out)
        out = self.bottleneck(out)
        for ubd, tub in zip(self.dense_up, self.trans_up):
            skip = skip_connections.pop()
            out = tub(out, skip)
            out = ubd(out)
        out = self.final_conv(out)
        return out


class Tiramisu2d(Tiramisu):
    _bottleneck = Bottleneck2d
    _conv = nn.Conv2d
    _denseblock = DenseBlock2d
    _pad = nn.ReplicationPad2d
    _trans_down = TransitionDown2d
    _trans_up = TransitionUp2d


class Tiramisu3d(Tiramisu):
    _bottleneck = Bottleneck3d
    _conv = nn.Conv3d
    _denseblock = DenseBlock3d
    _pad = nn.ReplicationPad3d
    _trans_down = TransitionDown3d
    _trans_up = TransitionUp3d
