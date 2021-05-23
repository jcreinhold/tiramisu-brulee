#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.model.dense

blocks/layers for densely-connected networks

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jul 02, 2020
"""

__all__ = [
    "Bottleneck2d",
    "Bottleneck3d",
    "DenseBlock2d",
    "DenseBlock3d",
    "TransitionDown2d",
    "TransitionDown3d",
    "TransitionUp2d",
    "TransitionUp3d",
]

from functools import partial
from typing import List

import torch
from torch import Tensor
from torch import nn

ACTIVATION = partial(nn.ReLU, inplace=True)


class ConvLayer(nn.Sequential):
    _conv = None
    _dropout = None
    _kernel_size = None
    _maxpool = None
    _norm = None
    _pad = None

    def __init__(self, in_channels: int, growth_rate: int, dropout_rate: float = 0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.add_module("norm", self._norm(in_channels))
        self.add_module("act", ACTIVATION())
        if self._use_padding():
            self.add_module("pad", self._pad(self._kernel_size // 2))
        conv = self._conv(in_channels, growth_rate, self._kernel_size, bias=False)
        self.add_module("conv", conv)
        if self._use_dropout():
            self.add_module("drop", self._dropout(dropout_rate))
        if self._use_maxpool():
            self.add_module("maxpool", self._maxpool(2))

    def _use_dropout(self) -> bool:
        return self.dropout_rate > 0.0

    def _use_padding(self) -> bool:
        return self._kernel_size > 2

    def _use_maxpool(self) -> bool:
        return self._maxpool is not None


class ConvLayer2d(ConvLayer):
    _conv = nn.Conv2d
    _dropout = partial(nn.Dropout2d, inplace=True)
    _kernel_size = 3
    _maxpool = None
    _norm = nn.BatchNorm2d
    _pad = nn.ReplicationPad2d


class ConvLayer3d(ConvLayer):
    _conv = nn.Conv3d
    _dropout = partial(nn.Dropout3d, inplace=True)
    _kernel_size = 3
    _maxpool = None
    _norm = nn.BatchNorm3d
    _pad = nn.ReplicationPad3d


class DenseBlock(nn.Module):
    _layer = None

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        n_layers: int,
        upsample: bool = False,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.n_layers = n_layers
        self.upsample = upsample
        self.dropout_rate = dropout_rate
        _layer = partial(
            self._layer, growth_rate=self.growth_rate, dropout_rate=self.dropout_rate
        )
        icr = self.in_channels_range
        self.layers = nn.ModuleList([_layer(ic) for ic in icr])

    def forward(self, tensor: Tensor) -> Tensor:
        if self.upsample:
            new_features = []
            # We pass all previous activations into each dense
            # layer normally but we only store each dense layer's
            # output in the new_features array. Note that all
            # concatenation is done on the channel axis (i.e., 1)
            for layer in self.layers:
                out = layer(tensor)
                tensor = torch.cat([tensor, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(tensor)
                tensor = torch.cat([tensor, out], 1)
            return tensor

    @property
    def in_channels_range(self) -> List[int]:
        ic, gr = self.in_channels, self.growth_rate
        return [ic + i * gr for i in range(self.n_layers)]


class DenseBlock2d(DenseBlock):
    _layer = ConvLayer2d


class DenseBlock3d(DenseBlock):
    _layer = ConvLayer3d


class TransitionDown2d(ConvLayer):
    _conv = nn.Conv2d
    _dropout = partial(nn.Dropout2d, inplace=True)
    _kernel_size = 1
    _maxpool = nn.MaxPool2d
    _norm = nn.BatchNorm2d
    _pad = nn.ReplicationPad2d


class TransitionDown3d(ConvLayer):
    _conv = nn.Conv3d
    _dropout = partial(nn.Dropout3d, inplace=True)
    _kernel_size = 1
    _maxpool = nn.MaxPool3d
    _norm = nn.BatchNorm3d
    _pad = nn.ReplicationPad3d


class TransitionUp(nn.Module):
    _conv_trans = None

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        kernel_size = 3
        self.conv_trans = self._conv_trans(
            in_channels, out_channels, kernel_size, stride=2, bias=False
        )

    def forward(self, tensor: Tensor, skip: Tensor) -> Tensor:
        out = self.conv_trans(tensor)
        out = self._crop_to_target(out, skip)
        out = torch.cat([out, skip], 1)
        return out

    @staticmethod
    def _crop_to_target(tensor: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError


class TransitionUp2d(TransitionUp):
    _conv_trans = nn.ConvTranspose2d

    @staticmethod
    def _crop_to_target(tensor: Tensor, target: Tensor) -> Tensor:
        _, _, max_height, max_width = target.shape
        _, _, h, w = tensor.size()
        h = (h - max_height) // 2
        w = (w - max_width) // 2
        hs = slice(h, h + max_height)
        ws = slice(w, w + max_width)
        return tensor[:, :, hs, ws]


class TransitionUp3d(TransitionUp):
    _conv_trans = nn.ConvTranspose3d

    @staticmethod
    def _crop_to_target(tensor: Tensor, target: Tensor) -> Tensor:
        _, _, max_height, max_width, max_depth = target.shape
        _, _, h, w, d = tensor.size()
        h = (h - max_height) // 2
        w = (w - max_width) // 2
        d = (d - max_depth) // 2
        hs = slice(h, h + max_height)
        ws = slice(w, w + max_width)
        ds = slice(d, d + max_depth)
        return tensor[:, :, hs, ws, ds]


class Bottleneck(nn.Sequential):
    _layer = None

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        n_layers: int,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        layer = self._layer(
            in_channels,
            growth_rate,
            n_layers,
            upsample=True,
            dropout_rate=dropout_rate,
        )
        self.add_module("bottleneck", layer)


class Bottleneck2d(Bottleneck):
    _layer = DenseBlock2d


class Bottleneck3d(Bottleneck):
    _layer = DenseBlock3d
