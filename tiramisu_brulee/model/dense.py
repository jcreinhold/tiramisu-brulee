"""Blocks/layers for densely-connected networks
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: Jul 02, 2020
"""

__all__ = [
    "ACTIVATION",
    "Bottleneck2d",
    "Bottleneck3d",
    "DenseBlock2d",
    "DenseBlock3d",
    "TransitionDown2d",
    "TransitionDown3d",
    "TransitionUp2d",
    "TransitionUp3d",
]

import builtins
import functools
import typing

import torch
import torch.nn as nn

ACTIVATION = functools.partial(nn.ReLU, inplace=True)


# partial not supported by mypy so avoid to type check
# https://github.com/python/mypy/issues/1484
class Dropout2d(nn.Dropout2d):
    def __init__(
        self, p: builtins.float = 0.5, *, inplace: builtins.bool = True
    ) -> None:
        super().__init__(p, inplace)


class Dropout3d(nn.Dropout3d):
    def __init__(
        self, p: builtins.float = 0.5, *, inplace: builtins.bool = True
    ) -> None:
        super().__init__(p, inplace)


class ConvLayer(nn.Sequential):
    _conv: typing.Union[typing.Type[nn.Conv2d], typing.Type[nn.Conv3d]]
    _dropout: typing.Union[typing.Type[nn.Dropout2d], typing.Type[nn.Dropout3d]]
    _kernel_size: typing.Union[
        typing.Tuple[builtins.int, builtins.int],
        typing.Tuple[builtins.int, builtins.int, builtins.int],
    ]
    _maxpool: typing.Union[None, typing.Type[nn.MaxPool2d], typing.Type[nn.MaxPool3d]]
    _norm: typing.Union[typing.Type[nn.BatchNorm2d], typing.Type[nn.BatchNorm3d]]
    _pad = typing.Union[
        typing.Type[nn.ReplicationPad2d], typing.Type[nn.ReplicationPad3d]
    ]

    def __init__(
        self,
        *,
        in_channels: builtins.int,
        out_channels: builtins.int,
        dropout_rate: builtins.float = 0.2,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.add_module("norm", self._norm(in_channels))
        self.add_module("act", ACTIVATION())
        if self._use_padding():
            padding = 2 * [ks // 2 for ks in self._kernel_size]
            pad = self._pad(padding)  # type: ignore[operator]
            self.add_module("pad", pad)
        conv = self._conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self._kernel_size,  # type: ignore[arg-type]
            bias=False,
        )
        self.add_module("conv", conv)
        if self._use_dropout():
            self.add_module("drop", self._dropout(dropout_rate))
        if self._maxpool is not None:  # use maxpool if not None
            self.add_module("maxpool", self._maxpool(2))

    def _use_dropout(self) -> builtins.bool:
        return self.dropout_rate > 0.0

    def _use_padding(self) -> builtins.bool:
        return any([ks > 2 for ks in self._kernel_size])


class ConvLayer2d(ConvLayer):
    _conv = nn.Conv2d
    _dropout = Dropout2d
    _kernel_size = (3, 3)
    _maxpool = None
    _norm = nn.BatchNorm2d
    _pad = nn.ReplicationPad2d


class ConvLayer3d(ConvLayer):
    _conv = nn.Conv3d
    _dropout = Dropout3d
    _kernel_size = (3, 3, 3)
    _maxpool = None
    _norm = nn.BatchNorm3d
    _pad = nn.ReplicationPad3d


class DenseBlock(nn.Module):
    _layer: typing.Union[typing.Type[ConvLayer2d], typing.Type[ConvLayer3d]]

    def __init__(
        self,
        *,
        in_channels: builtins.int,
        growth_rate: builtins.int,
        n_layers: builtins.int,
        upsample: builtins.bool = False,
        dropout_rate: builtins.float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.n_layers = n_layers
        self.upsample = upsample
        self.dropout_rate = dropout_rate
        # out_channels = growth_rate b/c out_channels added w/ each layer
        _layer = functools.partial(
            self._layer,
            out_channels=self.growth_rate,
            dropout_rate=self.dropout_rate,
        )
        icr = self.in_channels_range
        self.layers = nn.ModuleList([_layer(in_channels=ic) for ic in icr])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            new_features = []
            # We pass all previous activations builtins.into each dense
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
    def in_channels_range(self) -> typing.List[builtins.int]:
        ic, gr = self.in_channels, self.growth_rate
        return [ic + i * gr for i in range(self.n_layers)]


class DenseBlock2d(DenseBlock):
    _layer = ConvLayer2d


class DenseBlock3d(DenseBlock):
    _layer = ConvLayer3d


class TransitionDown2d(ConvLayer):
    _conv = nn.Conv2d
    _dropout = Dropout2d
    _kernel_size = (1, 1)
    _maxpool = nn.MaxPool2d
    _norm = nn.BatchNorm2d
    _pad = nn.ReplicationPad2d


class TransitionDown3d(ConvLayer):
    _conv = nn.Conv3d
    _dropout = Dropout3d
    _kernel_size = (1, 1, 1)
    _maxpool = nn.MaxPool3d
    _norm = nn.BatchNorm3d
    _pad = nn.ReplicationPad3d


class TransitionUp(nn.Module):
    _conv_trans: typing.Union[
        typing.Type[nn.ConvTranspose2d], typing.Type[nn.ConvTranspose3d]
    ]
    _kernel_size: typing.Union[
        typing.Tuple[builtins.int, builtins.int],
        typing.Tuple[builtins.int, builtins.int, builtins.int],
    ]
    _stride: typing.Union[
        typing.Tuple[builtins.int, builtins.int],
        typing.Tuple[builtins.int, builtins.int, builtins.int],
    ]

    def __init__(self, *, in_channels: builtins.int, out_channels: builtins.int):
        super().__init__()
        self.conv_trans = self._conv_trans(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self._kernel_size,  # type: ignore[arg-type]
            stride=self._stride,  # type: ignore[arg-type]
            bias=False,
        )

    def forward(self, tensor: torch.Tensor, *, skip: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv_trans(tensor)
        out = self._crop_to_target(out, target=skip)
        out = torch.cat([out, skip], 1)
        return out

    @staticmethod
    def _crop_to_target(tensor: torch.Tensor, *, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransitionUp2d(TransitionUp):
    _conv_trans = nn.ConvTranspose2d
    _kernel_size = (3, 3)
    _stride = (2, 2)

    @staticmethod
    def _crop_to_target(tensor: torch.Tensor, *, target: torch.Tensor) -> torch.Tensor:
        _, _, max_height, max_width = target.shape
        _, _, _h, _w = tensor.size()
        h = torch.div(_h - max_height, 2, rounding_mode="trunc")
        w = torch.div(_w - max_width, 2, rounding_mode="trunc")
        hs = slice(h, h + max_height)
        ws = slice(w, w + max_width)
        return tensor[:, :, hs, ws]


class TransitionUp3d(TransitionUp):
    _conv_trans = nn.ConvTranspose3d
    _kernel_size = (3, 3, 3)
    _stride = (2, 2, 2)

    @staticmethod
    def _crop_to_target(tensor: torch.Tensor, *, target: torch.Tensor) -> torch.Tensor:
        _, _, max_height, max_width, max_depth = target.shape
        _, _, _h, _w, _d = tensor.size()
        h = torch.div(_h - max_height, 2, rounding_mode="trunc")
        w = torch.div(_w - max_width, 2, rounding_mode="trunc")
        d = torch.div(_d - max_depth, 2, rounding_mode="trunc")
        hs = slice(h, h + max_height)
        ws = slice(w, w + max_width)
        ds = slice(d, d + max_depth)
        return tensor[:, :, hs, ws, ds]


class Bottleneck(nn.Sequential):
    _layer: typing.Union[typing.Type[DenseBlock2d], typing.Type[DenseBlock3d]]

    def __init__(
        self,
        *,
        in_channels: builtins.int,
        growth_rate: builtins.int,
        n_layers: builtins.int,
        dropout_rate: builtins.float = 0.2,
    ):
        super().__init__()
        layer = self._layer(
            in_channels=in_channels,
            growth_rate=growth_rate,
            n_layers=n_layers,
            upsample=True,
            dropout_rate=dropout_rate,
        )
        self.add_module("bottleneck", layer)


class Bottleneck2d(Bottleneck):
    _layer = DenseBlock2d


class Bottleneck3d(Bottleneck):
    _layer = DenseBlock3d
