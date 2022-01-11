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
import enum
import functools
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION = functools.partial(nn.ReLU, inplace=True)


# partial not supported well by mypy; avoid to type check in class vars below
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
    _conv: typing.ClassVar[typing.Union[typing.Type[nn.Conv2d], typing.Type[nn.Conv3d]]]
    _dropout: typing.ClassVar[
        typing.Union[typing.Type[nn.Dropout2d], typing.Type[nn.Dropout3d]]
    ]
    _kernel_size: typing.ClassVar[
        typing.Union[
            typing.Tuple[builtins.int, builtins.int],
            typing.Tuple[builtins.int, builtins.int, builtins.int],
        ]
    ]
    _maxpool: typing.ClassVar[
        typing.Union[None, typing.Type[nn.MaxPool2d], typing.Type[nn.MaxPool3d]]
    ]
    _norm: typing.ClassVar[
        typing.Union[typing.Type[nn.BatchNorm2d], typing.Type[nn.BatchNorm3d]]
    ]
    _padding_mode: typing.ClassVar[builtins.str] = "replicate"

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
        padding: typing.Union[builtins.str, builtins.int]
        padding = "same" if self._use_padding() else 0
        conv = self._conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self._kernel_size,  # type: ignore[arg-type]
            bias=False,
            padding=padding,
            padding_mode=self._padding_mode,
        )
        self.add_module("conv", conv)
        if self._use_dropout():
            self.add_module("drop", self._dropout(dropout_rate))
        if self._maxpool is not None:  # use maxpool if not None
            self.add_module("maxpool", self._maxpool(2))

    def _use_dropout(self) -> builtins.bool:
        return self.dropout_rate > 0.0

    def _use_padding(self) -> builtins.bool:
        return any(ks > 2 for ks in self._kernel_size)


class ConvLayer2d(ConvLayer):
    _conv = nn.Conv2d
    _dropout = Dropout2d
    _kernel_size = (3, 3)
    _maxpool = None
    _norm = nn.BatchNorm2d


class ConvLayer3d(ConvLayer):
    _conv = nn.Conv3d
    _dropout = Dropout3d
    _kernel_size = (3, 3, 3)
    _maxpool = None
    _norm = nn.BatchNorm3d


class DenseBlock(nn.Module):
    _layer: typing.ClassVar[
        typing.Union[typing.Type[ConvLayer2d], typing.Type[ConvLayer3d]]
    ]

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
                tensor = torch.cat((tensor, out), 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(tensor)
                tensor = torch.cat((tensor, out), 1)
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


class TransitionDown3d(ConvLayer):
    _conv = nn.Conv3d
    _dropout = Dropout3d
    _kernel_size = (1, 1, 1)
    _maxpool = nn.MaxPool3d
    _norm = nn.BatchNorm3d


@enum.unique
class ResizeMethod(enum.Enum):
    CROP: builtins.str = "crop"
    INTERPOLATE: builtins.str = "interpolate"

    @classmethod
    def from_string(cls, string: builtins.str) -> "ResizeMethod":
        if string.lower() == "crop":
            return cls.CROP
        elif string.lower() == "interpolate":
            return cls.INTERPOLATE
        else:
            msg = f"Only 'crop' and 'interpolate' allowed. Got {string}"
            raise ValueError(msg)


class TransitionUp(nn.Module):
    _conv: typing.ClassVar[typing.Union[typing.Type[nn.Conv2d], typing.Type[nn.Conv3d]]]
    _conv_trans: typing.ClassVar[
        typing.Union[typing.Type[nn.ConvTranspose2d], typing.Type[nn.ConvTranspose3d]]
    ]
    _kernel_size: typing.ClassVar[
        typing.Union[
            typing.Tuple[builtins.int, builtins.int],
            typing.Tuple[builtins.int, builtins.int, builtins.int],
        ]
    ]
    _stride: typing.ClassVar[
        typing.Union[
            typing.Tuple[builtins.int, builtins.int],
            typing.Tuple[builtins.int, builtins.int, builtins.int],
        ]
    ]
    _interp_mode: typing.ClassVar[builtins.str]

    def __init__(
        self,
        *,
        in_channels: builtins.int,
        out_channels: builtins.int,
        resize_method: ResizeMethod = ResizeMethod.CROP,
        resize_shape: typing.Optional[typing.Tuple[builtins.int, ...]] = None,
        static: builtins.bool = False,
        use_conv_transpose: builtins.bool = True,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        _conv_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self._kernel_size,
            bias=False,
        )
        conv_kwargs: typing.Dict[builtins.str, typing.Any] = _conv_kwargs.copy()
        conv_kwargs["padding"] = "same"
        conv_kwargs["padding_mode"] = "replicate"
        conv_trans_kwargs: typing.Dict[builtins.str, typing.Any] = _conv_kwargs.copy()
        conv_trans_kwargs["stride"] = self._stride
        self.conv: typing.Union[
            nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d
        ]
        self.resize: typing.Callable[..., torch.Tensor]
        if resize_method == ResizeMethod.CROP:
            self.conv = self._conv_trans(**conv_trans_kwargs)
            self.resize = self._crop_to_target
            setattr(self, "forward", self._forward_dynamic_trans)
        elif resize_method == ResizeMethod.INTERPOLATE and not static:
            if use_conv_transpose:
                self.conv = self._conv_trans(**conv_trans_kwargs)
                setattr(self, "forward", self._forward_dynamic_trans)
            else:
                self.conv = self._conv(**conv_kwargs)
                setattr(self, "forward", self._forward_dynamic_conv)
            self.resize = self._interpolate_to_target
        elif resize_method == ResizeMethod.INTERPOLATE and static:
            self.conv = self._conv(**conv_kwargs)
            setattr(self, "forward", self._forward_static)
        else:
            msg = f"resize_method needs to be a ResizeMethod. Got {resize_method}"
            raise ValueError(msg)

    def _forward_dynamic_trans(
        self, tensor: torch.Tensor, *, skip: torch.Tensor
    ) -> torch.Tensor:
        out: torch.Tensor = self.conv(tensor)
        out = self.resize(out, target=skip)
        out = torch.cat((out, skip), 1)
        return out

    def _forward_dynamic_conv(
        self, tensor: torch.Tensor, *, skip: torch.Tensor
    ) -> torch.Tensor:
        out: torch.Tensor = self.resize(tensor, target=skip)
        out = self.conv(out)
        out = torch.cat((out, skip), 1)
        return out

    def _forward_static(
        self, tensor: torch.Tensor, *, skip: torch.Tensor
    ) -> torch.Tensor:
        out: torch.Tensor = self._interpolate(tensor, scale_factor=2.0)
        out = self.conv(out)
        out = torch.cat((out, skip), 1)
        return out

    def _crop_to_target(
        self, tensor: torch.Tensor, *, target: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def _interpolate_to_target(
        self, tensor: torch.Tensor, *, target: torch.Tensor
    ) -> torch.Tensor:
        return self._interpolate(tensor, size=target.shape[2:])

    def _interpolate(self, tensor: torch.Tensor, **kwargs: typing.Any) -> torch.Tensor:
        interp_kwargs = dict(mode=self._interp_mode, align_corners=True, **kwargs)
        out: torch.Tensor = F.interpolate(tensor, **interp_kwargs)
        return out


class TransitionUp2d(TransitionUp):
    _conv = nn.Conv2d
    _conv_trans = nn.ConvTranspose2d
    _kernel_size = (3, 3)
    _stride = (2, 2)
    _interp_mode = "bilinear"

    def _crop_to_target(
        self, tensor: torch.Tensor, *, target: torch.Tensor
    ) -> torch.Tensor:
        if self.resize_shape is None:
            _, _, max_h, max_w = target.shape
        else:
            max_h, max_w = self.resize_shape
        _, _, _h, _w = tensor.size()
        h = torch.div(_h - max_h, 2, rounding_mode="trunc")
        w = torch.div(_w - max_w, 2, rounding_mode="trunc")
        return tensor[:, :, h : h + max_h, w : w + max_w]  # type: ignore[misc]


class TransitionUp3d(TransitionUp):
    _conv = nn.Conv3d
    _conv_trans = nn.ConvTranspose3d
    _kernel_size = (3, 3, 3)
    _stride = (2, 2, 2)
    _interp_mode = "trilinear"

    # flake8: noqa: E501
    def _crop_to_target(
        self, tensor: torch.Tensor, *, target: torch.Tensor
    ) -> torch.Tensor:
        if self.resize_shape is None:
            _, _, max_h, max_w, max_d = target.shape
        else:
            max_h, max_w, max_d = self.resize_shape
        _, _, _h, _w, _d = tensor.size()
        h = torch.div(_h - max_h, 2, rounding_mode="trunc")
        w = torch.div(_w - max_w, 2, rounding_mode="trunc")
        d = torch.div(_d - max_d, 2, rounding_mode="trunc")
        return tensor[:, :, h : h + max_h, w : w + max_w, d : d + max_d]  # type: ignore[misc]


class Bottleneck(nn.Sequential):
    _layer: typing.ClassVar[
        typing.Union[typing.Type[DenseBlock2d], typing.Type[DenseBlock3d]]
    ]

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
