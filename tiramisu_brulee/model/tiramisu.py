"""PyTorch Tiramisu network

PyTorch implementation of the Tiramisu network architecture.
Implementation based on `pytorch_tiramisu`.

Changes from `pytorch_tiramisu` include:
  1) removal of bias from conv layers,
  2) change zero padding to replication padding,
  3) cosmetic changes for brevity, clarity, consistency

References:
  Jégou, Simon, et al. "The one hundred layers tiramisu:
  Fully convolutional densenets for semantic segmentation."
  CVPR. 2017.

  https://github.com/bfortuner/pytorch_tiramisu

Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: Jul 01, 2020
"""

__all__ = [
    "Tiramisu2d",
    "Tiramisu3d",
]

import builtins
import typing

import torch
import torch.nn as nn

from tiramisu_brulee.model.dense import (
    Bottleneck2d,
    Bottleneck3d,
    DenseBlock2d,
    DenseBlock3d,
    ResizeMethod,
    TransitionDown2d,
    TransitionDown3d,
    TransitionUp2d,
    TransitionUp3d,
)


class Tiramisu(nn.Module):
    _bottleneck: typing.ClassVar[
        typing.Union[typing.Type[Bottleneck2d], typing.Type[Bottleneck3d]]
    ]
    _conv: typing.ClassVar[typing.Union[typing.Type[nn.Conv2d], typing.Type[nn.Conv3d]]]
    _denseblock: typing.ClassVar[
        typing.Union[typing.Type[DenseBlock2d], typing.Type[DenseBlock3d]]
    ]
    _trans_down: typing.ClassVar[
        typing.Union[typing.Type[TransitionDown2d], typing.Type[TransitionDown3d]]
    ]
    _trans_up: typing.ClassVar[
        typing.Union[typing.Type[TransitionUp2d], typing.Type[TransitionUp3d]]
    ]
    _first_kernel_size: typing.ClassVar[
        typing.Union[
            typing.Tuple[builtins.int, builtins.int],
            typing.Tuple[builtins.int, builtins.int, builtins.int],
        ]
    ]
    _final_kernel_size: typing.ClassVar[
        typing.Union[
            typing.Tuple[builtins.int, builtins.int],
            typing.Tuple[builtins.int, builtins.int, builtins.int],
        ]
    ]
    _padding_mode: typing.ClassVar[builtins.str] = "replicate"

    # flake8: noqa: E501
    def __init__(
        self,
        *,
        in_channels: builtins.int = 3,
        out_channels: builtins.int = 1,
        down_blocks: typing.Collection[builtins.int] = (5, 5, 5, 5, 5),
        up_blocks: typing.Collection[builtins.int] = (5, 5, 5, 5, 5),
        bottleneck_layers: builtins.int = 5,
        growth_rate: builtins.int = 16,
        first_conv_out_channels: builtins.int = 48,
        dropout_rate: builtins.float = 0.2,
        resize_method: ResizeMethod = ResizeMethod.CROP,
        input_shape: typing.Optional[typing.Tuple[builtins.int, ...]] = None,
        static_upsample: builtins.bool = False,
    ):
        """
        Base class for Tiramisu convolutional neural network

        See Also:
            Jégou, Simon, et al. "The one hundred layers tiramisu: Fully
            convolutional densenets for semantic segmentation." CVPR. 2017.

            Based on: https://github.com/bfortuner/pytorch_tiramisu

        Args:
            in_channels (builtins.int): number of input channels
            out_channels (builtins.int): number of output channels
            down_blocks (typing.Collection[builtins.int]): number of layers in each block in down path
            up_blocks (typing.Collection[builtins.int]): number of layers in each block in up path
            bottleneck_layers (builtins.int): number of layers in the bottleneck
            growth_rate (builtins.int): number of channels to grow by in each layer
            first_conv_out_channels (builtins.int): number of output channels in first conv
            dropout_rate (builtins.float): dropout rate/probability
            resize_method (ResizeMethod): method to resize the image in upsample branch
            input_shape: optionally provide shape of the input image (for onnx)
            static_upsample: use static upsampling when capable if input_shape provided
                (doesn't check upsampled size matches)
        """
        super().__init__()
        assert len(down_blocks) == len(up_blocks)

        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        skip_connection_channel_counts: typing.List[builtins.int] = []
        if input_shape is not None:
            tensor_shape = torch.as_tensor(input_shape)
            shapes = [tuple(map(int, input_shape))]

        self.first_conv = nn.Sequential(
            self._conv(
                in_channels,
                first_conv_out_channels,
                self._first_kernel_size,  # type: ignore[arg-type]
                bias=False,
                padding="same",
                padding_mode=self._padding_mode,
            ),
        )
        cur_channels_count: builtins.int = first_conv_out_channels

        # Downsampling path
        self.dense_down = nn.ModuleList([])
        self.trans_down = nn.ModuleList([])
        for i, n_layers in enumerate(down_blocks, 1):
            denseblock = self._denseblock(
                in_channels=cur_channels_count,
                growth_rate=growth_rate,
                n_layers=n_layers,
                upsample=False,
                dropout_rate=dropout_rate,
            )
            self.dense_down.append(denseblock)
            cur_channels_count += growth_rate * n_layers
            skip_connection_channel_counts.insert(0, cur_channels_count)
            trans_down_block = self._trans_down(
                in_channels=cur_channels_count,
                out_channels=cur_channels_count,
                dropout_rate=dropout_rate,
            )
            self.trans_down.append(trans_down_block)
            if i < len(down_blocks) and input_shape is not None:
                tensor_shape = torch.div(tensor_shape, 2, rounding_mode="floor")
                shapes.append(tuple(map(int, tensor_shape)))

        # Bottleneck
        self.bottleneck = self._bottleneck(
            in_channels=cur_channels_count,
            growth_rate=growth_rate,
            n_layers=bottleneck_layers,
            dropout_rate=dropout_rate,
        )
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        # Upsampling path
        self.dense_up = nn.ModuleList([])
        self.trans_up = nn.ModuleList([])
        up_info = zip(up_blocks, skip_connection_channel_counts)
        for i, (n_layers, sccc) in enumerate(up_info, 1):
            resize_shape = None if input_shape is None else shapes.pop()
            if resize_shape is not None and static_upsample:
                _static_upsample = all(x % 2 == 0 for x in resize_shape)
            else:
                _static_upsample = False
            trans_up_block = self._trans_up(
                in_channels=prev_block_channels,
                out_channels=prev_block_channels,
                resize_method=resize_method,
                resize_shape=resize_shape,
                static=_static_upsample,
            )
            self.trans_up.append(trans_up_block)
            cur_channels_count = prev_block_channels + sccc
            upsample = i < len(up_blocks)  # do not upsample on last block
            denseblock = self._denseblock(
                in_channels=cur_channels_count,
                growth_rate=growth_rate,
                n_layers=n_layers,
                upsample=upsample,
                dropout_rate=dropout_rate,
            )
            self.dense_up.append(denseblock)
            prev_block_channels = growth_rate * n_layers
            cur_channels_count += prev_block_channels

        self.final_conv = self._conv(
            in_channels=cur_channels_count,
            out_channels=out_channels,
            kernel_size=self._final_kernel_size,  # type: ignore[arg-type]
            bias=True,
            padding="same",
            padding_mode=self._padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.first_conv(x)
        skip_connections = []
        for dbd, tdb in zip(self.dense_down, self.trans_down):
            out = dbd(out)
            skip_connections.append(out)
            out = tdb(out)
        out = self.bottleneck(out)
        for ubd, tub in zip(self.dense_up, self.trans_up):
            skip = skip_connections.pop()
            out = tub(out, skip=skip)
            out = ubd(out)
        out = self.final_conv(out)
        assert isinstance(out, torch.Tensor)
        return out


class Tiramisu2d(Tiramisu):
    _bottleneck = Bottleneck2d
    _conv = nn.Conv2d
    _denseblock = DenseBlock2d
    _pad = nn.ReplicationPad2d
    _trans_down = TransitionDown2d
    _trans_up = TransitionUp2d
    _first_kernel_size = (3, 3)
    _final_kernel_size = (1, 1)


class Tiramisu3d(Tiramisu):
    _bottleneck = Bottleneck3d
    _conv = nn.Conv3d
    _denseblock = DenseBlock3d
    _pad = nn.ReplicationPad3d
    _trans_down = TransitionDown3d
    _trans_up = TransitionUp3d
    _first_kernel_size = (3, 3, 3)
    _final_kernel_size = (1, 1, 1)
