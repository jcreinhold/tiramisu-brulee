#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.util

miscellaneous tools for segmentation

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 16, 2021
"""

__all__ = [
    "append_num_to_filename",
    "BoundingBox3D",
    "image_one_hot",
    "minmax_scale_batch",
    "reshape_for_broadcasting",
    "to_np",
    "setup_log",
    "split_filename",
]

import logging
from pathlib import Path
from typing import Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from tiramisu_brulee.experiment.type import Indices

T = TypeVar("T", bound="BoundingBox3D")


def minmax_scale_batch(x: Tensor) -> Tensor:
    """ rescale a batch of image PyTorch tensors to be between 0 and 1 """
    dims = list(range(1, x.dim()))
    xmin = x.amin(dim=dims, keepdim=True)
    xmax = x.amax(dim=dims, keepdim=True)
    return (x - xmin) / (xmax - xmin)


def to_np(x: Tensor) -> np.ndarray:
    """ convert a PyTorch Tensor (potentially on GPU) to a numpy array """
    data = x.detach().cpu().numpy()
    assert isinstance(data, np.ndarray)
    return data


def image_one_hot(image: Tensor, num_classes: int) -> Tensor:
    num_channels = image.shape[1]
    if num_channels > 1:
        msg = f"Image must only have one channel. Got {num_channels} channels."
        raise RuntimeError(msg)
    encoded: Tensor = F.one_hot(image.long(), num_classes)
    encoded = encoded.transpose(1, -1)[..., 0].type(image.type())
    return encoded


class BoundingBox3D:
    def __init__(
        self,
        i_low: int,
        i_high: int,
        j_low: int,
        j_high: int,
        k_low: int,
        k_high: int,
        original_shape: Optional[Tuple[int, int, int]] = None,
    ):
        """ bounding box indices and crop/uncrop func for 3d vols """
        self.i = slice(i_low, i_high)
        self.j = slice(j_low, j_high)
        self.k = slice(k_low, k_high)
        self.original_shape = original_shape

    def crop_to_bbox(self, tensor: Tensor) -> Tensor:
        """ returns the tensor cropped around the saved bbox """
        return tensor[..., self.i, self.j, self.k]

    def __call__(self, tensor: Tensor) -> Tensor:
        return self.crop_to_bbox(tensor)

    def uncrop(self, tensor: Tensor) -> Tensor:
        """ places a tensor back into the saved original shape """
        assert tensor.ndim == 3, "expects tensors with shape HxWxD"
        assert self.original_shape is not None
        out = torch.zeros(self.original_shape, dtype=tensor.dtype, device=tensor.device)
        out[self.i, self.j, self.k] = tensor
        return out

    def uncrop_batch(self, batch: Tensor) -> Tensor:
        """ places a batch back into the saved original shape """
        assert batch.ndim == 5, "expects tensors with shape NxCxHxWxD"
        assert self.original_shape is not None
        batch_size, channel_size = batch.shape[:2]
        out_shape = (batch_size, channel_size) + tuple(self.original_shape)
        out = torch.zeros(out_shape, dtype=batch.dtype, device=batch.device)
        out[..., self.i, self.j, self.k] = batch
        return out

    @staticmethod
    def find_bbox(mask: Tensor, pad: int = 0) -> Indices:
        h = torch.where(torch.any(torch.any(mask, dim=1), dim=1))[0]
        w = torch.where(torch.any(torch.any(mask, dim=0), dim=1))[0]
        d = torch.where(torch.any(torch.any(mask, dim=0), dim=0))[0]
        h_low, h_high = h[0].item(), h[-1].item()
        w_low, w_high = w[0].item(), w[-1].item()
        d_low, d_high = d[0].item(), d[-1].item()
        i, j, k = mask.shape
        return (
            int(max(h_low - pad, 0)),
            int(min(h_high + pad, i)),
            int(max(w_low - pad, 0)),
            int(min(w_high + pad, j)),
            int(max(d_low - pad, 0)),
            int(min(d_high + pad, k)),
        )

    @classmethod
    def from_image(
        cls: Type[T], image: Tensor, pad: int = 0, foreground_min: float = 1e-4,
    ) -> T:
        """ find a bounding box for a 3D tensor (with optional padding) """
        foreground_mask = image > foreground_min
        assert isinstance(foreground_mask, Tensor)
        bbox_idxs = cls.find_bbox(foreground_mask, pad)
        original_shape = cls.get_shape(image)
        return cls(*bbox_idxs, original_shape=original_shape)

    @classmethod
    def from_batch(
        cls: Type[T],
        batch: Tensor,
        pad: int = 0,
        channel: int = 0,
        foreground_min: float = 1e-4,
    ) -> T:
        """ create bbox that works for a batch of 3d vols """
        assert batch.ndim == 5, "expects tensors with shape NxCxHxWxD"
        batch_size = batch.shape[0]
        assert batch_size > 0
        image_shape = batch.shape[2:]
        h_low, h_high = image_shape[0], -1
        w_low, w_high = image_shape[1], -1
        d_low, d_high = image_shape[2], -1
        for i in range(batch_size):
            image = batch[i, channel, ...]
            hl, hh, wl, wh, dl, dh = cls.find_bbox(image > foreground_min, pad)
            h_low, h_high = min(hl, h_low), max(hh, h_high)
            w_low, w_high = min(wl, w_low), max(wh, w_high)
            d_low, d_high = min(dl, d_low), max(dh, d_high)
        # noinspection PyUnboundLocalVariable
        original_shape = cls.get_shape(image)
        return cls(
            h_low, h_high, w_low, w_high, d_low, d_high, original_shape=original_shape,
        )

    @staticmethod
    def get_shape(image: Tensor) -> Tuple[int, int, int]:
        assert image.ndim == 3
        orig_x, orig_y, orig_z = tuple(image.shape)
        return (orig_x, orig_y, orig_z)


def reshape_for_broadcasting(tensor: Tensor, ndim: int) -> Tensor:
    """ expand dimensions of a 0- or 1-dimensional tensor to ndim for broadcast ops """
    assert tensor.ndim <= 1
    dims = [1 for _ in range(ndim - 1)]
    return tensor.view(-1, *dims)


def split_filename(filepath: Union[str, Path]) -> Tuple[Path, str, str]:
    """ split a filepath into the directory, base, and extension """
    filepath = Path(filepath).resolve()
    path = filepath.parent
    _base = Path(filepath.stem)
    ext = filepath.suffix
    if ext == ".gz":
        ext2 = _base.suffix
        base = str(_base.stem)
        ext = ext2 + ext
    else:
        base = str(_base)
    return Path(path), base, ext


def append_num_to_filename(filepath: Union[str, Path], num: int) -> Path:
    """ append num to the filename of filepath and return the modified path """
    path, base, ext = split_filename(filepath)
    base += f"_{num}"
    return path / (base + ext)


def setup_log(verbosity: int) -> None:
    """ set logger with verbosity logging level and message """
    if verbosity == 1:
        level = logging.getLevelName("INFO")
    elif verbosity >= 2:
        level = logging.getLevelName("DEBUG")
    else:
        level = logging.getLevelName("WARNING")
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt, level=level)
    logging.captureWarnings(True)
