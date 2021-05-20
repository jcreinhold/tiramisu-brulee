#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_seg.util

miscellaneous tools for lesion segmentation

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 16, 2021
"""

__all__ = [
    "append_num_to_filename",
    "BoundingBox3D",
    "extract_and_average",
    "minmax_scale_batch",
    "reshape_for_broadcasting",
    "to_np",
    "setup_log",
    "split_filename",
]

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

Indices = Tuple[int, int, int, int, int, int]


def minmax_scale_batch(x: Tensor) -> Tensor:
    """ rescale a batch of image PyTorch tensors to be between 0 and 1 """
    dims = list(range(1, x.dim()))
    xmin = x.amin(dim=dims, keepdim=True)
    xmax = x.amax(dim=dims, keepdim=True)
    return (x - xmin) / (xmax - xmin)


def to_np(x: Tensor) -> np.ndarray:
    """ convert a PyTorch Tensor (potentially on GPU) to a numpy array """
    return x.detach().cpu().numpy()


def extract_and_average(dicts: List[dict], field: str) -> list:
    if len(dicts) == 1:
        return dicts[0][field]
    else:
        return torch.cat([d[field] for d in dicts]).mean()


class BoundingBox3D:
    def __init__(
        self,
        i_low: int,
        i_high: int,
        j_low: int,
        j_high: int,
        k_low: int,
        k_high: int,
        original_shape: Optional[List[int]] = None,
    ):
        """ bounding box indices and crop/uncrop func for 3d vols """
        self.i = slice(i_low, i_high)
        self.j = slice(j_low, j_high)
        self.k = slice(k_low, k_high)
        self.original_shape = original_shape

    def crop_to_bbox(self, x: Tensor) -> Tensor:
        return x[..., self.i, self.j, self.k]

    def __call__(self, x: Tensor) -> Tensor:
        return self.crop_to_bbox(x)

    def uncrop(self, x: Tensor) -> Tensor:
        assert x.ndim == 3, "expects tensors with shape HxWxD"
        out = torch.zeros(self.original_shape, dtype=x.dtype, device=x.device)
        out[self.i, self.j, self.k] = x
        return out

    def uncrop_batch(self, batch: Tensor) -> Tensor:
        assert batch.ndim == 5, "expects tensors with shape NxCxHxWxD"
        batch_size, channel_size = batch.shape[:2]
        out_shape = (batch_size, channel_size) + self.original_shape
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
            max(h_low - pad, 0),
            min(h_high + pad, i),
            max(w_low - pad, 0),
            min(w_high + pad, j),
            max(d_low - pad, 0),
            min(d_high + pad, k),
        )

    @classmethod
    def from_image(cls, image: Tensor, pad: int = 0, foreground_min: float = 1e-4):
        """ find a bounding box for a 3D tensor (with optional padding) """
        bbox_idxs = cls.find_bbox(image > foreground_min, pad)
        return cls(*bbox_idxs, original_shape=image.shape)

    @classmethod
    def from_batch(
        cls, batch: Tensor, pad: int = 0, channel: int = 0, foreground_min: float = 1e-4
    ):
        """ create bbox that works for a batch of 3d vols """
        assert batch.ndim == 5, "expects tensors with shape NxCxHxWxD"
        batch_size = batch.shape[0]
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
        return cls(
            h_low, h_high, w_low, w_high, d_low, d_high, original_shape=image_shape
        )


def reshape_for_broadcasting(x: Tensor, ndim: int) -> Tensor:
    dims = [1 for _ in range(ndim - 1)]
    return x.view(-1, *dims)


def split_filename(filepath: Union[str, Path]) -> Tuple[Path, str, str]:
    """ split a filepath into the directory, base, and extension """
    filepath = Path(filepath).resolve()
    path = filepath.parent
    base = Path(filepath.stem)
    ext = filepath.suffix
    if ext == ".gz":
        ext2 = base.suffix
        base = base.stem
        ext = ext2 + ext
    return Path(path), base, ext


def append_num_to_filename(filepath: Union[str, Path], num: int) -> Path:
    path, base, ext = split_filename(filepath)
    base += f"_{num}"
    return path / (base + ext)


def setup_log(verbosity: int):
    """ get logger with appropriate logging level and message """
    if verbosity == 1:
        level = logging.getLevelName("INFO")
    elif verbosity >= 2:
        level = logging.getLevelName("DEBUG")
    else:
        level = logging.getLevelName("WARNING")
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt, level=level)
