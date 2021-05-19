#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_seg.util

miscellaneous tools for lesion segmentation

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 16, 2021
"""

__all__ = [
    'BoundingBox3D',
    'extract_and_average',
    'minmax_scale_batch',
    'reshape_for_broadcasting',
    'to_np',
    'setup_log',
    'split_filename',
]

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


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
    return torch.cat([d[field] for d in dicts]).mean()


class BoundingBox3D:
    def __init__(self,
                 i_low: int,
                 i_high: int,
                 j_low: int,
                 j_high: int,
                 k_low: int,
                 k_high: int,
                 original_shape: Optional[List[int]] = None):
        self.i = slice(i_low, i_high)
        self.j = slice(j_low, j_high)
        self.k = slice(k_low, k_high)
        self.original_shape = original_shape

    def crop_to_bbox(self, x: Tensor) -> Tensor:
        return x[..., self.i, self.j, self.j]

    def __call__(self, x: Tensor) -> Tensor:
        return self.crop_to_bbox(x)

    def uncrop(self, x: Tensor) -> Tensor:
        assert x.ndim == 3, "uncrop expects tensors with shape HxWxD"
        out = torch.zeros(self.original_shape, dtype=x.dtype, device=x.device)
        out[self.i, self.j, self.k] = x
        return out

    def uncrop_batch(self, batch: Tensor) -> Tensor:
        assert batch.ndim == 5, "uncrop_batch expects tensors with shape NxCxHxWxD"
        batch_size, channel_size = batch.shape[2:]
        out_shape = (batch_size, channel_size) + self.original_shape
        out = torch.zeros(out_shape, dtype=batch.dtype, device=batch.device)
        out[..., self.i, self.j, self.k] = batch
        return out

    @staticmethod
    def find_bbox(image: Tensor, pad: int = 0) -> Tuple[int, int, int, int, int, int]:
        r = torch.any(torch.any(image, dim=1), dim=1)
        c = torch.any(torch.any(image, dim=0), dim=1)
        z = torch.any(torch.any(image, dim=0), dim=0)
        rmin, rmax = torch.where(r)[0]
        cmin, cmax = torch.where(c)[0]
        zmin, zmax = torch.where(z)[0]
        i, j, k = image.shape
        return (max(rmin - pad, 0), min(rmax + pad, i),
                max(cmin - pad, 0), min(cmax + pad, j),
                max(zmin - pad, 0), min(zmax + pad, k))

    @classmethod
    def from_image(cls, image: Tensor, pad: int = 0):
        """ find a bounding box for a 3D tensor (with optional padding) """
        bbox_idxs = cls.find_bbox(image, pad)
        return cls(*bbox_idxs, original_shape=image.shape)

    @classmethod
    def from_batch(cls,
                   batch: Tensor,
                   pad: int = 0,
                   channel: int = 0,
                   foreground_min: float = 1e-4):
        assert batch.ndim == 5, "from_batch expects tensors with shape NxCxHxWxD"
        batch_size = batch.shape[0]
        image_shape = batch.shape[2:]
        h_low, h_high = float('inf'), float('-inf')
        w_low, w_high = float('inf'), float('-inf')
        d_low, d_high = float('inf'), float('-inf')
        for i in range(batch_size):
            image = batch[i, channel, ...]
            hl, hh, wl, wh, dl, dh = cls.find_bbox(image > foreground_min, pad=pad)
            h_low, h_high = min(hl, h_low), max(hh, h_high)
            w_low, w_high = min(wl, w_low), max(wh, w_high)
            d_low, d_high = min(dl, d_low), max(dh, d_high)
        return cls(h_low, h_high, w_low, w_high, d_low, d_high,
                   original_shape=image_shape)


def reshape_for_broadcasting(x: Tensor, ndim: int) -> Tensor:
    dims = [1 for _ in range(ndim - 1)]
    return x.view(-1, *dims)


def split_filename(filepath: str) -> Tuple[str, str, str]:
    """ split a filepath into the directory, base, and extension """
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def setup_log(verbosity: int):
    """ get logger with appropriate logging level and message """
    if verbosity == 1:
        level = logging.getLevelName('INFO')
    elif verbosity >= 2:
        level = logging.getLevelName('DEBUG')
    else:
        level = logging.getLevelName('WARNING')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)
