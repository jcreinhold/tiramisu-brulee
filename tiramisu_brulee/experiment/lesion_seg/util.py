#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_seg.util

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 16, 2021
"""

__all__ = [
    'bbox3D',
    'extract_and_average',
    'minmax_scale_batch',
    'reshape_for_broadcasting',
    'to_np',
    'setup_log',
    'split_filename',
]

from typing import List, Tuple

import logging
import os

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


def bbox3D(img: np.ndarray, pad: int = 0) -> Tuple[int, int, int, int, int, int]:
    """ find a bounding box for a 3D array (with optional padding) """
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    i, j, k = img.shape
    return (max(rmin - pad, 0), min(rmax + pad, i),
            max(cmin - pad, 0), min(cmax + pad, j),
            max(zmin - pad, 0), min(zmax + pad, k))


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
