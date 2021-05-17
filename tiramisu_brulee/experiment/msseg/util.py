#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.msseg

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 16, 2021
"""

__all__ = [
    'bbox3D',
    'l1_segmentation_loss',
    'minmax_scale_batch',
    'mse_segmentation_loss',
    'to_np',
    'setup_log',
    'split_filename',
]

from typing import Tuple

import logging
import os

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def minmax_scale_batch(x: Tensor) -> Tensor:
    """ rescale a batch of image PyTorch tensors to be between 0 and 1 """
    dims = list(range(1, x.dim()))
    xmin = x.amin(dim=dims, keepdim=True)
    xmax = x.amax(dim=dims, keepdim=True)
    return (x - xmin) / (xmax - xmin)


def to_np(x: Tensor) -> np.ndarray:
    """ convert a PyTorch Tensor (potentially on GPU) to a numpy array """
    return x.detach().cpu().numpy()


def l1_segmentation_loss(pred: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    pred = torch.sigmoid(pred)
    return F.l1_loss(pred, target, reduction=reduction)


def mse_segmentation_loss(pred: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    pred = torch.sigmoid(pred)
    return F.mse_loss(pred, target, reduction=reduction)


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
