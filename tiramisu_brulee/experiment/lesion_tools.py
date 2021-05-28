#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_tools

functions specific to handling/processing lesion segmentations

Author: Jacob Reinhold (jcreinhold)
Created on: May 16, 2021
"""

__all__ = [
    "almost_isbi15_score",
    "clean_segmentation",
]

import numpy as np
from scipy.ndimage.morphology import (
    binary_fill_holes,
    generate_binary_structure,
)
from skimage.morphology import remove_small_objects
import torch
from torch import Tensor
from torchmetrics.functional import (
    dice_score,
    precision,
    pearson_corrcoef,
)


def clean_segmentation(
    label: np.ndarray, fill_holes: bool = True, minimum_lesion_size: int = 3
) -> np.ndarray:
    """ clean binary array by removing small objs & filling holes """
    d = label.ndim
    if fill_holes:
        structure = generate_binary_structure(d, d)
        label = binary_fill_holes(label, structure=structure)
    if minimum_lesion_size > 0:
        label = remove_small_objects(
            label, min_size=minimum_lesion_size, connectivity=d,
        )
    return label


def almost_isbi15_score(pred: Tensor, target: Tensor) -> Tensor:
    """ ISBI 15 MS challenge score excluding the LTPR & LFPR components """
    batch_size = pred.shape[0]
    dice = dice_score(pred.int(), target.int())
    if dice.isnan():
        dice = torch.tensor(0.0, device=pred.device)
    ppv = precision(pred.int(), target.int(), mdmc_average="samplewise")
    isbi15_score = 0.5 * dice + 0.5 * ppv
    if batch_size > 1:
        dims = list(range(1, pred.ndim))
        corr = pearson_corrcoef(
            pred.sum(dim=dims).float(), target.sum(dim=dims).float(),
        )
        isbi15_score = 0.5 * isbi15_score + 0.5 * corr
    return isbi15_score
