#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_seg.lesion_tools

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
            label, min_size=minimum_lesion_size, connectivity=d
        )
    return label


def almost_isbi15_score(pred: Tensor, target: Tensor) -> Tensor:
    """ ISBI 15 MS challenge score excluding the LTPR & LFPR components """
    dice = dice_score(pred.int(), target.int())
    if dice.isnan():
        dice = torch.tensor(0.0, device=pred.device)
    ppv = precision(pred.int(), target.int(), mdmc_average="samplewise")
    corr = pearson_corrcoef(
        pred.flatten().float(), target.flatten().float()  # noqa  # noqa
    )
    return 0.25 * dice + 0.25 * ppv + 0.5 * corr
