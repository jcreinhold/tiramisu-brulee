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
import torch.nn.functional as F
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
    batch_size, num_classes = pred.shape[0:2]
    multiclass = num_classes > 1
    dice = dice_score(pred.int(), target.int())
    if dice.isnan():
        dice = torch.tensor(0.0, device=pred.device)
    if multiclass and pred.shape != target.shape:
        is_integer_label = pred.ndim != target.ndim
        target = F.one_hot(target.long(), num_classes)
        if not is_integer_label and target.shape[1] > 1:
            msg = f"Target channel size must be 1 or 0. Got {target.shape[1]}"
            raise ValueError(msg)
        elif not is_integer_label:
            target = target[:, 0, ...]
        target = torch.movedim(target, -1, 1)
    ppv = precision(
        pred.int(),
        target.int(),
        num_classes=num_classes if multiclass else None,
        mdmc_average="samplewise",
        multiclass=multiclass or None,
    )
    isbi15_score = 0.5 * dice + 0.5 * ppv
    if batch_size > 1 and not multiclass:
        dims = list(range(1, pred.ndim))
        corr = pearson_corrcoef(
            pred.sum(dim=dims).float(), target.sum(dim=dims).float(),
        )
        isbi15_score = 0.5 * isbi15_score + 0.5 * corr
    return isbi15_score
