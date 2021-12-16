"""Functions specific to handling/processing lesion segmentations
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: May 16, 2021
"""

__all__ = [
    "almost_isbi15_score",
    "clean_segmentation",
]

import builtins
import typing

import numpy as np
import torch
from scipy.ndimage.morphology import binary_fill_holes, generate_binary_structure
from skimage.morphology import remove_small_objects
from torchmetrics.functional import dice_score, pearson_corrcoef, precision

from tiramisu_brulee.experiment.util import image_one_hot


def clean_segmentation(
    label: np.ndarray,
    *,
    fill_holes: builtins.bool = True,
    minimum_lesion_size: builtins.int = 3,
) -> np.ndarray:
    """clean binary array by removing small objs & filling holes"""
    d = label.ndim
    if fill_holes:
        structure = generate_binary_structure(d, d)
        label = binary_fill_holes(label, structure=structure)
    if minimum_lesion_size > 0:
        label = remove_small_objects(
            label,
            min_size=minimum_lesion_size,
            connectivity=d,
        )
    return label


def almost_isbi15_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    return_dice_ppv: builtins.bool = False,
) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """ISBI 15 MS challenge score excluding the LTPR & LFPR components"""
    batch_size, num_classes = pred.shape[0:2]
    multiclass = num_classes > 1
    one_hot_classes = num_classes if multiclass else 2
    pred_one_hot = (
        pred if multiclass else image_one_hot(pred, num_classes=one_hot_classes)
    )
    dice = dice_score(pred_one_hot, target.int())
    if multiclass and pred.shape != target.shape:
        is_integer_label = pred.ndim != target.ndim
        if is_integer_label:
            target.unsqueeze_(1)
            assert pred.ndim == target.ndim
        target = image_one_hot(target.long(), num_classes=num_classes)
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
            pred.sum(dim=dims).float(),
            target.sum(dim=dims).float(),
        )
        isbi15_score = 0.5 * isbi15_score + 0.5 * corr
    if return_dice_ppv:
        return isbi15_score, dice, ppv
    else:
        return isbi15_score
