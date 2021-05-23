#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.loss

various segmentation loss functions

References:
    [1] https://gitlab.com/shan-deep-networks/pytorch-metrics/
    [2] https://github.com/catalyst-team/catalyst/
    [3] https://github.com/facebookresearch/fvcore
    [4] S.A. Taghanaki et al. "Combo loss: Handling input and
        output imbalance in multi-organ segmentation." Computerized
        Medical Imaging and Graphics 75 (2019): 24-33.

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jul 01, 2020
"""

__all__ = [
    "binary_combo_loss",
    "binary_focal_loss",
    "deeply_supervised_loss",
    "dice_loss",
    "l1_segmentation_loss",
    "mse_segmentation_loss",
]

from typing import Callable, List, Optional, Union

import torch
from torch import Tensor
import torch.nn.functional as F


def per_channel_dice(
    arr1: Tensor, arr2: Tensor, eps: float = 1e-3, keepdim: bool = False
) -> Tensor:
    """ compute dice score for each channel separately and reduce """
    spatial_dims = tuple(range(2 - len(arr1.shape), 0))
    intersection = torch.sum(arr1 * arr2, dim=spatial_dims, keepdim=keepdim)
    x_sum = torch.sum(arr1, dim=spatial_dims, keepdim=keepdim)
    y_sum = torch.sum(arr2, dim=spatial_dims, keepdim=keepdim)
    pc_dice = (2 * intersection + eps) / (x_sum + y_sum + eps)
    return pc_dice


def weighted_channel_avg(arr: Tensor, weight: Tensor) -> Tensor:
    weight = weight[None, ...].repeat([arr.shape[0], 1])
    weighted = torch.mean(weight * arr)
    return weighted


def dice_loss(
    pred: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: str = "mean",
    eps: float = 1e-3,
) -> Tensor:
    """ sorensen-dice coefficient loss function """
    keepdim = reduction != "mean"
    pc_dice = per_channel_dice(pred, target, eps=eps, keepdim=keepdim)
    if reduction == "mean":
        if weight is None:
            dice = torch.mean(pc_dice)
        else:
            dice = weighted_channel_avg(pc_dice, weight)
    elif reduction == "none":
        dice = pc_dice
    else:
        raise NotImplementedError(f"{reduction} not implemented.")
    return 1 - dice


def binary_focal_loss(
    pred: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: str = "mean",
    gamma: float = 2.0,
) -> Tensor:
    """ focal loss for binary classification or segmentation """
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    if gamma > 0.0:
        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)
        loss_val = ce_loss * ((1 - p_t) ** gamma)
    else:
        loss_val = ce_loss
    if weight is not None:
        weight_t = weight * target + (1 - weight) * (1 - target)
        loss_val = weight_t * loss_val
    if reduction == "mean":
        loss_val = loss_val.mean()
    elif reduction == "sum":
        loss_val = loss_val.sum()
    elif reduction == "batchwise_mean":
        loss_val = loss_val.sum(0)
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError(f"{reduction} not implemented.")
    return loss_val


def binary_combo_loss(
    pred: Tensor,
    target: Tensor,
    reduction: str = "mean",
    focal_weight: Optional[Tensor] = None,
    focal_gamma: float = 0.0,
    combo_weight: float = 0.5,
) -> Tensor:
    """ combo loss (dice + focal weighted by combo_weight) """
    assert 0.0 <= combo_weight <= 1.0
    assert 0.0 <= focal_gamma
    f_loss = binary_focal_loss(pred, target, focal_weight, reduction, focal_gamma)
    p = torch.sigmoid(pred)
    d_loss = dice_loss(p, target, reduction=reduction)
    loss = combo_weight * f_loss + (1 - combo_weight) * d_loss
    return loss


def deeply_supervised_loss(
    preds: List[Tensor],
    target: Tensor,
    loss_func: Callable,
    level_weights: Union[float, List[float]] = 1.0,
    **loss_func_kwargs,
) -> Tensor:
    """ compute loss_func by comparing multiple same-shape preds to target """
    if isinstance(level_weights, float):
        level_weights = [level_weights] * len(preds)
    loss_val = 0.0
    for lw, x in zip(level_weights, preds):
        loss_val += lw * loss_func(x, target, **loss_func_kwargs)
    return loss_val


def l1_segmentation_loss(
    pred: Tensor, target: Tensor, reduction: str = "mean"
) -> Tensor:
    """ l1 loss for segmentation by applying sigmoid to pred -> l1 """
    pred = torch.sigmoid(pred)
    return F.l1_loss(pred, target, reduction=reduction)


def mse_segmentation_loss(
    pred: Tensor, target: Tensor, reduction: str = "mean"
) -> Tensor:
    """ mse loss for segmentation by applying sigmoid to pred -> mse """
    pred = torch.sigmoid(pred)
    return F.mse_loss(pred, target, reduction=reduction)
