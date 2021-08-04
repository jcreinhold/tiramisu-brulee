#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.loss

various segmentation loss functions

See Also:
    https://gitlab.com/shan-deep-networks/pytorch-metrics/

    https://github.com/catalyst-team/catalyst/

    https://github.com/facebookresearch/fvcore

    S.A. Taghanaki et al. "Combo loss: Handling input and
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
    tensor1: Tensor, tensor2: Tensor, eps: float = 1e-3, keepdim: bool = False,
) -> Tensor:
    """ compute dice score for each channel separately and reduce """
    spatial_dims = tuple(range(2 - len(tensor1.shape), 0))
    intersection = torch.sum(tensor1 * tensor2, dim=spatial_dims, keepdim=keepdim)
    x_sum = torch.sum(tensor1, dim=spatial_dims, keepdim=keepdim)
    y_sum = torch.sum(tensor2, dim=spatial_dims, keepdim=keepdim)
    pc_dice = (2 * intersection + eps) / (x_sum + y_sum + eps)
    return pc_dice


def weighted_channel_avg(tensor: Tensor, weight: Tensor) -> Tensor:
    weight = weight[None, ...].repeat([tensor.shape[0], 1])
    weighted = torch.mean(weight * tensor)
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
    dice: Tensor
    if reduction == "mean":
        if weight is None:
            dice = torch.mean(pc_dice)
        else:
            dice = weighted_channel_avg(pc_dice, weight)
    elif reduction == "none":
        dice = pc_dice
    else:
        raise NotImplementedError(f"{reduction} not implemented.")
    one_minus_dice: Tensor = 1.0 - dice
    return one_minus_dice


def binary_focal_loss(
    pred: Tensor,
    target: Tensor,
    pos_weight: Optional[Union[float, Tensor]] = None,
    reduction: str = "mean",
    gamma: float = 2.0,
) -> Tensor:
    """ focal loss for binary classification or segmentation """
    use_focal = gamma > 0.0
    bce_reduction = "none" if use_focal else reduction
    if use_focal:
        bce_pos_weight = None
    else:
        if pos_weight is not None and isinstance(pos_weight, float):
            bce_pos_weight = torch.tensor(
                [pos_weight], dtype=pred.dtype, device=pred.device
            )
        elif pos_weight is None or isinstance(pos_weight, torch.Tensor):
            bce_pos_weight = pos_weight
        else:
            raise ValueError(
                f"pos_weight must be a none, float, or tensor. Got {type(pos_weight)}."
            )
    bce_loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction=bce_reduction, pos_weight=bce_pos_weight,
    )
    loss_val: Tensor
    if use_focal:
        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)
        loss_val = bce_loss * ((1 - p_t) ** gamma)
    else:
        loss_val = bce_loss
    if pos_weight is not None and use_focal:
        weight = pos_weight / (1.0 + pos_weight)
        weight_t = weight * target + (1 - weight) * (1 - target)
        loss_val = weight_t * loss_val
    if use_focal:
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
    pos_weight: Optional[float] = None,
    focal_gamma: float = 0.0,
    combo_weight: float = 0.5,
) -> Tensor:
    """ combo loss (dice + focal weighted by combo_weight) for binary labels """
    assert 0.0 <= combo_weight <= 1.0
    assert 0.0 <= focal_gamma
    f_loss = binary_focal_loss(pred, target, pos_weight, reduction, focal_gamma,)
    p = torch.sigmoid(pred)
    d_loss = dice_loss(p, target, reduction=reduction)
    loss = combo_weight * f_loss + (1 - combo_weight) * d_loss
    return loss


def combo_loss(
    pred: Tensor,
    target: Tensor,
    num_classes: int,
    reduction: str = "mean",
    combo_weight: float = 0.5,
) -> Tensor:
    """ combo loss (dice + focal weighted by combo_weight) for multi-class labels """
    assert 0.0 <= combo_weight <= 1.0
    assert 2 <= num_classes
    channel_not_removed = pred.ndim == target.ndim
    if channel_not_removed and target.shape[1] > 1:
        msg = f"Channel size must be 1 or 0. Got {target.shape[1]}"
        raise ValueError(msg)
    _target = target[:, 0, ...] if channel_not_removed else target
    _target = _target.long()
    f_loss = F.cross_entropy(pred, _target, reduction=reduction)
    p = torch.softmax(pred, dim=1)
    target_one_hot = F.one_hot(_target, num_classes)
    target_one_hot = torch.movedim(target_one_hot, -1, 1)
    target_one_hot = target_one_hot.float()
    d_loss = dice_loss(p, target_one_hot, reduction=reduction)
    loss = combo_weight * f_loss + (1 - combo_weight) * d_loss
    return loss


def deeply_supervised_loss(  # type: ignore[no-untyped-def]
    preds: List[Tensor],
    target: Tensor,
    loss_func: Callable,
    level_weights: Union[float, List[float]] = 1.0,
    **loss_func_kwargs,
) -> Tensor:
    """ compute loss_func by comparing multiple same-shape preds to target """
    if isinstance(level_weights, float):
        level_weights = [level_weights] * len(preds)
    loss_val = torch.tensor(0.0, dtype=target.dtype, device=target.device)
    for lw, x in zip(level_weights, preds):
        loss_val += lw * loss_func(x, target, **loss_func_kwargs)
    return loss_val


def l1_segmentation_loss(
    pred: Tensor, target: Tensor, reduction: str = "mean",
) -> Tensor:
    """ l1 loss for segmentation by applying sigmoid to pred -> l1 """
    return F.l1_loss(torch.sigmoid(pred), target, reduction=reduction)


def mse_segmentation_loss(
    pred: Tensor, target: Tensor, reduction: str = "mean",
) -> Tensor:
    """ mse loss for segmentation by applying sigmoid to pred -> mse """
    return F.mse_loss(torch.sigmoid(pred), target, reduction=reduction)
