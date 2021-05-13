#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.loss

loss functions to support lesion segmentation

References:
    [1] https://gitlab.com/shan-deep-networks/pytorch-metrics/
    [2] https://github.com/catalyst-team/catalyst/
    [3] https://github.com/facebookresearch/fvcore
    [4] S.A. Taghanaki et al. "Combo loss: Handling input and
        output imbalance in multi-organ segmentation." Computerized
        Medical Imaging and Graphics 75 (2019): 24-33.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 01, 2020
"""

__all__ = ['binary_combo_loss',
           'binary_focal_loss',
           'deeply_supervised_loss',
           'dice_loss']

from typing import *

import torch
from torch import Tensor
import torch.nn.functional as F


def per_channel_dice(x: Tensor, y: Tensor, eps: float = 1e-3, keepdim: bool = False) -> Tensor:
    spatial_dims = tuple(range(2 - len(x.shape), 0))
    intersection = torch.sum(x * y, dim=spatial_dims, keepdim=keepdim)
    x_sum = torch.sum(x, dim=spatial_dims, keepdim=keepdim)
    y_sum = torch.sum(y, dim=spatial_dims, keepdim=keepdim)
    pc_dice = (2 * intersection + eps) / (x_sum + y_sum + eps)
    return pc_dice


def weighted_channel_avg(x: Tensor, weight: Tensor) -> Tensor:
    weight = weight[None, ...].repeat([x.shape[0], 1])
    weighted = torch.mean(weight * x)
    return weighted


def dice_loss(x: Tensor, y: Tensor, weight: Optional[Tensor] = None,
              reduction: str = 'mean', eps: float = 1e-3) -> Tensor:
    keepdim = reduction != 'mean'
    pc_dice = per_channel_dice(x, y, eps=eps, keepdim=keepdim)
    if reduction == 'mean':
        if weight is None:
            dice = torch.mean(pc_dice)
        else:
            dice = weighted_channel_avg(pc_dice, weight)
    elif reduction == 'none':
        dice = pc_dice
    else:
        raise NotImplementedError(f'{reduction} not implemented.')
    return 1 - dice


def binary_focal_loss(x: Tensor, y: Tensor, weight: Optional[Tensor] = None,
                      reduction: str = 'mean', gamma: float = 2.) -> Tensor:
    ce_loss = F.binary_cross_entropy_with_logits(x, y, reduction="none")
    if gamma > 0.:
        p = torch.sigmoid(x)
        p_t = p * y + (1 - p) * (1 - y)
        loss = ce_loss * ((1 - p_t) ** gamma)
    else:
        loss = ce_loss
    if weight is not None:
        weight_t = weight * y + (1 - weight) * (1 - y)
        loss = weight_t * loss
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "batchwise_mean":
        loss = loss.sum(0)
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError(f'{reduction} not implemented.')
    return loss


def binary_combo_loss(x: Tensor, y: Tensor, weight: Optional[Tensor] = None,
                      reduction: str = 'mean', gamma: float = 0., alpha: float = 0.5) -> Tensor:
    f_loss = binary_focal_loss(x, y, weight, reduction, gamma)
    p = torch.sigmoid(x)
    d_loss = dice_loss(p, y, reduction=reduction)
    loss = alpha * f_loss + (1 - alpha) * d_loss
    return loss


def deeply_supervised_loss(xs: List[Tensor], y: Tensor, loss_func: Callable,
                           level_weights: Union[float, List[float]] = 1.,
                           **loss_func_kwargs) -> Tensor:
    if isinstance(level_weights, float):
        level_weights = [level_weights] * len(xs)
    loss = 0.
    for lw, x in zip(level_weights, xs):
        loss += lw * loss_func(x, y, **loss_func_kwargs)
    return loss


if __name__ == "__main__":
    x = torch.randn(1, 1, 32, 32)
    xs = [x] * 2
    y = torch.randn(1, 1, 32, 32)
    loss = deeply_supervised_loss(xs, y, binary_combo_loss, [0.5, 1.], weight=0.6)
