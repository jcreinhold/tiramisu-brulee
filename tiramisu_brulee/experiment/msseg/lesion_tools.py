#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_tools

Author: Jacob Reinhold (jcreinhold)
Created on: May 16, 2021
"""

__all__ = [
    'clean_segmentation',
]

import numpy as np
from scipy.ndimage.morphology import binary_fill_holes, generate_binary_structure
from skimage.morphology import remove_small_objects
from torchmetrics import Metric

from lesion_metrics import isbi15_score, corr


def clean_segmentation(label: np.ndarray,
                       fill_holes: bool = True,
                       minimum_lesion_size: int = 3) -> np.ndarray:
    d = label.ndim
    if fill_holes:
        structure = generate_binary_structure(d, d)
        label = binary_fill_holes(label, structure=structure)
    if minimum_lesion_size > 0:
        label = remove_small_objects(label, min_size=minimum_lesion_size, connectivity=d)
    return label


class ISBIScore(Metric):
    def forward(self, y_hat, y):
        y_, y_hat_ = y.squeeze(), y_hat.squeeze()
        if y_.ndim == 3 and y_hat_.ndim == 3:  # batch size 1
            isbiscore = isbi15_score(y_hat_, y_)
        elif y_.ndim == 4 and y_hat_.ndim == 4:
            isbiscore = 0.
            for y_i, y_hat_i in zip(y_, y_hat_):
                isbiscore += isbi15_score(y_hat_i, y_i, reweighted=False)
            dims = (1, 2, 3)
            pred_vols = y_hat_.sum(axis=dims)
            true_vols = y_.sum(axis=dims)
            vol_corr = corr(pred_vols, true_vols)
            isbiscore += vol_corr / 4
            isbiscore /= y_.shape[0]
        else:
            raise ValueError(f'y ndim={y_.ndim}; y_hat ndim={y_hat_.ndim} not valid.')
        if np.isnan(isbiscore):
            isbiscore = 0.
        return isbiscore
