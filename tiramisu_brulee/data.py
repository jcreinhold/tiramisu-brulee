#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.data

general file/data-handling operations

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 01, 2020
"""

__all__ = ['csv_to_subjectlist',
           'glob_ext']

from typing import *

from glob import glob
from logging import getLogger
from os.path import join

import pandas as pd
import torch
import torchio

VALID_NAMES = ('ct', 'flair', 'label', 'pd',
               't1', 't1c', 't2', 'weight', 'div')

logger = getLogger(__name__)


def _get_type(name: str):
    name_ = name.lower()
    if name_ == "label":
        type = torchio.LABEL
    elif name_ == "weight" or name_ == "div":
        type = 'float'
    elif name_ in VALID_NAMES:
        type = torchio.INTENSITY
    else:
        logger.warning(f"{name} not in known {VALID_NAMES}. "
                       f"Assuming an non-label image type.")
        type = torchio.INTENSITY
    return type


def glob_ext(path: str, ext: str = '*.nii*') -> List[str]:
    """ grab all `ext` files in a directory and sort them for consistency """
    fns = sorted(glob(join(path, ext)))
    return fns


def csv_to_subjectlist(filename: str) -> List[torchio.Subject]:
    """ Convert a csv file to a list of torchio subjects

    Args:
        filename: Path to csv file formatted with
            `subject` in a column, describing the
            id/name of the subject (must be unique).
            Row will fill in the filenames per type.
            Other columns headers must be one of:
            ct, flair, label, pd, t1, t1c, t2, weight, div
            (`label` should correspond to a
             segmentation mask)
            (`weight` and `div` should correspond to a float)

    Returns:
        subject_list (List[torchio.Subject]): list of torchio Subjects
    """
    df = pd.read_csv(filename, index_col='subject')
    names = df.columns.to_list()
    if any([name not in VALID_NAMES for name in names]):
        raise ValueError(f'Column name needs to be in {VALID_NAMES}')

    subject_list = []
    for row in df.iterrows():
        subject_name = row[0]
        data = {}
        for name in names:
            type = _get_type(name)
            val = row[1][name]
            if type == 'float':
                data[name] = torch.tensor(val, dtype=torch.float32)
            else:
                data[name] = torchio.Image(val, type=type)
        subject = torchio.Subject(
            name=subject_name,
            **data)
        subject_list.append(subject)

    return subject_list
