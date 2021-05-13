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

import contextlib
from glob import glob
import os
from os.path import join

import pandas as pd
import torch

with open(os.devnull, "w") as f:
    with contextlib.redirect_stdout(f):
        import torchio

Type = Union[torchio.LABEL, torchio.INTENSITY, None]

VALID_NAMES = ('ct', 'flair', 'label', 'pd',
               't1', 't1c', 't2', 'weight', 'div')


def glob_ext(path: str, ext: str = '*.nii*') -> List[str]:
    """ grab all `ext` files in a directory and sort them for consistency """
    fns = sorted(glob(join(path, ext)))
    return fns


def _check_type(name: str) -> Type:
    if name == "label":
        type = torchio.LABEL
    elif name == "weight":
        type = None
    else:
        type = torchio.INTENSITY
    return type


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
            type = _check_type(name)
            val = row[1][name]
            if name == "weight" or name == "div":
                data[name] = torch.tensor(val, dtype=torch.float32)
            else:
                data[name] = torchio.Image(val, type=type)
        subject = torchio.Subject(
            name=subject_name,
            **data)
        subject_list.append(subject)

    return subject_list
