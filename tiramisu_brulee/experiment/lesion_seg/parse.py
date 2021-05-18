#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_seg.parser

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 17, 2021
"""

__all__ = [
    'file',
    'nonnegative_int',
    'positive_float',
    'positive_int',
    'threshold',
]

import argparse
from pathlib import Path


def file(string: str) -> Path:
    path = Path(string)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f'{string} is not a valid path.')
    return path


def positive_float(string: str) -> float:
    num = float(string)
    if num <= 0.:
        raise argparse.ArgumentTypeError(f'{string} needs to be a positive float.')
    return num


def positive_int(string: str) -> int:
    num = int(string)
    if num <= 0:
        raise argparse.ArgumentTypeError(f'{string} needs to be a positive integer.')
    return num


def nonnegative_int(string: str) -> int:
    num = int(string)
    if num < 0:
        raise argparse.ArgumentTypeError(f'{string} needs to be a nonnegative integer.')
    return num


def threshold(string: str) -> float:
    num = float(string)
    if num <= 0. or num >= 1.:
        raise argparse.ArgumentTypeError(f'{string} needs to be between 0 and 1.')
    return num
