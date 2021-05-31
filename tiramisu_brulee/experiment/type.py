#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.type

experiment-specific types

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 28, 2021
"""

__all__ = [
    "ArgParser",
    "ArgType",
    "file_path",
    "Indices",
    "ModelNum",
    "Namespace",
    "new_parse_type",
    "nonnegative_int",
    "PatchShapeOption",
    "PatchShape",
    "positive_float",
    "positive_int",
    "positive_int_or_none",
    "positive_odd_int",
    "probability_float",
]

import argparse
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import jsonargparse

Indices = Tuple[int, int, int, int, int, int]
ModelNum = namedtuple("ModelNum", ["num", "out_of"])
Namespace = Union[argparse.Namespace, jsonargparse.Namespace]
PatchShape2D = Tuple[int, int]
PatchShape3D = Tuple[int, int, int]
PatchShape = Union[PatchShape2D, PatchShape3D]
PatchShape2DOption = Tuple[Optional[int], Optional[int]]
PatchShape3DOption = Tuple[Optional[int], Optional[int], Optional[int]]
PatchShapeOption = Union[PatchShape2DOption, PatchShape3DOption]
ArgType = Optional[Union[Namespace, List[str]]]
ArgParser = Union[argparse.ArgumentParser, jsonargparse.ArgumentParser]


class _ParseType:
    @property
    def __name__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__name__


class file_path(_ParseType):
    def __call__(self, string: str) -> str:
        path = Path(string)
        if not path.is_file():
            msg = f"{string} is not a valid path."
            raise argparse.ArgumentTypeError(msg)
        return str(path.resolve())


class positive_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num <= 0.0:
            msg = f"{string} needs to be a positive float."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num <= 0:
            msg = f"{string} needs to be a positive integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_odd_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num <= 0 or not (num % 2):
            msg = f"{string} needs to be a positive odd integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_int_or_none(_ParseType):
    def __call__(self, string: str) -> int:
        if isinstance(string, str):
            str_lower = string.lower()
            if str_lower in ("none", "null"):
                return None
        elif string is None:
            return None
        return positive_int()(string)


class nonnegative_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num < 0:
            msg = f"{string} needs to be a nonnegative integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class probability_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num <= 0.0 or num >= 1.0:
            msg = f"{string} needs to be between 0 and 1."
            raise argparse.ArgumentTypeError(msg)
        return num


def new_parse_type(func: Callable, name: str):
    class NewParseType:
        def __str__(self):
            return name

        def __call__(self, val: Any):
            return func(val)

    return NewParseType()
