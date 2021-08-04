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
    "Batch",
    "BatchElement",
    "file_path",
    "Indices",
    "ModelNum",
    "Namespace",
    "new_parse_type",
    "nonnegative_float",
    "nonnegative_int",
    "nonnegative_int_or_none_or_all",
    "PatchShapeOption",
    "PatchShape",
    "positive_float",
    "positive_float_or_none",
    "positive_int",
    "positive_int_or_none",
    "positive_odd_int_or_none",
    "probability_float",
    "probability_float_or_none",
]

import argparse
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import jsonargparse
from torch import Tensor

BatchElement = Union[Tensor, Dict[str, Any], List[Any]]
Batch = Dict[str, BatchElement]
Indices = Tuple[int, int, int, int, int, int]
ModelNum = namedtuple("ModelNum", ["num", "out_of"])
Namespace = Union[argparse.Namespace, jsonargparse.Namespace]
PatchShape2D = Tuple[int, int]
PatchShape3D = Tuple[int, int, int]
PatchShape = Union[PatchShape2D, PatchShape3D]
PatchShape2DOption = Tuple[Optional[int], Optional[int]]
PatchShape3DOption = Tuple[Optional[int], Optional[int], Optional[int]]
PatchShapeOption = Union[PatchShape2DOption, PatchShape3DOption]
ArgType = Optional[Union[Namespace, Iterable[str]]]
ArgParser = Union[argparse.ArgumentParser, jsonargparse.ArgumentParser]


def return_none(func: Callable) -> Callable:
    def new_func(self, string: Any) -> Any:  # type: ignore[no-untyped-def]
        if string is None:
            return None
        elif isinstance(string, str):
            if string.lower() in ("none", "null"):
                return None
        return func(self, string)

    return new_func


def return_str(match_string: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        def new_func(self, string: Any) -> Any:  # type: ignore[no-untyped-def]
            if isinstance(string, str):
                if string.lower() == match_string:
                    return match_string
            return func(self, string)

        return new_func

    return decorator


class _ParseType:
    @property
    def __name__(self) -> str:
        name = self.__class__.__name__
        assert isinstance(name, str)
        return name

    def __str__(self) -> str:
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


class positive_float_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> Union[float, None]:
        return positive_float()(string)


class positive_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num <= 0:
            msg = f"{string} needs to be a positive integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_odd_int_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> Union[int, None]:
        num = int(string)
        if num <= 0 or not (num % 2):
            msg = f"{string} needs to be a positive odd integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_int_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> Union[int, None]:
        return positive_int()(string)


class nonnegative_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num < 0:
            msg = f"{string} needs to be a nonnegative integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class nonnegative_int_or_none_or_all(_ParseType):
    @return_none
    @return_str("all")
    def __call__(self, string: str) -> Union[int, None, str]:
        return nonnegative_int()(string)


class nonnegative_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num < 0.0:
            msg = f"{string} needs to be a nonnegative float."
            raise argparse.ArgumentTypeError(msg)
        return num


class probability_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num < 0.0 or num > 1.0:
            msg = f"{string} needs to be between 0 and 1."
            raise argparse.ArgumentTypeError(msg)
        return num


class probability_float_or_none(_ParseType):
    @return_none
    def __call__(self, string: str) -> Union[float, None]:
        return probability_float()(string)


class NewParseType:
    def __init__(self, func: Callable, name: str):
        self.name = name
        self.func = func

    def __str__(self) -> str:
        return self.name

    def __call__(self, val: Any) -> Any:
        return self.func(val)


def new_parse_type(func: Callable, name: str) -> NewParseType:
    return NewParseType(func, name)
