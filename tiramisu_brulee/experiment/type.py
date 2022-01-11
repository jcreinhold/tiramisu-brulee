"""Experiment-specific types
Author: Jacob Reinhold <jcreinhold@gmail.com>
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
    "PathLike",
    "positive_float",
    "positive_float_or_none",
    "positive_int",
    "positive_int_or_none",
    "positive_odd_int_or_none",
    "probability_float",
    "probability_float_or_none",
    "TiramisuBruleeInfo",
]

import argparse
import builtins
import collections
import os
import pathlib
import typing

import jsonargparse
import torch

BatchElement = typing.Union[
    torch.Tensor, typing.Dict[builtins.str, typing.Any], typing.List[typing.Any]
]
Batch = typing.Dict[builtins.str, BatchElement]
Indices = typing.Tuple[
    builtins.int, builtins.int, builtins.int, builtins.int, builtins.int, builtins.int
]
ModelNum = collections.namedtuple("ModelNum", ["num", "out_of"])
Namespace = typing.Union[argparse.Namespace, jsonargparse.Namespace]
PatchShape2D = typing.Tuple[builtins.int, builtins.int]
PatchShape3D = typing.Tuple[builtins.int, builtins.int, builtins.int]
PatchShape = typing.Union[PatchShape2D, PatchShape3D]
PatchShape2DOption = typing.Tuple[
    typing.Optional[builtins.int], typing.Optional[builtins.int]
]
PatchShape3DOption = typing.Tuple[
    typing.Optional[builtins.int],
    typing.Optional[builtins.int],
    typing.Optional[builtins.int],
]
PatchShapeOption = typing.Union[PatchShape2DOption, PatchShape3DOption]
ArgType = typing.Optional[typing.Union[Namespace, typing.Iterable[builtins.str]]]
ArgParser = typing.Union[argparse.ArgumentParser, jsonargparse.ArgumentParser]
TiramisuBruleeInfo = collections.namedtuple("TiramisuBruleeInfo", ["version", "commit"])
PathLike = typing.Union[builtins.str, os.PathLike]


# flake8: noqa: E501
def return_none(func: typing.Callable) -> typing.Callable:
    def new_func(self, string: typing.Any) -> typing.Any:  # type: ignore[no-untyped-def]
        if string is None:
            return None
        elif isinstance(string, builtins.str):
            if string.lower() in ("none", "null"):
                return None
        return func(self, string)

    return new_func


# flake8: noqa: E501
def return_str(match_string: builtins.str) -> typing.Callable:
    def decorator(func: typing.Callable) -> typing.Callable:
        def new_func(self, string: typing.Any) -> typing.Any:  # type: ignore[no-untyped-def]
            if isinstance(string, builtins.str):
                if string.lower() == match_string:
                    return match_string
            return func(self, string)

        return new_func

    return decorator


class _ParseType:
    @property
    def __name__(self) -> builtins.str:
        name = self.__class__.__name__
        assert isinstance(name, builtins.str)
        return name

    def __str__(self) -> builtins.str:
        return self.__name__


class file_path(_ParseType):
    def __call__(self, string: builtins.str) -> builtins.str:
        path = pathlib.Path(string)
        if not path.is_file():
            msg = f"{string} is not a valid path."
            raise argparse.ArgumentTypeError(msg)
        return str(path.resolve())


class positive_float(_ParseType):
    def __call__(self, string: builtins.str) -> builtins.float:
        num = float(string)
        if num <= 0.0:
            msg = f"{string} needs to be a positive float."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_float_or_none(_ParseType):
    @return_none
    def __call__(self, string: builtins.str) -> typing.Union[builtins.float, None]:
        return positive_float()(string)


class positive_int(_ParseType):
    def __call__(self, string: builtins.str) -> builtins.int:
        num = int(string)
        if num <= 0:
            msg = f"{string} needs to be a positive integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_odd_int_or_none(_ParseType):
    @return_none
    def __call__(self, string: builtins.str) -> typing.Union[builtins.int, None]:
        num = int(string)
        if num <= 0 or not (num % 2):
            msg = f"{string} needs to be a positive odd integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class positive_int_or_none(_ParseType):
    @return_none
    def __call__(self, string: builtins.str) -> typing.Union[builtins.int, None]:
        return positive_int()(string)


class nonnegative_int(_ParseType):
    def __call__(self, string: builtins.str) -> builtins.int:
        num = int(string)
        if num < 0:
            msg = f"{string} needs to be a nonnegative integer."
            raise argparse.ArgumentTypeError(msg)
        return num


class nonnegative_int_or_none_or_all(_ParseType):
    @return_none
    @return_str("all")
    def __call__(
        self, string: builtins.str
    ) -> typing.Union[builtins.int, None, builtins.str]:
        return nonnegative_int()(string)


class nonnegative_float(_ParseType):
    def __call__(self, string: builtins.str) -> builtins.float:
        num = float(string)
        if num < 0.0:
            msg = f"{string} needs to be a nonnegative float."
            raise argparse.ArgumentTypeError(msg)
        return num


class probability_float(_ParseType):
    def __call__(self, string: builtins.str) -> builtins.float:
        num = float(string)
        if num < 0.0 or num > 1.0:
            msg = f"{string} needs to be between 0 and 1."
            raise argparse.ArgumentTypeError(msg)
        return num


class probability_float_or_none(_ParseType):
    @return_none
    def __call__(self, string: builtins.str) -> typing.Union[builtins.float, None]:
        return probability_float()(string)


class NewParseType:
    def __init__(self, func: typing.Callable, name: builtins.str):
        self.name = name
        self.func = func

    def __str__(self) -> builtins.str:
        return self.name

    def __call__(self, val: typing.Any) -> typing.Any:
        return self.func(val)


def new_parse_type(func: typing.Callable, name: builtins.str) -> NewParseType:
    return NewParseType(func, name)
