#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_seg.parse

parsing functions (including types) for argparse
and config files in lesion_seg

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 17, 2021
"""

__all__ = [
    "file_path",
    "fix_type_funcs",
    "get_best_model_path",
    "get_experiment_directory",
    "generate_train_config_yaml",
    "generate_predict_config_yaml",
    "none_string_to_none",
    "nonnegative_int",
    "path_to_str",
    "positive_float",
    "positive_int",
    "probability_float",
    "remove_args",
]

from argparse import ArgumentTypeError
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Callable, List, Optional, Union

from jsonargparse import ArgumentParser
import yaml

logger = getLogger(__name__)


class _ParseType:
    @property
    def __name__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__name__


class file_path(_ParseType):
    def __call__(self, string: str) -> Path:
        path = Path(string)
        if not path.is_file():
            msg = f"{string} is not a valid path."
            raise ArgumentTypeError(msg)
        return path


class positive_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num <= 0.0:
            msg = f"{string} needs to be a positive float."
            raise ArgumentTypeError(msg)
        return num


class positive_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num <= 0:
            msg = f"{string} needs to be a positive integer."
            raise ArgumentTypeError(msg)
        return num


class nonnegative_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num < 0:
            msg = f"{string} needs to be a nonnegative integer."
            raise ArgumentTypeError(msg)
        return num


class probability_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num <= 0.0 or num >= 1.0:
            msg = f"{string} needs to be between 0 and 1."
            raise ArgumentTypeError(msg)
        return num


def get_best_model_path(checkpoint_callback, only_best: bool = False) -> Path:
    """ gets the best model path from a ModelCheckpoint instance """
    best_model_path = checkpoint_callback.best_model_path
    if only_best and not best_model_path:
        raise ValueError("best_model_path empty.")
    last_model_path = checkpoint_callback.last_model_path
    model_path = best_model_path or last_model_path
    return Path(model_path).resolve()


def get_experiment_directory(model_path: Path) -> Path:
    """ gets the experiment directory from a checkpoint model path """
    if isinstance(model_path, str):
        model_path = Path(model_path).resolve()
    return model_path.parents[1]


def _generate_config_yaml(
    exp_dir: Path,
    parser: ArgumentParser,
    dict_args: dict,
    best_model_path: Path,
    stage: str,
):
    assert stage in ("train", "predict")
    config = vars(parser.get_defaults())
    for k, v in dict_args.items():
        if k in config:
            config[k] = v
    config_filename = exp_dir / f"{stage}_config.yaml"
    if best_model_path is not None:
        config["model_path"] = str(best_model_path)
    if stage == "predict":
        config["predict_csv"] = "CHANGE ME!"
    with open(config_filename, "w") as f:
        yaml.dump(config, f)
    logger.info(f"{stage} configuration file generated: " f"{config_filename}")


def generate_train_config_yaml(
    exp_dir: Path, parser: ArgumentParser, dict_args: dict, **kwargs
):
    if dict_args["config"] is not None:
        return  # user used config file, so we do not need to generate one
    _generate_config_yaml(exp_dir, parser, dict_args, None, "train")


def generate_predict_config_yaml(
    exp_dir: Path,
    parser: ArgumentParser,
    dict_args: dict,
    best_model_path: Optional[Path] = None,
):
    _generate_config_yaml(exp_dir, parser, dict_args, best_model_path, "predict")


def remove_args(parser: ArgumentParser, args: List[str]):
    """ remove a list of args (w/o leading --) from a parser """
    # https://stackoverflow.com/questions/32807319/disable-remove-argument-in-argparse
    for arg in args:
        for action in parser._actions:
            opt_str = action.option_strings[-1]
            dest = action.dest
            if opt_str[0] == arg or dest == arg:
                parser._remove_action(action)
                break

        for action in parser._action_groups:
            group_actions = action._group_actions
            for group_action in group_actions:
                if group_action.dest == arg:
                    group_actions.remove(group_action)
                    break


# flake8: noqa: E731
def fix_type_funcs(parser):
    """ fixes type functions in pytorch-lightning's `add_argparse_args` """
    for action in parser._actions:
        if action.type is not None:
            type_func_name = action.type.__name__
            if type_func_name.startswith("str_to_"):
                func = deepcopy(action.type)
                action.type = lambda val: func(str(val))
            elif "gpus" in type_func_name:
                action.type = _gpus_allowed_type
            elif action.dest == "progress_bar_refresh_rate":
                action.type = lambda val: val if val is None else int(val)


def _map_attrs(
    args: ArgumentParser, cond: Callable, target: Callable
) -> ArgumentParser:
    """ map attributes to some func of the value if it satisfied a cond """
    attrs = [a for a in dir(args) if not a.startswith("_")]
    for attr in attrs:
        val = getattr(args, attr)
        if cond(val):
            setattr(args, attr, target(val))
    return args


# flake8: noqa: E731
def none_string_to_none(args: ArgumentParser):
    """ goes through an instance of parsed args and maps 'None' -> None """
    cond = lambda val: val == "None"
    target = lambda val: None
    args = _map_attrs(args, cond, target)
    return args


# flake8: noqa: E731
def path_to_str(args):
    """ goes through an instance of parsed args and maps Path -> str """
    cond = lambda val: isinstance(val, Path)
    target = lambda val: str(val)
    args = _map_attrs(args, cond, target)
    return args


def _gpus_allowed_type(val: Union[None, str, float, int]) -> Union[None, float, int]:
    """ replaces pytorch-lightning's version to work w/ parser """
    if val is None:
        return val
    elif isinstance(val, list):
        return [int(v) for v in val]
    elif "," in str(val):
        return str(val)
    else:
        return int(val)
