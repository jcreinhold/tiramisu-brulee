#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.parse

parsing functions for argparse and config files

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 17, 2021
"""

__all__ = [
    "dict_to_csv",
    "fix_type_funcs",
    "get_best_model_path",
    "get_experiment_directory",
    "generate_train_config_yaml",
    "generate_predict_config_yaml",
    "none_string_to_none",
    "parse_unknown_to_dict",
    "path_to_str",
    "remove_args",
]

from copy import deepcopy
from logging import getLogger
from pathlib import Path
from pprint import pformat
from typing import Callable, Dict, IO, Iterable, List, Optional, Union

from jsonargparse import ArgumentParser, Namespace
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml

from tiramisu_brulee.experiment.type import new_parse_type
from tiramisu_brulee.experiment.util import append_num_to_filename

logger = getLogger(__name__)


def get_best_model_path(
    checkpoint_callback: ModelCheckpoint, only_best: bool = False
) -> Path:
    """ gets the best model path from a ModelCheckpoint instance """
    best_model_path = checkpoint_callback.best_model_path
    if only_best and not best_model_path:
        raise ValueError("best_model_path empty.")
    last_model_path = checkpoint_callback.last_model_path
    model_path = best_model_path or last_model_path
    return Path(model_path).resolve()


def get_experiment_directory(model_path: Union[str, Path]) -> Path:
    """ gets the experiment directory from a checkpoint model path """
    if isinstance(model_path, str):
        model_path = Path(model_path).resolve()
    return model_path.parents[1]


def _generate_config_yaml(
    exp_dirs: List[Path],
    parser: ArgumentParser,
    dict_args: dict,
    best_model_paths: Optional[List[Path]],
    stage: str,
) -> None:
    """ generate config yaml file(s) for `stage`, store in experiment dir """
    assert stage in ("train", "predict")
    config = vars(parser.get_defaults())
    for k, v in dict_args.items():
        if k in config:
            if isinstance(v, Path):
                v = str(v)
            config[k] = v
    for exp_dir in exp_dirs:
        config_filename = exp_dir / f"{stage}_config.yaml"
        if config_filename.is_file():
            orig_fn = config_filename
            i = 1
            while config_filename.is_file():
                config_filename = append_num_to_filename(orig_fn, i)
                i += 1
        if best_model_paths is not None:
            config["model_path"] = [str(bmp) for bmp in best_model_paths]
        if stage == "predict":
            config["predict_csv"] = "SET ME!"
        with open(config_filename, "w") as f:
            yaml.dump(config, f)
        logger.info(f"{stage} configuration file generated: {config_filename}")


def generate_train_config_yaml(
    exp_dirs: List[Path],
    parser: ArgumentParser,
    dict_args: dict,
    best_model_paths: Optional[List[Path]] = None,
) -> None:
    """ generate config yaml file(s) for training, store in experiment dir """
    if dict_args["config"] is not None:
        return  # user used config file, so we do not need to generate one
    if isinstance(exp_dirs, Path):
        exp_dirs = [exp_dirs]
    _generate_config_yaml(exp_dirs, parser, dict_args, None, "train")


def generate_predict_config_yaml(
    exp_dirs: List[Path],
    parser: ArgumentParser,
    dict_args: dict,
    best_model_paths: Optional[List[Path]] = None,
) -> None:
    """ generate config yaml file(s) for prediction, store in experiment dir """
    if isinstance(exp_dirs, Path):
        exp_dirs = [exp_dirs]
    remove_args(parser, ["config"])
    _generate_config_yaml(exp_dirs, parser, dict_args, best_model_paths, "predict")


def remove_args(parser: ArgumentParser, args: Iterable[str]) -> None:
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
def fix_type_funcs(parser: ArgumentParser) -> None:
    """ fixes type functions in pytorch-lightning's `add_argparse_args` """
    for action in parser._actions:
        if action.type is not None:
            type_func_name = action.type.__name__
            if type_func_name.startswith("str_to_"):
                func = deepcopy(action.type)
                action.type = new_parse_type(
                    lambda val: func(str(val)), type_func_name,
                )
            elif "gpus" in type_func_name:
                action.type = new_parse_type(_gpus_allowed_type, type_func_name)
            elif action.dest == "progress_bar_refresh_rate":
                action.type = new_parse_type(
                    lambda val: val and int(val), "none_or_int",
                )
            elif action.type.__str__ is object.__str__:
                action.type = new_parse_type(action.type, type_func_name)


def _map_attrs(args: Namespace, cond: Callable, target: Callable) -> Namespace:
    """ map attributes to some func of the value if it satisfied a cond """
    attrs = [a for a in dir(args) if not a.startswith("_")]
    for attr in attrs:
        val = getattr(args, attr)
        if cond(val):
            setattr(args, attr, target(val))
    return args


# flake8: noqa: E731
def none_string_to_none(args: Namespace) -> Namespace:
    """ goes through an instance of parsed args and maps 'None' -> None """
    cond = lambda val: val == "None"
    target = lambda val: None
    args = _map_attrs(args, cond, target)
    return args


# flake8: noqa: E731
def path_to_str(args: Namespace) -> Namespace:
    """ goes through an instance of parsed args and maps Path -> str """
    cond = lambda val: isinstance(val, Path)
    target = lambda val: str(val)
    args = _map_attrs(args, cond, target)
    return args


def _gpus_allowed_type(
    val: Union[None, str, float, int]
) -> Union[None, float, int, List[int], str]:
    """ replaces pytorch-lightning's version to work w/ parser """
    if val is None:
        return val
    elif isinstance(val, list):
        return [int(v) for v in val]
    elif "," in str(val):
        return str(val)
    else:
        return int(val)


def parse_unknown_to_dict(unknown: List[str]) -> dict:
    """ parse unknown arguments (usually modalities and their path) to dict """
    nargs = len(unknown)
    if nargs % 2 != 0:
        raise ValueError(
            "Every modality needs a path. Check for typos in your arguments."
        )
    modality_path = {}
    for i in range(0, nargs, 2):
        modality = unknown[i]
        path = unknown[i + 1]
        if not modality.startswith("--"):
            raise ValueError(
                f"Each modality needs `--` before the name. Received {modality}."
            )
        if path.startswith("-"):
            raise ValueError(
                f"Each path must not contain `-` before the name. Received {path}."
            )
        modality = modality.lstrip("-")
        modality_path[modality] = path
    if "out" not in modality_path:
        msg = "Output path required but not supplied.\n"
        msg += f"Parsed modalities:\n{pformat(modality_path)}"
        raise ValueError(msg)
    return modality_path


def dict_to_csv(modality_path: Dict[str, str], open_file: IO) -> None:
    """
    takes a dictionary of modalities and paths
    (one for each modality) and an open file
    (e.g., open("file.csv", "w")) and writes
    the modalities as headers and the paths as
    entries under those headers

    used for to wrangle single time-point
    prediction into the same interface as
    multi time-point prediction
    """
    headers, paths = ["subject"], ["pred_subj"]
    for modality, path in modality_path.items():
        headers.append(modality)
        paths.append(path)
    headers_str = ",".join(headers) + "\n"
    paths_str = ",".join(paths)
    open_file.write(headers_str)
    open_file.write(paths_str)
    open_file.flush()
