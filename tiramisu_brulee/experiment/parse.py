"""Parsing functions for argparse and config files
Author: Jacob Reinhold <jcreinhold@gmail.com>
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

import builtins
import copy
import logging
import pathlib
import pprint
import typing
import warnings

import yaml
from jsonargparse import ArgumentParser, Namespace
from pytorch_lightning.callbacks import ModelCheckpoint

from tiramisu_brulee.experiment.type import PathLike, new_parse_type
from tiramisu_brulee.experiment.util import append_num_to_filename

logger = logging.getLogger(__name__)


def get_best_model_path(
    checkpoint_callback: ModelCheckpoint, only_best: builtins.bool = False
) -> pathlib.Path:
    """gets the best model path from a ModelCheckpoint instance"""
    best_model_path = checkpoint_callback.best_model_path
    if only_best and not best_model_path:
        raise ValueError("best_model_path empty.")
    last_model_path = checkpoint_callback.last_model_path
    model_path = best_model_path or last_model_path
    return pathlib.Path(model_path).resolve()


def get_experiment_directory(model_path: PathLike) -> pathlib.Path:
    """gets the experiment directory from a checkpoint model path"""
    if not isinstance(model_path, pathlib.Path):
        model_path = pathlib.Path(model_path).resolve()
    return model_path.parents[1]


def _generate_config_yaml(
    exp_dirs: typing.List[pathlib.Path],
    parser: ArgumentParser,
    dict_args: dict,
    best_model_paths: typing.Optional[typing.List[pathlib.Path]],
    stage: builtins.str,
) -> typing.List[builtins.str]:
    """generate config yaml file(s) for `stage`, store in experiment dir"""
    assert stage in ("train", "predict")
    config = vars(parser.get_defaults())
    for k, v in dict_args.items():
        if k in config:
            if isinstance(v, pathlib.Path):
                v = str(v)
            config[k] = v
    config_filenames = []
    for exp_dir in exp_dirs:
        config_filename = exp_dir / f"{stage}_config.yaml"
        if config_filename.is_file():
            orig_fn = config_filename
            i = 1
            while config_filename.is_file():
                config_filename = append_num_to_filename(orig_fn, num=i)
                i += 1
        if best_model_paths is not None:
            config["model_path"] = [str(bmp) for bmp in best_model_paths]
        if stage == "predict":
            config["predict_csv"] = "SET ME!"
        with open(config_filename, "w") as f:
            yaml.dump(config, f)
        logger.info(f"{stage} configuration file generated: {config_filename}")
        config_filenames.append(config_filename)
    return [str(fn.resolve()) for fn in config_filenames]


def generate_train_config_yaml(
    exp_dirs: typing.List[pathlib.Path],
    parser: ArgumentParser,
    dict_args: dict,
    best_model_paths: typing.Optional[typing.List[pathlib.Path]] = None,
) -> typing.List[builtins.str]:
    """generate config yaml file(s) for training, store in experiment dir"""
    if dict_args["config"] is not None:
        return []  # user used config file, so we do not need to generate one
    if isinstance(exp_dirs, pathlib.Path):
        exp_dirs = [exp_dirs]
    fns = _generate_config_yaml(exp_dirs, parser, dict_args, None, "train")
    return fns


def generate_predict_config_yaml(
    exp_dirs: typing.List[pathlib.Path],
    parser: ArgumentParser,
    dict_args: dict,
    best_model_paths: typing.Optional[typing.List[pathlib.Path]] = None,
) -> typing.List[builtins.str]:
    """generate config yaml file(s) for prediction, store in experiment dir"""
    if isinstance(exp_dirs, pathlib.Path):
        exp_dirs = [exp_dirs]
    remove_args(parser, ["config"])
    bmps = best_model_paths
    fns = _generate_config_yaml(exp_dirs, parser, dict_args, bmps, "predict")
    return fns


def remove_args(parser: ArgumentParser, args: typing.Iterable[builtins.str]) -> None:
    """remove a set of args from a parser"""
    # https://stackoverflow.com/questions/32807319/disable-remove-argument-in-argparse
    # for some reason, going through the actions once and checking if each action is in
    # the set of args doesn't work & the action/groups need to be removed at same time

    _action_args = set(args)
    _action_group_args = set(args)

    for arg in args:
        for action in parser._actions:
            opt_str = action.option_strings[-1]
            dest = action.dest
            if opt_str == arg or dest == arg:
                parser._remove_action(action)
                _action_args.remove(arg)
                break

        for action in parser._action_groups:
            group_actions = action._group_actions
            for group_action in group_actions:
                if group_action.dest == arg:
                    group_actions.remove(group_action)
                    _action_group_args.remove(group_action.dest)
                    break

    # ignore the inability to remove the "callbacks" argument b/c irrelevant
    intersection = _action_args.intersection(_action_group_args) - {"callbacks"}
    if intersection:
        warnings.warn(f"unable to remove {intersection}")


# flake8: noqa: E731
def fix_type_funcs(parser: ArgumentParser) -> None:
    """fixes type functions in pytorch-lightning's `add_argparse_args`"""
    for action in parser._actions:
        if action.type is not None:
            type_func_name = action.type.__name__
            if type_func_name.startswith("str_to_"):
                func = copy.deepcopy(action.type)
                action.type = new_parse_type(
                    lambda val: func(str(val)),
                    type_func_name,
                )
            elif "gpus" in type_func_name:
                action.type = new_parse_type(_gpus_allowed_type, type_func_name)
            elif action.dest == "progress_bar_refresh_rate":
                action.type = new_parse_type(
                    lambda val: val and int(val),
                    "none_or_int",
                )
            elif action.type.__str__ is object.__str__:
                action.type = new_parse_type(action.type, type_func_name)


def _map_attrs(
    args: Namespace, cond: typing.Callable, target: typing.Callable
) -> Namespace:
    """map attributes to some func of the value if it satisfied a cond"""
    attrs = [a for a in dir(args) if not a.startswith("_")]
    for attr in attrs:
        val = getattr(args, attr)
        if cond(val):
            setattr(args, attr, target(val))
    return args


# flake8: noqa: E731
def none_string_to_none(args: Namespace) -> Namespace:
    """goes through an instance of parsed args and maps 'None' -> None"""
    cond = lambda val: val == "None"
    target = lambda val: None
    args = _map_attrs(args, cond, target)
    return args


# flake8: noqa: E731
def path_to_str(args: Namespace) -> Namespace:
    """goes through an instance of parsed args and maps Path -> str"""
    cond = lambda val: isinstance(val, pathlib.Path)
    target = lambda val: str(val)
    args = _map_attrs(args, cond, target)
    return args


def _gpus_allowed_type(
    val: typing.Union[None, builtins.str, builtins.float, builtins.int]
) -> typing.Union[
    None, builtins.float, builtins.int, typing.List[builtins.int], builtins.str
]:
    """replaces pytorch-lightning's version to work w/ parser"""
    if val is None:
        return val
    elif isinstance(val, list):
        return [int(v) for v in val]
    elif "," in str(val):
        return str(val)
    else:
        return int(val)


def parse_unknown_to_dict(
    unknown: typing.List[builtins.str], *, names_only: builtins.bool = False
) -> typing.Dict[builtins.str, typing.Optional[builtins.str]]:
    """parse unknown arguments (usually modalities and their path) to dict"""
    nargs = len(unknown)
    if nargs % 2 != 0 and not names_only:
        msg = "Every modality needs a path. Check for typos in your arguments."
        raise ValueError(msg)
    modality_path: typing.Dict[builtins.str, typing.Optional[builtins.str]] = {}
    step = 1 if names_only else 2
    for i in range(0, nargs, step):
        modality = unknown[i]
        if not modality.startswith("--"):
            msg = f"Each modality needs `--` before the name. Received {modality}."
            raise ValueError(msg)
        modality = modality.lstrip("-")
        if not names_only:
            path = unknown[i + 1]
            if path.startswith("-"):
                msg = f"Paths cannot contain `-` before the name. Received {path}."
                raise ValueError(msg)
            modality_path[modality] = path
        else:
            modality_path[modality] = None
    if "out" not in modality_path and not names_only:
        msg = "Output path required but not supplied.\n"
        msg += f"Parsed modalities:\n{pprint.pformat(modality_path)}"
        raise ValueError(msg)
    return modality_path


def dict_to_csv(
    modality_path: typing.Dict[builtins.str, builtins.str], open_file: typing.IO
) -> None:
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
    paths_str = ",".join('"{}"'.format(p.strip('"')) for p in paths) + "\n"
    logger.debug(f"{headers_str}{paths_str}")
    open_file.write(headers_str)
    open_file.write(paths_str)
    open_file.flush()
