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
    'file_path',
    'get_best_model_path',
    'get_experiment_directory',
    'generate_train_config_yaml',
    'generate_predict_config_yaml',
    'nonnegative_int',
    'positive_float',
    'positive_int',
    'probability_float',
    'remove_args',
]

from argparse import ArgumentTypeError
from logging import getLogger
from pathlib import Path
from typing import List, Optional

from jsonargparse import ArgumentParser
import yaml

logger = getLogger(__name__)


class _ParseType:
    def __str__(self):
        return self.__class__.__name__


class file_path(_ParseType):
    def __call__(self, string: str) -> Path:
        path = Path(string)
        if not path.is_file():
            msg = f'{string} is not a valid path.'
            raise ArgumentTypeError(msg)
        return path


class positive_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num <= 0.:
            msg = f'{string} needs to be a positive float.'
            raise ArgumentTypeError(msg)
        return num


class positive_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num <= 0:
            msg = f'{string} needs to be a positive integer.'
            raise ArgumentTypeError(msg)
        return num


class nonnegative_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num < 0:
            msg = f'{string} needs to be a nonnegative integer.'
            raise ArgumentTypeError(msg)
        return num


class probability_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num <= 0. or num >= 1.:
            msg = f'{string} needs to be between 0 and 1.'
            raise ArgumentTypeError(msg)
        return num


def get_best_model_path(
    checkpoint_callback,
    only_best: bool = False
) -> Path:
    """ gets the best model path from a ModelCheckpoint instance """
    best_model_path = checkpoint_callback.best_model_path
    if only_best and not best_model_path:
        raise ValueError('best_model_path empty.')
    last_model_path = checkpoint_callback.last_model_path
    model_path = best_model_path or last_model_path
    return Path(model_path).resolve()


def get_experiment_directory(model_path: Path) -> Path:
    """ gets the experiment directory from a checkpoint model path """
    return model_path.parents[1]


def _generate_config_yaml(
    exp_dir: Path,
    parser: ArgumentParser,
    dict_args: dict,
    best_model_path: Path,
    stage: str
):
    assert stage in ('train', 'predict')
    config = vars(parser.get_defaults())
    for k, v in dict_args.items():
        if k in config:
            config[k] = v
    config_filename = exp_dir / f'{stage}_config.yaml'
    if best_model_path is not None:
        config['model_path'] = str(best_model_path)
    if stage == 'predict':
        config['predict_csv'] = 'CHANGE ME!'
    with open(config_filename, 'w') as f:
        yaml.dump(config, f)
    logger.info(
        f'{stage} configuration file generated: '
        f'{config_filename}'
    )


def generate_train_config_yaml(
    exp_dir: Path,
    parser: ArgumentParser,
    dict_args: dict,
    **kwargs
):
    if dict_args['config'] is not None:
        return  # user used config file, so we do not need to generate one
    _generate_config_yaml(exp_dir, parser, dict_args, None, 'train')


def generate_predict_config_yaml(
    exp_dir: Path,
    parser: ArgumentParser,
    dict_args: dict,
    best_model_path: Optional[Path] = None
):
    _generate_config_yaml(
        exp_dir,
        parser,
        dict_args,
        best_model_path,
        'predict'
    )


def remove_args(parser: ArgumentParser, args: List[str]):
    """ remove a list of args (w/o leading --) from a parser """
    # https://stackoverflow.com/questions/32807319/disable-remove-argument-in-argparse
    for arg in args:
        for action in parser._actions:
            action_dict = vars(action)
            opt_str = action_dict['option_strings'][0]
            dest = action_dict['dest']
            if opt_str[0] == arg or dest == arg:
                parser._remove_action(action)
                break

        for action in parser._action_groups:
            action_dict = vars(action)
            group_actions = action_dict['_group_actions']
            for group_action in group_actions:
                if group_action.dest == arg:
                    group_actions.remove(group_action)
                    break
