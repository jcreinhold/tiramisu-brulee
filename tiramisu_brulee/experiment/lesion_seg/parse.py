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

from argparse import ArgumentParser, ArgumentTypeError
from logging import getLogger
from pathlib import Path
import subprocess
from typing import List, Optional

import yaml

logger = getLogger(__name__)


class _ParseType:
    def __str__(self):
        return self.__class__.__name__


class file_path(_ParseType):
    def __call__(self, string: str) -> Path:
        path = Path(string)
        if not path.is_file():
            raise ArgumentTypeError(f'{string} is not a valid path.')
        return path


class positive_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num <= 0.:
            raise ArgumentTypeError(f'{string} needs to be a positive float.')
        return num


class positive_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num <= 0:
            raise ArgumentTypeError(f'{string} needs to be a positive integer.')
        return num


class nonnegative_int(_ParseType):
    def __call__(self, string: str) -> int:
        num = int(string)
        if num < 0:
            raise ArgumentTypeError(f'{string} needs to be a nonnegative integer.')
        return num


class probability_float(_ParseType):
    def __call__(self, string: str) -> float:
        num = float(string)
        if num <= 0. or num >= 1.:
            raise ArgumentTypeError(f'{string} needs to be between 0 and 1.')
        return num


def get_best_model_path(checkpoint_callback) -> Path:
    """ gets the best model path from a ModelCheckpoint instance """
    best_model_path = checkpoint_callback.best_model_path
    return Path(best_model_path).resolve()


def get_experiment_directory(model_path: Path) -> Path:
    """ gets the experiment directory from a checkpoint model path """
    return model_path.parents[1]


def _generate_config_yaml(exp_dir: Path,
                          dict_args: dict,
                          best_model_path: Path,
                          stage: str):
    cmd = f"lesion-{stage} --print_config",
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0:
        config = yaml.unsafe_load(result.stdout)
        for k, v in dict_args:
            if k in config:
                config[k] = v
        config_filename = exp_dir / f'{stage}_config.yaml'
        if best_model_path is not None:
            config['model_path'] = str(best_model_path)
        with open(config_filename, 'w') as f:
            yaml.dump(config, f)
        logger.info(
            f'{stage} configuration file generated: '
            f'{config_filename}'
        )
    else:
        logger.info(f'Could not generate a {stage} configuration file.')
        logger.info(f'Command ({cmd}) failed with retcode: {result.returncode}.')
        logger.debug(f'stdout:\n{result.stdout}')
        logger.debug(f'stderr:\n{result.stderr}')


def generate_train_config_yaml(exp_dir: Path,
                               dict_args: dict):
    if dict_args['config'] is not None:
        return  # user used config file, so we do not need to generate one
    _generate_config_yaml(exp_dir, dict_args, None, 'train')


def generate_predict_config_yaml(exp_dir: Path,
                                 dict_args: dict,
                                 best_model_path: Optional[Path] = None):
    _generate_config_yaml(exp_dir, dict_args, best_model_path, 'predict')


def remove_args(parser: ArgumentParser, args: List[str]):
    """ remove a list of args (w/o leading --) from a parser """
    # https://stackoverflow.com/questions/32807319/disable-remove-argument-in-argparse
    for arg in args:
        for action in parser._actions:
            if ((vars(action)['option_strings']
                 and vars(action)['option_strings'][0] == arg)
                 or vars(action)['dest'] == arg):
                parser._remove_action(action)

        for action in parser._action_groups:
            vars_action = vars(action)
            var_group_actions = vars_action['_group_actions']
            for x in var_group_actions:
                if x.dest == arg:
                    var_group_actions.remove(x)
                    break
