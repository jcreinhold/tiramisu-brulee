#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.cli.predict

command-line interface functions for predicting
lesion segmentations with Tiramisu neural network

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 25, 2021
"""

__all__ = [
    "predict",
    "predict_image",
]

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from functools import partial
import gc
import inspect
import logging
from pathlib import Path
import tempfile
from typing import Optional, Tuple, Union

import jsonargparse
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
import torch
import torchio as tio

from tiramisu_brulee.experiment.type import (
    ArgType,
    file_path,
    Namespace,
    ArgParser,
)
from tiramisu_brulee.experiment.data import (
    csv_to_subjectlist,
    LesionSegDataModulePredictBase,
    LesionSegDataModulePredictPatches,
    LesionSegDataModulePredictWhole,
)
from tiramisu_brulee.experiment.lesion_tools import clean_segmentation
from tiramisu_brulee.experiment.parse import (
    dict_to_csv,
    fix_type_funcs,
    generate_predict_config_yaml,
    get_experiment_directory,
    none_string_to_none,
    parse_unknown_to_dict,
    path_to_str,
    remove_args,
)
from tiramisu_brulee.experiment.seg import LesionSegLightningTiramisu
from tiramisu_brulee.experiment.type import ModelNum
from tiramisu_brulee.experiment.util import (
    append_num_to_filename,
    setup_log,
)
from tiramisu_brulee.experiment.cli.common import (
    check_patch_size,
    handle_fast_dev_run,
    pseudo3d_dims_setup,
)


def predict_parser(use_python_argparse: bool = True) -> ArgParser:
    """ argument parser for using a Tiramisu CNN for prediction """
    if use_python_argparse:
        ArgumentParser = argparse.ArgumentParser
        config_action = None
    else:
        ArgumentParser = jsonargparse.ArgumentParser
        config_action = jsonargparse.ActionConfigFile
    desc = "Use a Tiramisu CNN to segment lesions"
    parser = ArgumentParser(prog="lesion-predict", description=desc)
    parser.add_argument(
        "--config",
        action=config_action,  # type: ignore[arg-type]
        help="path to a configuration file in json or yaml format",
    )
    necessary_trainer_args = {
        "benchmark",
        "gpus",
        "precision",
        "progress_bar_refresh_rate",
    }
    parser = _predict_parser_shared(parser, necessary_trainer_args, True)
    return parser


def predict_image_parser() -> argparse.ArgumentParser:
    """ argument parser for using a Tiramisu CNN for single time-point prediction """
    desc = "Use a Tiramisu CNN to segment lesions for a single time-point prediction"
    parser = argparse.ArgumentParser(prog="lesion-predict-image", description=desc)
    necessary_trainer_args = {
        "benchmark",
        "gpus",
        "precision",
        "progress_bar_refresh_rate",
    }
    parser = _predict_parser_shared(parser, necessary_trainer_args, False)
    return parser


def _predict_parser_shared(
    parser: ArgParser, necessary_trainer_args: set, add_csv: bool,
) -> ArgParser:
    exp_parser = parser.add_argument_group("Experiment")
    exp_parser.add_argument(
        "-mp",
        "--model-path",
        type=file_path(),
        nargs="+",
        required=True,
        default=["SET ME!"],
        help="path to output the trained model",
    )
    exp_parser.add_argument(
        "-sd", "--seed", type=int, default=0, help="set seed for reproducibility",
    )
    exp_parser.add_argument(
        "-oa",
        "--only-aggregate",
        action="store_true",
        default=False,
        help="only aggregate results (useful to test different thresholds)",
    )
    exp_parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="increase output verbosity (e.g., -vv is more than -v)",
    )
    parser = LesionSegLightningTiramisu.add_other_arguments(parser)
    parser = LesionSegLightningTiramisu.add_testing_arguments(parser)
    parser = LesionSegDataModulePredictBase.add_arguments(parser, add_csv=add_csv)
    parser = Trainer.add_argparse_args(parser)
    trainer_args = set(inspect.signature(Trainer).parameters.keys())
    unnecessary_args = trainer_args - necessary_trainer_args
    unnecessary_args = handle_fast_dev_run(unnecessary_args)
    remove_args(parser, unnecessary_args)
    fix_type_funcs(parser)
    return parser


def predict(args: ArgType = None) -> int:
    """ use a Tiramisu CNN for prediction """
    parser = predict_parser(False)
    if args is None:
        args = parser.parse_args(_skip_check=True)  # type: ignore[call-overload]
    elif isinstance(args, list):
        args = parser.parse_args(args, _skip_check=True)  # type: ignore[call-overload]
    _predict(args, parser, True)
    return 0


def predict_image(args: ArgType = None) -> int:
    """ use a Tiramisu CNN for prediction for a single time-point """
    parser = predict_image_parser()
    if args is None:
        args, unknown = parser.parse_known_args()
    elif isinstance(args, list):
        args, unknown = parser.parse_known_args(args)
    else:
        raise ValueError("input args must be None or a list of strings to parse")
    modality_paths = parse_unknown_to_dict(unknown)
    with tempfile.NamedTemporaryFile("w") as f:
        dict_to_csv(modality_paths, f)
        args.predict_csv = f.name
        _predict(args, parser, False)
    return 0


def _predict_whole_image(
    args: Namespace, model_path: Path, model_num: ModelNum,
) -> None:
    """ predict a whole image volume as opposed to patches """
    dict_args = vars(args)
    pp = args.predict_probability
    trainer = Trainer.from_argparse_args(args)
    dm = LesionSegDataModulePredictWhole.from_csv(**dict_args)
    dm.setup()
    model = LesionSegLightningTiramisu.load_from_checkpoint(
        str(model_path), predict_probability=pp, _model_num=model_num,
    )
    logging.debug(model)
    trainer.predict(model, datamodule=dm)
    # kill multiprocessing workers for next iteration
    dm.teardown()
    trainer.teardown()


def _predict_patch_image(
    args: Namespace,
    model_path: Path,
    model_num: ModelNum,
    pseudo3d_dim: Union[None, int],
) -> None:
    """ predict a volume with patches as opposed to a whole volume """
    dict_args = vars(args)
    dict_args["pseudo3d_dim"] = pseudo3d_dim
    pp = args.predict_probability
    subject_list = csv_to_subjectlist(args.predict_csv)
    model = LesionSegLightningTiramisu.load_from_checkpoint(
        str(model_path), predict_probability=pp, _model_num=model_num,
    )
    logging.debug(model)
    for subject in subject_list:
        trainer = Trainer.from_argparse_args(args)
        dm = LesionSegDataModulePredictPatches(subject, **dict_args)
        dm.setup()
        trainer.predict(model, datamodule=dm)
        # kill multiprocessing workers, free memory for the next iteration
        dm.teardown()
        trainer.teardown()
        del dm, trainer
        gc.collect()


def _predict(args: Namespace, parser: ArgParser, use_multiprocessing: bool) -> None:
    args = none_string_to_none(args)
    args = path_to_str(args)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    seed_everything(args.seed, workers=True)
    n_models = len(args.model_path)
    if not args.only_aggregate:
        args.predict_probability = n_models > 1 or args.predict_probability
        patch_predict = args.patch_size is not None
        use_pseudo3d = args.pseudo3d_dim is not None
        if patch_predict:
            check_patch_size(args.patch_size, use_pseudo3d)
        pseudo3d_dims = pseudo3d_dims_setup(args.pseudo3d_dim, n_models, "predict")
        predict_iter_data = zip(args.model_path, pseudo3d_dims)
        for i, (model_path, p3d) in enumerate(predict_iter_data, 1):
            model_num = ModelNum(num=i, out_of=n_models)
            nth_model = f" ({i}/{n_models})"
            if patch_predict:
                _predict_patch_image(args, model_path, model_num, p3d)
            else:
                _predict_whole_image(args, model_path, model_num)
            logger.info("Finished prediction" + nth_model)
            # force garbage collection for the next iteration
            gc.collect()
            torch.cuda.empty_cache()
    if n_models > 1:
        num_workers = args.num_workers if use_multiprocessing else 0
        aggregate(
            args.predict_csv,
            n_models,
            args.threshold,
            args.fill_holes,
            args.min_lesion_size,
            num_workers,
        )
    _generate_config_yamls_in_predict(args, parser)


def _generate_config_yamls_in_predict(args: ArgType, parser: ArgParser) -> None:
    assert isinstance(args, (argparse.Namespace, jsonargparse.Namespace))
    is_fast_dev_run = args.fast_dev_run if hasattr(args, "fast_dev_run") else False
    if (
        not is_fast_dev_run
        and not args.only_aggregate
        and hasattr(parser, "get_defaults")
    ):
        exp_dirs = []
        for mp in args.model_path:
            exp_dirs.append(get_experiment_directory(mp))
        generate_predict_config_yaml(exp_dirs, parser, vars(args))


def aggregate(
    predict_csv: str,
    n_models: int,
    threshold: float = 0.5,
    fill_holes: bool = False,
    min_lesion_size: int = 3,
    num_workers: Optional[int] = None,
) -> None:
    """ aggregate output from multiple model predictions """
    df = pd.read_csv(predict_csv)
    out_fns = df["out"]
    n_fns = len(out_fns)
    out_fn_iter = enumerate(out_fns, 1)
    _aggregator = partial(
        _aggregate,
        threshold=threshold,
        n_models=n_models,
        n_fns=n_fns,
        fill_holes=fill_holes,
        min_lesion_size=min_lesion_size,
    )
    use_multiprocessing = num_workers is None or num_workers > 0
    if use_multiprocessing:
        with ProcessPoolExecutor(num_workers) as executor:
            executor.map(_aggregator, out_fn_iter)
    else:
        deque(map(_aggregator, out_fn_iter), maxlen=0)


# noinspection PyUnboundLocalVariable
def _aggregate(
    n_fn: Tuple[int, str],
    threshold: float,
    n_models: int,
    n_fns: int,
    fill_holes: bool,
    min_lesion_size: int,
) -> None:
    """ aggregate helper for concurrent/parallel processing """
    assert n_models >= 1
    n, fn = n_fn
    data = []
    for i in range(1, n_models + 1):
        _fn = append_num_to_filename(fn, i)
        image = tio.ScalarImage(_fn)
        array = image.numpy()
        data.append(array.squeeze())
    agg = np.mean(data, axis=0) > threshold
    agg = clean_segmentation(agg, fill_holes, min_lesion_size)
    agg = agg.astype(array.dtype)
    image.set_data(agg[np.newaxis])
    image.save(fn)
    logging.info(f"Save aggregated prediction: {fn} ({n}/{n_fns})")
