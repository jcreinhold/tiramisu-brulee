#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.tiramisu_brulee.experiment.lesion_seg.cli

command-line interface functions for lesion segmentation with 3D Tiramisu

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 25, 2021
"""

__all__ = [
    "predict",
    "predict_image",
    "train",
]

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from functools import partial
import inspect
import logging
from pathlib import Path
import tempfile
from typing import List, Optional, Tuple, Union
import warnings

import jsonargparse
from jsonargparse import (
    ActionConfigFile,
    Namespace,
)
import nibabel as nib
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from tiramisu_brulee.experiment.lesion_seg.data import (
    Mixup,
    LesionSegDataModulePredict,
    LesionSegDataModuleTrain,
)
from tiramisu_brulee.experiment.lesion_seg.lesion_tools import clean_segmentation
from tiramisu_brulee.experiment.lesion_seg.parse import (
    dict_to_csv,
    file_path,
    fix_type_funcs,
    generate_predict_config_yaml,
    generate_train_config_yaml,
    get_best_model_path,
    get_experiment_directory,
    none_string_to_none,
    parse_unknown_to_dict,
    path_to_str,
    remove_args,
)
from tiramisu_brulee.experiment.lesion_seg.seg import (
    LesionSegLightningTiramisu,
    ModelNum,
)
from tiramisu_brulee.experiment.lesion_seg.util import (
    append_num_to_filename,
    setup_log,
)

ArgType = Optional[Union[Namespace, List[str]]]
Parser = Union[argparse.ArgumentParser, jsonargparse.ArgumentParser]
EXPERIMENT_NAME = "lesion_tiramisu_experiment"

# num of dataloader workers is set to 0 for compatibility w/ torchio, so ignore warning
dataloader_warning = "The dataloader, train dataloader, does not have many workers"
warnings.filterwarnings("ignore", dataloader_warning, category=UserWarning)


def train_parser(use_python_argparse: bool = True) -> Parser:
    """ argument parser for training a 3D Tiramisu CNN """
    if use_python_argparse:
        ArgumentParser = argparse.ArgumentParser
        config_action = None
    else:
        ArgumentParser = jsonargparse.ArgumentParser
        config_action = ActionConfigFile
    desc = "Train a 3D Tiramisu CNN to segment lesions"
    parser = ArgumentParser(prog="lesion-train", description=desc,)
    parser.add_argument(
        "--config",
        action=config_action,
        help="path to a configuration file in json or yaml format",
    )
    exp_parser = parser.add_argument_group("Experiment")
    exp_parser.add_argument(
        "-sd", "--seed", type=int, default=0, help="set seed for reproducibility"
    )
    exp_parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="increase output verbosity (e.g., -vv is more than -v)",
    )
    parser = LesionSegLightningTiramisu.add_model_arguments(parser)
    parser = LesionSegLightningTiramisu.add_training_arguments(parser)
    parser = LesionSegLightningTiramisu.add_other_arguments(parser)
    parser = LesionSegDataModuleTrain.add_arguments(parser)
    parser = Mixup.add_arguments(parser)
    parser = Trainer.add_argparse_args(parser)
    unnecessary_args = {
        "checkpoint_callback",
        "logger",
        "max_steps",
        "min_steps",
        "truncated_bptt_steps",
        "weights_save_path",
    }
    if use_python_argparse:
        unnecessary_args.union({"min_epochs", "max_epochs"})
    else:
        parser.link_arguments("n_epochs", "min_epochs")  # noqa
        parser.link_arguments("n_epochs", "max_epochs")  # noqa
    remove_args(parser, unnecessary_args)
    fix_type_funcs(parser)
    return parser


def train(args: ArgType = None, return_best_model_paths: bool = False) -> int:
    """ train a 3D Tiramisu CNN for segmentation """
    parser = train_parser(False)
    if args is None:
        args = parser.parse_args(_skip_check=True)  # noqa
    elif isinstance(args, list):
        args = parser.parse_args(args, _skip_check=True)  # noqa
    args = none_string_to_none(args)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    seed_everything(args.seed, workers=True)
    if args.default_root_dir is not None:
        root_dir = Path(args.default_root_dir).resolve()
    else:
        root_dir = Path.cwd()
    name = EXPERIMENT_NAME
    args = path_to_str(args)
    n_models_to_train = len(args.train_csv)
    if n_models_to_train != len(args.valid_csv):
        raise ValueError(
            "Number of training and validation CSVs must be equal.\n"
            f"Got {n_models_to_train} != {len(args.valid_csv)}"
        )
    csvs = zip(args.train_csv, args.valid_csv)
    checkpoint_kwargs = dict(
        filename="{epoch}-{val_loss:.3f}-{val_isbi15_score:.3f}",
        monitor="val_isbi15_score",
        save_top_k=3,
        save_last=True,
        mode="max",
        every_n_val_epochs=args.checkpoint_every_n_epochs,
    )
    best_model_paths = []
    dict_args = vars(args)
    use_multigpus = not (args.gpus is None or args.gpus <= 1)
    for i, (train_csv, valid_csv) in enumerate(csvs, 1):
        tb_logger = TensorBoardLogger(str(root_dir), name=name)
        checkpoint_callback = ModelCheckpoint(**checkpoint_kwargs)
        plugins = args.plugins
        if use_multigpus and args.accelerator == "ddp":
            plugins = DDPPlugin(find_unused_parameters=False)
        trainer = Trainer.from_argparse_args(
            args,
            logger=tb_logger,
            checkpoint_callback=True,
            callbacks=[checkpoint_callback],
            plugins=plugins,
        )
        nth_model = f" ({i}/{n_models_to_train})"
        dict_args["train_csv"] = train_csv
        dict_args["valid_csv"] = valid_csv
        dm = LesionSegDataModuleTrain.from_csv(**dict_args)
        dm.setup()
        model = LesionSegLightningTiramisu(**dict_args)
        logger.debug(model)
        if args.auto_scale_batch_size or args.auto_lr_find:
            tuning_output = trainer.tune(model, datamodule=dm)
            msg = f"Tune output{nth_model}:\n{tuning_output}"
            logger.info(msg)
        trainer.fit(model, datamodule=dm)
        best_model_path = get_best_model_path(checkpoint_callback)
        best_model_paths.append(best_model_path)
        msg = f"Best model path: {best_model_path}" + nth_model + "\n"
        msg += "Finished training" + nth_model
        logger.info(msg)
        del dm, model, trainer, tb_logger, checkpoint_callback
    if not args.fast_dev_run:
        exp_dirs = []
        for bmp in best_model_paths:
            exp_dirs.append(get_experiment_directory(bmp))
        config_kwargs = dict(
            exp_dirs=exp_dirs, dict_args=dict_args, best_model_paths=best_model_paths,
        )
        generate_train_config_yaml(**config_kwargs, parser=parser)
        generate_predict_config_yaml(**config_kwargs, parser=predict_parser(False))
    if return_best_model_paths:
        return best_model_paths
    else:
        return 0


def _predict_parser_shared(
    parser: Parser, necessary_trainer_args: set, add_csv: bool
) -> Parser:
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
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="increase output verbosity (e.g., -vv is more than -v)",
    )
    parser = LesionSegLightningTiramisu.add_other_arguments(parser)
    parser = LesionSegLightningTiramisu.add_testing_arguments(parser)
    parser = LesionSegDataModulePredict.add_arguments(parser, add_csv=add_csv)
    parser = Trainer.add_argparse_args(parser)
    trainer_args = set(inspect.signature(Trainer).parameters.keys())  # noqa
    unnecessary_args = trainer_args - necessary_trainer_args
    remove_args(parser, unnecessary_args)
    fix_type_funcs(parser)
    return parser


def predict_parser(use_python_argparse: bool = True) -> Parser:
    """ argument parser for using a 3D Tiramisu CNN for prediction """
    if use_python_argparse:
        ArgumentParser = argparse.ArgumentParser
        config_action = None
    else:
        ArgumentParser = jsonargparse.ArgumentParser
        config_action = ActionConfigFile
    desc = "Use a Tiramisu CNN to segment lesions"
    parser = ArgumentParser(prog="lesion-predict", description=desc)
    parser.add_argument(
        "--config",
        action=config_action,
        help="path to a configuration file in json or yaml format",
    )
    necessary_trainer_args = {
        "benchmark",
        "fast_dev_run",
        "gpus",
        "precision",
    }
    parser = _predict_parser_shared(parser, necessary_trainer_args, True)
    return parser


def predict_image_parser() -> argparse.ArgumentParser:
    """ argument parser for using a 3D Tiramisu CNN for single-timepoint prediction """
    desc = "Use a Tiramisu CNN to segment lesions for a single-timepoint prediction"
    parser = argparse.ArgumentParser(prog="lesion-predict-image", description=desc)
    necessary_trainer_args = {
        "fast_dev_run",
        "gpus",
        "precision",
    }
    parser = _predict_parser_shared(parser, necessary_trainer_args, False)
    return parser


def _aggregate(
    n_fn: Tuple[int, str],
    threshold: float,
    n_models: int,
    n_fns: int,
    fill_holes: bool,
    min_lesion_size: int,
):
    """ aggregate helper for concurrent/parallel processing """
    n, fn = n_fn
    data = []
    for i in range(1, n_models + 1):
        _fn = append_num_to_filename(fn, i)
        nib_image = nib.load(_fn)
        data.append(nib_image.get_fdata())
    agg = np.mean(data, axis=0) > threshold
    agg = clean_segmentation(agg, fill_holes, min_lesion_size)
    agg = agg.astype(np.float32)
    nib.Nifti1Image(agg, nib_image.affine).to_filename(fn)  # noqa
    logging.info(f"Save aggregated prediction: {fn} ({n}/{n_fns})")


def aggregate(
    predict_csv: str,
    n_models: int,
    threshold: float = 0.5,
    fill_holes: bool = False,
    min_lesion_size: int = 3,
    num_workers: Optional[int] = None,
):
    """ aggregate output from multiple model predictions """
    csv = pd.read_csv(predict_csv)
    out_fns = csv["out"]
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


def _predict(args: Namespace, parser: Parser, use_multiprocessing: bool):
    args = none_string_to_none(args)
    args = path_to_str(args)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    seed_everything(args.seed, workers=True)
    n_models = len(args.model_path)
    dict_args = vars(args)
    pp = dict_args["predict_probability"] = n_models > 1
    for i, model_path in enumerate(args.model_path, 1):
        model_num = ModelNum(num=i, out_of=n_models)
        nth_model = f" ({i}/{n_models})"
        trainer = Trainer.from_argparse_args(args)
        dm = LesionSegDataModulePredict.from_csv(**dict_args)
        dm.setup()
        model = LesionSegLightningTiramisu.load_from_checkpoint(
            model_path, predict_probability=pp, _model_num=model_num,
        )
        logger.debug(model)
        trainer.predict(model, datamodule=dm)
        logger.info("Finished prediction" + nth_model)
        del dm, model, trainer
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
    if not args.fast_dev_run:
        exp_dirs = []
        for mp in args.model_path:
            exp_dirs.append(get_experiment_directory(mp))
        generate_predict_config_yaml(exp_dirs, parser, dict_args)


def predict(args: ArgType = None) -> int:
    """ use a 3D Tiramisu CNN for prediction """
    parser = predict_parser(False)
    if args is None:
        args = parser.parse_args(_skip_check=True)  # noqa
    elif isinstance(args, list):
        args = parser.parse_args(args, _skip_check=True)  # noqa
    _predict(args, parser, True)
    return 0


def predict_image(args: ArgType = None) -> int:
    """ use a 3D Tiramisu CNN for prediction for a single-timepoint """
    parser = predict_image_parser()
    if args is None:
        args, unknown = parser.parse_known_args()
    elif isinstance(args, list):
        args, unknown = parser.parse_known_args(args)
    else:
        raise ValueError("input args must be None or a list of strings to parse")
    modality_paths = parse_unknown_to_dict(unknown)
    with tempfile.NamedTemporaryFile("w") as f:
        dict_to_csv(modality_paths, f)  # noqa
        args.predict_csv = f.name
        _predict(args, parser, False)
    return 0
