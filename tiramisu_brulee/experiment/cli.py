#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.cli

command-line interface functions for lesion
segmentation with Tiramisu neural network

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
import nibabel as nib
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from tiramisu_brulee.experiment.type import (
    ArgType,
    file_path,
    Namespace,
    ArgParser,
)
from tiramisu_brulee.experiment.data import (
    csv_to_subjectlist,
    Mixup,
    LesionSegDataModulePredictBase,
    LesionSegDataModulePredictPatches,
    LesionSegDataModulePredictWhole,
    LesionSegDataModuleTrain,
)
from tiramisu_brulee.experiment.lesion_tools import clean_segmentation
from tiramisu_brulee.experiment.parse import (
    dict_to_csv,
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
from tiramisu_brulee.experiment.seg import LesionSegLightningTiramisu
from tiramisu_brulee.experiment.type import ModelNum
from tiramisu_brulee.experiment.util import (
    append_num_to_filename,
    setup_log,
)

EXPERIMENT_NAME = "lesion_tiramisu_experiment"

# num of dataloader workers is set to 0 for compatibility w/ torchio, so ignore warning
train_dataloader_warning = (
    "The dataloader, train dataloader, does not have many workers"
)
val_dataloader_warning = "The dataloader, val dataloader 0, does not have many workers"
warnings.filterwarnings("ignore", train_dataloader_warning, category=UserWarning)


def train_parser(use_python_argparse: bool = True) -> ArgParser:
    """ argument parser for training a Tiramisu CNN """
    if use_python_argparse:
        ArgumentParser = argparse.ArgumentParser
        config_action = None
    else:
        ArgumentParser = jsonargparse.ArgumentParser
        config_action = jsonargparse.ActionConfigFile
    desc = "Train a Tiramisu CNN to segment lesions"
    parser = ArgumentParser(prog="lesion-train", description=desc,)
    parser.add_argument(
        "--config",
        action=config_action,
        help="path to a configuration file in json or yaml format",
    )
    exp_parser = parser.add_argument_group("Experiment")
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
    parser = LesionSegLightningTiramisu.add_io_arguments(parser)
    parser = LesionSegLightningTiramisu.add_model_arguments(parser)
    parser = LesionSegLightningTiramisu.add_other_arguments(parser)
    parser = LesionSegLightningTiramisu.add_training_arguments(parser)
    parser = LesionSegDataModuleTrain.add_arguments(parser)
    parser = Mixup.add_arguments(parser)
    parser = Trainer.add_argparse_args(parser)
    unnecessary_args = {
        "checkpoint_callback",
        "distributed_backend",
        "in_channels",
        "logger",
        "max_steps",
        "min_steps",
        "out_channels",
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
    """ train a Tiramisu CNN for segmentation """
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
    use_pseudo3d = args.pseudo3d_dim is not None
    if use_pseudo3d:
        warnings.filterwarnings("ignore", val_dataloader_warning, category=UserWarning)
    _check_patch_size(args.patch_size, use_pseudo3d)
    pseudo3d_dims = _pseudo3d_dims_setup(args.pseudo3d_dim, n_models_to_train, "train")
    dict_args["network_dim"] = 2 if use_pseudo3d else 3
    use_multigpus = not (args.gpus is None or args.gpus <= 1)
    train_iter_data = zip(args.train_csv, args.valid_csv, pseudo3d_dims)
    for i, (train_csv, valid_csv, p3d) in enumerate(train_iter_data, 1):
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
        channels_per_image = args.pseudo3d_size if use_pseudo3d else 1
        dict_args["in_channels"] = args.num_input * channels_per_image
        dict_args["out_channels"] = args.num_output
        dict_args["pseudo3d_dim"] = p3d
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


def _pseudo3d_dims_setup(
    pseudo3d_dim: Union[None, List[int]], n_models: int, stage: str
) -> Union[List[None], List[int]]:
    assert stage in ("train", "predict")
    if stage == "predict":
        stage = "us"
    n_p3d = 0 if pseudo3d_dim is None else len(pseudo3d_dim)
    if n_p3d == 1:
        pseudo3d_dims = pseudo3d_dim * n_models
    elif n_p3d == n_models:
        pseudo3d_dims = pseudo3d_dim
    elif pseudo3d_dim is None:
        pseudo3d_dims = [None] * n_models
    else:
        raise ValueError(
            "pseudo3d_dim must be None (for 3D network), 1 value to be used "
            f"across all models to be {stage}ed, or N values corresponding to each "
            f"of the N models to be {stage}ed. Got {n_p3d} != {n_models}."
        )
    return pseudo3d_dims


def _check_patch_size(patch_size: List[int], use_pseudo3d: bool) -> List[int]:
    n_patch_elems = len(patch_size)
    if n_patch_elems != 2 and use_pseudo3d:
        raise ValueError(
            f"Number of patch size elements must be 2 for "
            f"pseudo-3D (2D) network. Got {len(patch_size)}."
        )
    elif n_patch_elems != 3 and not use_pseudo3d:
        raise ValueError(
            f"Number of patch size elements must be 3 for "
            f"a 3D network. Got {len(patch_size)}."
        )


def _predict_parser_shared(
    parser: ArgParser, necessary_trainer_args: set, add_csv: bool
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
    trainer_args = set(inspect.signature(Trainer).parameters.keys())  # noqa
    unnecessary_args = trainer_args - necessary_trainer_args
    remove_args(parser, unnecessary_args)
    fix_type_funcs(parser)
    return parser


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
        action=config_action,
        help="path to a configuration file in json or yaml format",
    )
    necessary_trainer_args = {
        "benchmark",
        "fast_dev_run",
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
        "fast_dev_run",
        "gpus",
        "precision",
        "progress_bar_refresh_rate",
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


def _predict_whole_image(
    args: Namespace, model_path: Path, model_num: ModelNum,
):
    """ predict a whole image volume as opposed to patches """
    dict_args = vars(args)
    pp = args.predict_probability
    trainer = Trainer.from_argparse_args(args)
    dm = LesionSegDataModulePredictWhole.from_csv(**dict_args)
    dm.setup()
    model = LesionSegLightningTiramisu.load_from_checkpoint(
        model_path, predict_probability=pp, _model_num=model_num,
    )
    logging.debug(model)
    trainer.predict(model, datamodule=dm)
    del dm, model, trainer


def _predict_patch_image(
    args: Namespace,
    model_path: Path,
    model_num: ModelNum,
    pseudo3d_dim: Union[None, int],
):
    """ predict a volume with patches as opposed to a whole volume """
    dict_args = vars(args)
    dict_args["pseudo3d_dim"] = pseudo3d_dim
    pp = args.predict_probability
    subject_list = csv_to_subjectlist(args.predict_csv)
    model = LesionSegLightningTiramisu.load_from_checkpoint(
        model_path, predict_probability=pp, _model_num=model_num,
    )
    logging.debug(model)
    for subject in subject_list:
        trainer = Trainer.from_argparse_args(args)
        dm = LesionSegDataModulePredictPatches(subject, **dict_args)
        dm.setup()
        trainer.predict(model, datamodule=dm)
        del dm, trainer
    del model


def _predict(args: Namespace, parser: ArgParser, use_multiprocessing: bool):
    args = none_string_to_none(args)
    args = path_to_str(args)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    seed_everything(args.seed, workers=True)
    n_models = len(args.model_path)
    if not args.only_aggregate:
        args.predict_probability = n_models > 1
        patch_predict = args.patch_size is not None
        use_pseudo3d = args.pseudo3d_dim is not None
        if patch_predict:
            _check_patch_size(args.patch_size, use_pseudo3d)
        pseudo3d_dims = _pseudo3d_dims_setup(args.pseudo3d_dim, n_models, "predict")
        predict_iter_data = zip(args.model_path, pseudo3d_dims)
        for i, (model_path, p3d) in enumerate(predict_iter_data, 1):
            model_num = ModelNum(num=i, out_of=n_models)
            nth_model = f" ({i}/{n_models})"
            if patch_predict:
                _predict_patch_image(args, model_path, model_num, p3d)
            else:
                _predict_whole_image(args, model_path, model_num)
            logger.info("Finished prediction" + nth_model)
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
    if not args.fast_dev_run and not args.only_aggregate:
        exp_dirs = []
        for mp in args.model_path:
            exp_dirs.append(get_experiment_directory(mp))
        generate_predict_config_yaml(exp_dirs, parser, vars(args))


def predict(args: ArgType = None) -> int:
    """ use a Tiramisu CNN for prediction """
    parser = predict_parser(False)
    if args is None:
        args = parser.parse_args(_skip_check=True)  # noqa
    elif isinstance(args, list):
        args = parser.parse_args(args, _skip_check=True)  # noqa
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
        dict_to_csv(modality_paths, f)  # noqa
        args.predict_csv = f.name
        _predict(args, parser, False)
    return 0
