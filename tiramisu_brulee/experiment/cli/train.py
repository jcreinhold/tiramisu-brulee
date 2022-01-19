#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.cli.train

command-line interface functions for training
lesion segmentation Tiramisu neural networks

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 25, 2021
"""

__all__ = [
    "train",
]

import argparse
import builtins
import copy
import gc
import itertools
import logging
import os
import pathlib
import time
import typing
import warnings

import jsonargparse
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from tiramisu_brulee.experiment.cli.common import (
    EXPERIMENT_NAME,
    check_patch_size,
    handle_fast_dev_run,
    pseudo3d_dims_setup,
    tiramisu_brulee_info,
)
from tiramisu_brulee.experiment.cli.predict import predict_parser
from tiramisu_brulee.experiment.data import LesionSegDataModuleTrain, Mixup
from tiramisu_brulee.experiment.parse import (
    fix_type_funcs,
    generate_predict_config_yaml,
    generate_train_config_yaml,
    get_best_model_path,
    get_experiment_directory,
    none_string_to_none,
    path_to_str,
    remove_args,
)
from tiramisu_brulee.experiment.seg import LesionSegLightningTiramisu
from tiramisu_brulee.experiment.type import ArgParser, ArgType
from tiramisu_brulee.experiment.util import setup_log

# num of dataloader workers is set to 0 for compatibility w/ torchio, so ignore warning
warnings.filterwarnings("ignore", ".*does not have many workers*", category=UserWarning)


def train_parser(use_python_argparse: builtins.bool = True) -> ArgParser:
    """argument parser for training a Tiramisu CNN"""
    if use_python_argparse:
        ArgumentParser = argparse.ArgumentParser
        config_action = None
    else:
        ArgumentParser = jsonargparse.ArgumentParser
        config_action = jsonargparse.ActionConfigFile
    desc = "Train a Tiramisu CNN to segment lesions"
    parser = ArgumentParser(
        prog="lesion-train",
        description=desc,
    )
    parser.add_argument(
        "--config",
        action=config_action,  # type: ignore[arg-type]
        help="path to a configuration file in json or yaml format",
    )
    exp_parser = parser.add_argument_group("Experiment")
    exp_parser.add_argument(
        "-sd",
        "--seed",
        type=int,
        default=0,
        help="set seed for reproducibility",
    )
    exp_parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="increase output verbosity (e.g., -vv is more than -v)",
    )
    exp_parser.add_argument(
        "-uri",
        "--tracking-uri",
        type=str,
        default=None,
        help="use this URI for tracking metrics and artifacts with an MLFlow server",
    )
    exp_parser.add_argument(
        "-md",
        "--model-dir",
        type=str,
        default=None,
        help="save models to this directory if provided, "
        "otherwise save in default_root_dir",
    )
    exp_parser.add_argument(
        "-en",
        "--experiment-name",
        type=str,
        default=EXPERIMENT_NAME,
        help="name of the experiment/output directory",
    )
    exp_parser.add_argument(
        "-tn",
        "--trial-name",
        type=str,
        default=None,
        help="name of the trial/output version directory",
    )
    exp_parser.add_argument(
        "-stk",
        "--save-top-k",
        type=int,
        default=3,
        help="save the top k models in checkpoints",
    )
    parser = LesionSegLightningTiramisu.add_io_arguments(parser)
    parser = LesionSegLightningTiramisu.add_model_arguments(parser)
    parser = LesionSegLightningTiramisu.add_other_arguments(parser)
    parser = LesionSegLightningTiramisu.add_training_arguments(parser)
    parser = LesionSegDataModuleTrain.add_arguments(parser)
    parser = Mixup.add_arguments(parser)
    parser = Trainer.add_argparse_args(parser)
    unnecessary_args = {
        "enable_checkpointing",
        "in_channels",
        "logger",
        "max_steps",
        "min_steps",
        "out_channels",
        "weights_save_path",
    }
    if use_python_argparse:
        unnecessary_args.union({"min_epochs", "max_epochs"})
    else:
        parser.link_arguments("n_epochs", "min_epochs")  # type: ignore[attr-defined]
        parser.link_arguments("n_epochs", "max_epochs")  # type: ignore[attr-defined]
    unnecessary_args = handle_fast_dev_run(unnecessary_args)
    remove_args(parser, unnecessary_args)
    fix_type_funcs(parser)
    return parser


def train(
    args: ArgType = None,
    *,
    return_best_model_paths: builtins.bool = False,
) -> typing.Union[typing.List[pathlib.Path], builtins.int]:
    """train a Tiramisu CNN for segmentation"""
    parser = train_parser(False)
    if args is None:
        args = parser.parse_args(_skip_check=True)  # type: ignore[call-overload]
    elif isinstance(args, list):
        args = parser.parse_args(args, _skip_check=True)  # type: ignore[call-overload]
    args = none_string_to_none(args)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    seed_everything(args.seed, workers=True)
    args = path_to_str(args)
    n_models_to_train = _compute_num_models_to_train(args)
    best_model_paths: typing.List[pathlib.Path] = []
    use_pseudo3d = args.pseudo3d_dim is not None
    check_patch_size(args.patch_size, use_pseudo3d)
    pseudo3d_dims = pseudo3d_dims_setup(args.pseudo3d_dim, n_models_to_train, "train")
    individual_run_args = copy.deepcopy(vars(args))
    individual_run_args["network_dim"] = 2 if use_pseudo3d else 3
    channels_per_image = args.pseudo3d_size if use_pseudo3d else 1
    individual_run_args["in_channels"] = args.num_input * channels_per_image
    train_iter_data = zip(args.train_csv, args.valid_csv, pseudo3d_dims)
    trainer: typing.Optional[Trainer] = None
    for i, (train_csv, valid_csv, p3d) in enumerate(train_iter_data, 1):
        trainer, checkpoint_callback = _setup_trainer_and_checkpoint(args)
        nth_model = f" ({i}/{n_models_to_train})"
        individual_run_args["train_csv"] = train_csv
        individual_run_args["valid_csv"] = valid_csv
        individual_run_args["pseudo3d_dim"] = p3d
        dm = LesionSegDataModuleTrain.from_csv(**individual_run_args)
        model = LesionSegLightningTiramisu(**individual_run_args)
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
        # kill multiprocessing workers, free memory for the next iteration
        dm.teardown()
        trainer.teardown()
        del dm, model, checkpoint_callback
        if i != n_models_to_train:
            del trainer
            trainer = None
        gc.collect()
        torch.cuda.empty_cache()
        if n_models_to_train > 1 and args.num_workers > 0:
            time.sleep(5.0)
    if trainer is not None:
        assert isinstance(trainer.logger, (TensorBoardLogger, MLFlowLogger))
        _generate_config_yamls_in_train(args, best_model_paths, trainer.logger)
    return best_model_paths if return_best_model_paths else 0


def _compute_num_models_to_train(args: ArgType) -> builtins.int:
    assert isinstance(args, (argparse.Namespace, jsonargparse.Namespace))
    n_models_to_train = len(args.train_csv)
    if n_models_to_train != len(args.valid_csv):
        raise ValueError(
            "Number of training and validation CSVs must be equal.\n"
            f"Got {n_models_to_train} != {len(args.valid_csv)}"
        )
    return n_models_to_train


def _format_checkpoints(args: ArgType) -> typing.Dict[builtins.str, typing.Any]:
    assert isinstance(args, (argparse.Namespace, jsonargparse.Namespace))
    checkpoint_format = "{epoch}-{val_loss:.3f}"
    if args.track_metric != "loss":
        checkpoint_format += f"-{{val_{args.track_metric}:.3f}}"
    checkpoint_kwargs = dict(
        dirpath=args.model_dir,
        filename=checkpoint_format,
        monitor=f"val_{args.track_metric}",
        save_top_k=args.save_top_k,
        save_last=True,
        mode="max" if args.track_metric != "loss" else "min",
        every_n_epochs=args.checkpoint_every_n_epochs,
    )
    return checkpoint_kwargs


def _artifact_directory(args: ArgType) -> builtins.str:
    assert isinstance(args, (argparse.Namespace, jsonargparse.Namespace))
    if args.default_root_dir is not None:
        artifact_dir = pathlib.Path(args.default_root_dir).resolve()
    else:
        artifact_dir = pathlib.Path.cwd()
    return str(artifact_dir)


def _setup_experiment_logger(
    args: ArgType,
) -> typing.Union[TensorBoardLogger, MLFlowLogger]:
    assert isinstance(args, (argparse.Namespace, jsonargparse.Namespace))
    exp_logger: typing.Union[TensorBoardLogger, MLFlowLogger]
    if args.tracking_uri is not None:
        exp_logger = MLFlowLogger(
            experiment_name=args.experiment_name,
            run_name=args.trial_name,
            tracking_uri=args.tracking_uri,
        )
    else:
        artifact_dir = _artifact_directory(args)
        ignore_tensorboard_dir = bool(os.getenv("TIRAMISU_IGNORE_TB_DIR", False))
        tensorboard_dir = pathlib.Path(
            "/opt/ml/output/tensorboard"
        ).resolve()  # for SageMaker
        if not tensorboard_dir.is_dir() or ignore_tensorboard_dir:
            exp_logger = TensorBoardLogger(
                artifact_dir,
                name=args.experiment_name,
                version=args.trial_name,
                sub_dir="tensorboard",
            )
        else:
            logging.info(f"Saving tensorboard logs to {tensorboard_dir}.")
            logging.info("Set an env variable TIRAMISU_IGNORE_TB_DIR=true to prevent.")
            exp_logger = TensorBoardLogger(str(tensorboard_dir), name="", version="")
    return exp_logger


def _generate_config_yamls_in_train(
    args: ArgType,
    best_model_paths: typing.List[pathlib.Path],
    logger: typing.Union[TensorBoardLogger, MLFlowLogger],
) -> None:
    assert isinstance(args, (argparse.Namespace, jsonargparse.Namespace))
    has_fdr = hasattr(args, "fast_dev_run")
    generate_config_yaml = (not args.fast_dev_run) if has_fdr else True
    if generate_config_yaml:
        n_models_to_train = _compute_num_models_to_train(args)
        dict_args = vars(args)
        exp_dirs = [get_experiment_directory(bmp) for bmp in best_model_paths]
        if args.pseudo3d_dim == "all":
            dict_args["pseudo3d_dim"] = [0, 1, 2] * n_models_to_train
            best_model_paths = [bmp for bmp in best_model_paths for _ in range(3)]
        else:
            dict_args["pseudo3d_dim"] = args.pseudo3d_dim
        train_cfgs = generate_train_config_yaml(
            exp_dirs=exp_dirs,
            dict_args=dict_args,
            best_model_paths=best_model_paths,
            parser=train_parser(False),
        )
        predict_cfgs = generate_predict_config_yaml(
            exp_dirs=exp_dirs,
            dict_args=dict_args,
            best_model_paths=best_model_paths,
            parser=predict_parser(False),
        )
        if args.tracking_uri is not None:
            assert isinstance(logger, MLFlowLogger)
            run_id = logger.run_id
            for cfg in itertools.chain(train_cfgs, predict_cfgs):
                logger.experiment.log_artifact(run_id, cfg)


class MLFlowModelCheckpoint(ModelCheckpoint):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        mlflow_logger: MLFlowLogger,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mlflow_logger = mlflow_logger

    # flake8: noqa: E501
    def save_checkpoint(
        self,
        trainer: Trainer,
        unused: typing.Optional[LightningModule] = None,
    ) -> None:
        try:
            super().save_checkpoint(trainer=trainer, unused=unused)  # type: ignore[call-arg]
        except TypeError:
            super().save_checkpoint(trainer=trainer)
        run_id = self.mlflow_logger.run_id
        self.mlflow_logger.experiment.log_artifact(run_id, self.best_model_path)


def _create_checkpoint_callback(
    args: ArgType,
    logger: typing.Union[TensorBoardLogger, MLFlowLogger],
) -> typing.Union[ModelCheckpoint, MLFlowModelCheckpoint]:
    checkpoint_kwargs = _format_checkpoints(args)
    assert isinstance(args, (argparse.Namespace, jsonargparse.Namespace))
    if args.tracking_uri is None:
        return ModelCheckpoint(**checkpoint_kwargs)
    else:
        assert isinstance(logger, MLFlowLogger)
        return MLFlowModelCheckpoint(mlflow_logger=logger, **checkpoint_kwargs)


def _setup_trainer_and_checkpoint(
    args: ArgType,
) -> typing.Tuple[Trainer, ModelCheckpoint]:
    assert isinstance(args, (argparse.Namespace, jsonargparse.Namespace))
    use_multigpus = not (args.gpus is None or args.gpus <= 1)
    exp_logger = _setup_experiment_logger(args)
    checkpoint_callback = _create_checkpoint_callback(args, exp_logger)
    plugins = args.plugins
    if use_multigpus and args.accelerator == "ddp":
        plugins = DDPPlugin(find_unused_parameters=False)
    trainer = Trainer.from_argparse_args(
        args,
        logger=exp_logger,
        callbacks=[checkpoint_callback],
        plugins=plugins,
    )
    if args.tracking_uri is not None:
        tb_info = tiramisu_brulee_info()
        run_id = trainer.logger.run_id
        trainer.logger.experiment.set_tag(run_id, "Version", tb_info.version)
        trainer.logger.experiment.set_tag(run_id, "Commit", tb_info.commit)
    return trainer, checkpoint_callback
