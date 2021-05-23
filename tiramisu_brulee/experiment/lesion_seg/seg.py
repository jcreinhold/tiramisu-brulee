#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_seg.seg

3D Tiramisu network for lesion segmentation

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 14, 2021
"""

__all__ = [
    "LesionSegLightningTiramisu",
]

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import deque, namedtuple
from functools import partial
import inspect
import logging
from pathlib import Path
import tempfile
from typing import List, Optional, Tuple, Union

from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    Namespace,
    set_config_read_mode,
)
import nibabel as nib
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch import Tensor
from torch.optim import AdamW, RMSprop, lr_scheduler

from tiramisu_brulee.experiment.lightningtiramisu import LightningTiramisu
from tiramisu_brulee.experiment.lesion_seg.data import (
    Mixup,
    LesionSegDataModulePredict,
    LesionSegDataModuleTrain,
)
from tiramisu_brulee.experiment.lesion_seg.lesion_tools import (
    almost_isbi15_score,
    clean_segmentation,
)
from tiramisu_brulee.experiment.lesion_seg.parse import (
    dict_to_csv,
    file_path,
    fix_type_funcs,
    generate_predict_config_yaml,
    generate_train_config_yaml,
    get_best_model_path,
    get_experiment_directory,
    none_string_to_none,
    nonnegative_int,
    parse_unknown_to_dict,
    path_to_str,
    positive_float,
    positive_int,
    probability_float,
    remove_args,
)
from tiramisu_brulee.experiment.lesion_seg.util import (
    append_num_to_filename,
    BoundingBox3D,
    minmax_scale_batch,
    setup_log,
    to_np,
)
from tiramisu_brulee.loss import (
    binary_combo_loss,
    l1_segmentation_loss,
    mse_segmentation_loss,
)

ArgType = Optional[Union[Namespace, List[str]]]
ModelNum = namedtuple("ModelNum", ["num", "out_of"])
EXPERIMENT_NAME = "lesion_tiramisu_experiment"
set_config_read_mode(fsspec_enabled=True)


class LesionSegLightningTiramisu(LightningTiramisu):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        down_blocks: List[int] = (4, 4, 4, 4, 4),
        up_blocks: List[int] = (4, 4, 4, 4, 4),
        bottleneck_layers: int = 4,
        growth_rate: int = 16,
        first_conv_out_channels: int = 48,
        dropout_rate: float = 0.2,
        init_type: str = "normal",
        gain: float = 0.02,
        n_epochs: int = 1,
        learning_rate: float = 1e-3,
        betas: Tuple[int, int] = (0.9, 0.99),
        weight_decay: float = 1e-7,
        combo_weight: float = 0.6,
        decay_after: int = 8,
        loss_function: str = "combo",
        rmsprop: bool = False,
        soft_labels: bool = False,
        threshold: float = 0.5,
        min_lesion_size: int = 3,
        fill_holes: bool = True,
        predict_probability: bool = False,
        mixup: bool = False,
        mixup_alpha: float = 0.4,
        _model_num: ModelNum = ModelNum(1, 1),  # internal param for ith of n models
        **kwargs,
    ):
        network_dim = 3  # only support 3D input
        super().__init__(
            network_dim,
            in_channels,
            out_channels,
            down_blocks,
            up_blocks,
            bottleneck_layers,
            growth_rate,
            first_conv_out_channels,
            dropout_rate,
            init_type,
            gain,
            n_epochs,
            learning_rate,
            betas,
            weight_decay,
        )
        self._model_num = _model_num
        self.save_hyperparameters(ignore="_model_num")

    def setup(self, stage: Optional[str] = None):
        if self.hparams.loss_function == "combo":
            self.criterion = partial(
                binary_combo_loss, combo_weight=self.hparams.combo_weight
            )
        elif self.hparams.loss_function == "mse":
            self.criterion = mse_segmentation_loss
        elif self.hparams.loss_function == "l1":
            self.criterion = l1_segmentation_loss
        else:
            raise ValueError(f"{self.hparams.loss_function} not supported.")
        if self.hparams.mixup:
            self._mix = Mixup(self.hparams.mixup_alpha)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        src, tgt = batch
        if self.hparams.mixup:
            src, tgt = self._mix(src, tgt)
        pred = self(src)
        loss = self.criterion(pred, tgt)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        src, tgt = batch
        pred = self(src)
        loss = self.criterion(pred, tgt)
        with torch.no_grad():
            pred_seg = torch.sigmoid(pred) > self.hparams.threshold
        isbi15_score = almost_isbi15_score(pred_seg, tgt)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
        )
        self.log(
            "val_isbi15_score",
            isbi15_score,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        if batch_idx == 0:
            images = dict(truth=tgt, pred=pred,)
            for i in range(src.shape[1]):
                images[f"input_channel_{i}"] = src[:, i : i + 1, ...]
        else:
            images = None
        return dict(val_loss=loss, images=images)

    def validation_epoch_end(self, outputs: list):
        self._log_images(outputs[0]["images"])

    def predict_step(
        self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tensor:
        src = batch["src"]
        bbox = BoundingBox3D.from_batch(src, pad=0)
        src = bbox(src)
        pred = self(src)
        pred_seg = torch.sigmoid(pred)
        if not self.hparams.predict_probability:
            pred_seg = pred_seg > self.hparams.threshold
        pred_seg = bbox.uncrop_batch(pred_seg)
        pred_seg = pred_seg.float()
        return pred_seg

    def on_predict_batch_end(
        self,
        pred_step_outputs: Tensor,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int,
    ):
        data = to_np(pred_step_outputs).squeeze()
        if data.ndim == 5:
            raise ValueError("Predictions should only have one channel.")
        elif data.ndim == 3:
            data = data[None, ...]
        if not self.hparams.predict_probability:
            data = [clean_segmentation(seg) for seg in data]
        data = [seg.astype(np.float32) for seg in data]
        affine_matrices = batch["affine"]
        out_fns = batch["out"]
        nifti_attrs = zip(data, affine_matrices, out_fns)
        for data, affine, fn in nifti_attrs:
            if self._model_num != ModelNum(num=1, out_of=1):
                fn = append_num_to_filename(fn, self._model_num.num)
            logging.info(f"Saving {fn}.")
            nib.Nifti1Image(data, affine).to_filename(fn)

    def configure_optimizers(self):
        if self.hparams.rmsprop:
            momentum, alpha = self.hparams.betas
            optimizer = RMSprop(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=momentum,
                alpha=alpha,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                weight_decay=self.hparams.weight_decay,
            )

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.decay_rule)
        return [optimizer], [scheduler]

    def decay_rule(self, epoch: int) -> float:
        numerator = max(0, epoch - self.hparams.decay_after)
        denominator = float(self.hparams.n_epochs + 1)
        lr = 1.0 - numerator / denominator
        return lr

    def _log_images(self, images: dict):
        n = self.current_epoch
        mid_slice = None
        for key, image in images.items():
            with torch.no_grad():
                if mid_slice is None:
                    mid_slice = image.shape[-1] // 2
                image_slice = image[..., mid_slice]
                if self.hparams.soft_labels and key == "pred":
                    image_slice = torch.sigmoid(image_slice)
                elif key == "pred":
                    threshold = self.hparams.threshold
                    image_slice = torch.sigmoid(image_slice) > threshold
                elif key == "truth":
                    image_slice = image_slice > 0.0
                else:
                    image_slice = minmax_scale_batch(image_slice)
            self.logger.experiment.add_images(key, image_slice, n, dataformats="NCHW")

    @staticmethod
    def add_model_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument(
            "-ic",
            "--in-channels",
            type=positive_int(),
            default=1,
            help="number of input channels (should match the number "
            "of non-label/other fields in the input csv)",
        )
        parser.add_argument(
            "-oc",
            "--out-channels",
            type=positive_int(),
            default=1,
            help="number of output channels (usually 1 for segmentation)",
        )
        parser.add_argument(
            "-dr",
            "--dropout-rate",
            type=positive_float(),
            default=0.2,
            help="dropout rate/probability",
        )
        parser.add_argument(
            "-it",
            "--init-type",
            type=str,
            default="he_uniform",
            choices=(
                "normal",
                "xavier_normal",
                "he_normal",
                "he_uniform",
                "orthogonal",
            ),
            help="use this type of initialization for the network",
        )
        parser.add_argument(
            "-ig",
            "--init-gain",
            type=positive_float(),
            default=0.2,
            help="use this initialization gain for initialization",
        )
        parser.add_argument(
            "-db",
            "--down-blocks",
            type=positive_int(),
            default=[4, 4, 4, 4, 4],
            nargs="+",
            help="tiramisu down-sample path specification",
        )
        parser.add_argument(
            "-ub",
            "--up-blocks",
            type=positive_int(),
            default=[4, 4, 4, 4, 4],
            nargs="+",
            help="tiramisu up-sample path specification",
        )
        parser.add_argument(
            "-bl",
            "--bottleneck-layers",
            type=positive_int(),
            default=4,
            help="tiramisu bottleneck specification",
        )
        parser.add_argument(
            "-gr",
            "--growth-rate",
            type=positive_int(),
            default=12,
            help="tiramisu growth rate (number of channels "
            "added between each layer in a dense block)",
        )
        parser.add_argument(
            "-fcoc",
            "--first-conv-out-channels",
            type=positive_int(),
            default=48,
            help="number of output channels in first conv",
        )
        return parent_parser

    @staticmethod
    def add_training_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Training")
        parser.add_argument(
            "-bt",
            "--betas",
            type=positive_float(),
            default=[0.9, 0.99],
            nargs=2,
            help="AdamW momentum parameters (for RMSprop, momentum and alpha)",
        )
        parser.add_argument(
            "-cen",
            "--checkpoint-every-n-epochs",
            type=positive_int(),
            default=1,
            help="save model weights (checkpoint) every n epochs",
        )
        parser.add_argument(
            "-cw",
            "--combo-weight",
            type=positive_float(),
            default=0.6,
            help="weight of positive class in combo loss",
        )
        parser.add_argument(
            "-da",
            "--decay-after",
            type=positive_int(),
            default=8,
            help="decay learning rate after this number of epochs",
        )
        parser.add_argument(
            "-lr",
            "--learning-rate",
            type=positive_float(),
            default=3e-4,
            help="learning rate for the optimizer",
        )
        parser.add_argument(
            "-lf",
            "--loss-function",
            type=str,
            default="combo",
            choices=("combo", "l1", "mse"),
            help="loss function to train the network",
        )
        parser.add_argument(
            "-ne",
            "--n-epochs",
            type=positive_int(),
            default=64,
            help="number of epochs",
        )
        parser.add_argument(
            "-rp",
            "--rmsprop",
            action="store_true",
            default=False,
            help="use rmsprop instead of adam",
        )
        parser.add_argument(
            "-wd",
            "--weight-decay",
            type=positive_float(),
            default=1e-5,
            help="weight decay parameter for adamw",
        )
        parser.add_argument(
            "-sl",
            "--soft-labels",
            action="store_true",
            default=False,
            help="use soft labels (i.e., non-binary labels) for training",
        )
        return parent_parser

    @staticmethod
    def add_other_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Other")
        parser.add_argument(
            "-th",
            "--threshold",
            type=probability_float(),
            default=0.5,
            help="probability threshold for segmentation",
        )
        return parent_parser

    @staticmethod
    def add_testing_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Testing")
        parser.add_argument(
            "-mls",
            "--min-lesion-size",
            type=nonnegative_int(),
            default=3,
            help="in testing, remove lesions smaller in voxels than this",
        )
        parser.add_argument(
            "-fh",
            "--fill-holes",
            action="store_true",
            default=False,
            help="in testing, preform binary hole filling",
        )
        parser.add_argument(
            "-pp",
            "--predict-probability",
            action="store_true",
            default=False,
            help="in testing, store the probability instead of the binary prediction",
        )
        return parent_parser


def train_parser() -> ArgumentParser:
    """ argument parser for training a 3D Tiramisu CNN """
    desc = "Train a 3D Tiramisu CNN to segment lesions"
    parser = ArgumentParser(prog="lesion-train", description=desc,)
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
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
    parser.link_arguments("n_epochs", "min_epochs")  # noqa
    parser.link_arguments("n_epochs", "max_epochs")  # noqa
    unnecessary_args = [
        "checkpoint_callback",
        "logger",
        "min_steps",
        "max_steps",
        "truncated_bptt_steps",
        "weights_save_path",
    ]
    remove_args(parser, unnecessary_args)
    fix_type_funcs(parser)
    return parser


def train(args: ArgType = None, return_best_model_paths: bool = False) -> int:
    """ train a 3D Tiramisu CNN for segmentation """
    parser = train_parser()
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
    for i, (train_csv, valid_csv) in enumerate(csvs, 1):
        tb_logger = TensorBoardLogger(str(root_dir), name=name)
        checkpoint_callback = ModelCheckpoint(**checkpoint_kwargs)
        trainer = Trainer.from_argparse_args(
            args,
            logger=tb_logger,
            checkpoint_callback=True,
            callbacks=[checkpoint_callback],
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
        generate_predict_config_yaml(**config_kwargs, parser=predict_parser())
    if return_best_model_paths:
        return best_model_paths
    else:
        return 0


def _predict_parser_shared(
    parser: ArgumentParser, necessary_trainer_args: set, add_csv: bool
) -> ArgumentParser:
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


def predict_parser() -> ArgumentParser:
    """ argument parser for using a 3D Tiramisu CNN for prediction """
    desc = "Use a Tiramisu CNN to segment lesions"
    parser = ArgumentParser(prog="lesion-predict", description=desc)
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="path to a configuration file in json or yaml format",
    )
    necessary_trainer_args = {
        "benchmark",
        "gpus",
        "fast_dev_run",
        "default_root_dir",
    }
    parser = _predict_parser_shared(parser, necessary_trainer_args, True)
    return parser


def predict_image_parser() -> ArgumentParser:
    """ argument parser for using a 3D Tiramisu CNN for single-timepoint prediction """
    desc = "Use a Tiramisu CNN to segment lesions for a single-timepoint prediction"
    parser = argparse.ArgumentParser(prog="lesion-predict-image", description=desc)
    necessary_trainer_args = {
        "gpus",
        "fast_dev_run",
        "default_root_dir",
    }
    parser = _predict_parser_shared(parser, necessary_trainer_args, False)
    return parser


def _aggregate(n_fn: Tuple[int, str], threshold: float, n_models: int, n_fns: int):
    """ aggregate helper for concurrent/parallel processing """
    n, fn = n_fn
    data = []
    for i in range(1, n_models + 1):
        _fn = append_num_to_filename(fn, i)
        nib_image = nib.load(_fn)
        data.append(nib_image.get_fdata())
    aggregated = (np.mean(data, axis=0) > threshold).astype(np.float32)
    nib.Nifti1Image(aggregated, nib_image.affine).to_filename(fn)  # noqa
    logging.info(f"Save aggregated prediction: {fn} ({n}/{n_fns})")


def aggregate(
    predict_csv: str,
    n_models: int,
    threshold: float = 0.5,
    num_workers: Optional[int] = None,
):
    """ aggregate output from multiple model predictions """
    csv = pd.read_csv(predict_csv)
    out_fns = csv["out"]
    n_fns = len(out_fns)
    out_fn_iter = enumerate(out_fns, 1)
    _aggregator = partial(
        _aggregate, threshold=threshold, n_models=n_models, n_fns=n_fns
    )
    use_multiprocessing = True if num_workers is None else num_workers > 0
    if use_multiprocessing:
        with ProcessPoolExecutor(num_workers) as executor:
            executor.map(_aggregator, out_fn_iter)
    else:
        deque(map(_aggregator, out_fn_iter), maxlen=0)


def _predict(args: Namespace, parser: ArgumentParser, use_multiprocessing: bool):
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
        aggregate(args.predict_csv, n_models, args.threshold, num_workers)
    if not args.fast_dev_run:
        exp_dirs = []
        for mp in args.model_path:
            exp_dirs.append(get_experiment_directory(mp))
        generate_predict_config_yaml(exp_dirs, parser, dict_args)


def predict(args: ArgType = None) -> int:
    """ use a 3D Tiramisu CNN for prediction """
    parser = predict_parser()
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
