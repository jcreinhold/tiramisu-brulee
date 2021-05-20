#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_seg.seg

3D Tiramisu network for lesion segmentation

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 14, 2021
"""

__all__ = [
    'LesionSegLightningTiramisu',
]

from functools import partial
import inspect
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    set_config_read_mode,
)
import nibabel as nib
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
    file_path,
    fix_type_funcs,
    generate_predict_config_yaml,
    generate_train_config_yaml,
    get_best_model_path,
    get_experiment_directory,
    none_string_to_none,
    nonnegative_int,
    positive_float,
    positive_int,
    probability_float,
    remove_args,
)
from tiramisu_brulee.experiment.lesion_seg.util import (
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

EXPERIMENT_NAME = 'lesion_tiramisu_experiment'
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
        init_type: str = 'normal',
        gain: float = 0.02,
        n_epochs: int = 1,
        learning_rate: float = 1e-3,
        betas: Tuple[int, int] = (0.9, 0.99),
        weight_decay: float = 1e-7,
        combo_weight: float = 0.6,
        decay_after: int = 8,
        loss_function: str = 'combo',
        rmsprop: bool = False,
        soft_labels: bool = False,
        threshold: float = 0.5,
        min_lesion_size: int = 3,
        fill_holes: bool = True,
        mixup: bool = False,
        mixup_alpha: float = 0.4,
        **kwargs
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
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if self.hparams.loss_function == 'combo':
            self.criterion = partial(
                binary_combo_loss,
                combo_weight=self.hparams.combo_weight
            )
        elif self.hparams.loss_function == 'mse':
            self.criterion = mse_segmentation_loss
        elif self.hparams.loss_function == 'l1':
            self.criterion = l1_segmentation_loss
        else:
            raise ValueError(f'{self.hparams.loss_function} not supported.')
        if self.hparams.mixup:
            self._mix = Mixup(self.hparams.mixup_alpha)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        src, tgt = batch
        if self.hparams.mixup:
            src, tgt = self._mix(src, tgt)
        pred = self(src)
        loss = self.criterion(pred, tgt)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        src, tgt = batch
        pred = self(src)
        loss = self.criterion(pred, tgt)
        with torch.no_grad():
            pred_seg = torch.sigmoid(pred) > self.hparams.threshold
        isbi15_score = almost_isbi15_score(pred_seg, tgt)
        self.log(
            'val_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        self.log(
            'val_isbi15_score',
            isbi15_score,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        if batch_idx == 0:
            images = dict(
                truth=tgt,
                pred=pred,
            )
            for i in range(src.shape[1]):
                images[f'input_channel_{i}'] = src[:, i:i + 1, ...]
        else:
            images = None
        return dict(val_loss=loss, images=images)

    def validation_epoch_end(self, outputs: list):
        self._log_images(outputs[0]['images'])

    def predict_step(
        self,
        batch: dict,
        batch_idx: int,
        dataloader_idx: Optional[int] = None
    ) -> Tensor:
        src = batch['src']
        bbox = BoundingBox3D.from_batch(src, pad=0)
        src = bbox(src)
        pred = self(src)
        pred_seg = torch.sigmoid(pred) > self.hparams.threshold
        pred_seg = bbox.uncrop_batch(pred_seg)
        pred_seg = pred_seg.float()
        return pred_seg

    def on_predict_batch_end(
        self,
        pred_step_outputs: Tensor,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int
    ):
        pred_step_outputs = to_np(pred_step_outputs).squeeze()
        for pred_seg in pred_step_outputs:
            clean_segmentation(pred_seg, )
        affine_matrices = batch['affine']
        out_fns = batch['out']
        nifti_attrs = zip(pred_step_outputs, affine_matrices, out_fns)
        for pred, affine, fn in nifti_attrs:
            logging.info(f'Saving {fn}.')
            nib.Nifti1Image(pred, affine).to_filename(fn)

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
                if self.hparams.soft_labels and key == 'pred':
                    image_slice = torch.sigmoid(image_slice)
                elif key == 'pred':
                    threshold = self.hparams.threshold
                    image_slice = torch.sigmoid(image_slice) > threshold
                elif key == 'truth':
                    image_slice = image_slice > 0.
                else:
                    image_slice = minmax_scale_batch(image_slice)
            self.logger.experiment.add_images(
                key,
                image_slice,
                n,
                dataformats='NCHW'
            )

    # flake8: noqa: E501
    @staticmethod
    def add_model_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument('-ic', '--in-channels', type=positive_int(), default=1,
                            help='number of input channels (should match the number '
                                 'of non-label/other fields in the input csv)')
        parser.add_argument('-oc', '--out-channels', type=positive_int(), default=1,
                            help='number of output channels (usually 1 for segmentation)')
        parser.add_argument('-dr', '--dropout-rate', type=positive_float(), default=0.2,
                            help='dropout rate/probability')
        parser.add_argument('-it', '--init-type', type=str, default='he_uniform',
                            choices=('normal', 'xavier_normal', 'he_normal', 'he_uniform', 'orthogonal'),
                            help='use this type of initialization for the network')
        parser.add_argument('-ig', '--init-gain', type=positive_float(), default=0.2,
                            help='use this initialization gain for initialization')
        parser.add_argument('-db', '--down-blocks', type=positive_int(), default=[4, 4, 4, 4, 4], nargs='+',
                            help='tiramisu down-sample path specification')
        parser.add_argument('-ub', '--up-blocks', type=positive_int(), default=[4, 4, 4, 4, 4], nargs='+',
                            help='tiramisu up-sample path specification')
        parser.add_argument('-bl', '--bottleneck-layers', type=positive_int(), default=4,
                            help='tiramisu bottleneck specification')
        parser.add_argument('-gr', '--growth-rate', type=positive_int(), default=12,
                            help='tiramisu growth rate (number of channels '
                                 'added between each layer in a dense block)')
        parser.add_argument('-fcoc', '--first-conv-out-channels', type=positive_int(), default=48,
                            help='number of output channels in first conv')
        return parent_parser

    # flake8: noqa: E501
    @staticmethod
    def add_training_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Training")
        parser.add_argument('-bt', '--betas', type=positive_float(), default=[0.9, 0.99], nargs=2,
                            help='AdamW momentum parameters (or, for RMSprop, momentum and alpha params)')
        parser.add_argument('-cen', '--checkpoint-every-n-epochs', type=positive_int(), default=1,
                            help='save model weights (checkpoint) every n epochs')
        parser.add_argument('-cw', '--combo-weight', type=positive_float(), default=0.6,
                            help='weight of positive class in combo loss')
        parser.add_argument('-da', '--decay-after', type=positive_int(), default=8,
                            help='decay learning rate after this number of epochs')
        parser.add_argument('-lr', '--learning-rate', type=positive_float(), default=3e-4,
                            help='learning rate for the optimizer')
        parser.add_argument('-lf', '--loss-function', type=str, default='combo',
                            choices=('combo', 'l1', 'mse'),
                            help='loss function to train the network')
        parser.add_argument('-ne', '--n-epochs', type=positive_int(), default=64,
                            help='number of epochs')
        parser.add_argument('-rp', '--rmsprop', action='store_true', default=False,
                            help="use rmsprop instead of adam")
        parser.add_argument('-wd', '--weight-decay', type=positive_float(), default=1e-5,
                            help="weight decay parameter for adamw")
        parser.add_argument('-sl', '--soft-labels', action='store_true', default=False,
                            help="use soft labels (i.e., non-binary labels) for training")
        return parent_parser

    # flake8: noqa: E501
    @staticmethod
    def add_other_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Other")
        parser.add_argument('-th', '--threshold', type=probability_float(), default=0.5,
                            help='probability threshold for segmentation')
        return parent_parser

    # flake8: noqa: E501
    @staticmethod
    def add_testing_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Testing")
        parser.add_argument('-mls', '--min-lesion-size', type=nonnegative_int(), default=3,
                            help='in testing, remove lesions smaller in voxels than this')
        parser.add_argument('-fh', '--fill-holes', action='store_true', default=False,
                            help='in testing, preform binary hole filling')
        return parent_parser


# flake8: noqa: E501
def train_parser():
    desc = 'Train a 3D Tiramisu CNN to segment lesions'
    parser = ArgumentParser(
        prog='lesion-train',
        description=desc,
    )
    parser.add_argument('--config', action=ActionConfigFile,
                        help='path to a configuration file in json or yaml format')
    exp_parser = parser.add_argument_group('Experiment')
    exp_parser.add_argument('-sd', '--seed', type=int, default=0,
                            help='set seed for reproducibility')
    exp_parser.add_argument('-v', '--verbosity', action="count", default=0,
                            help="increase output verbosity (e.g., -vv is more than -v)")
    parser = LesionSegLightningTiramisu.add_model_arguments(parser)
    parser = LesionSegLightningTiramisu.add_training_arguments(parser)
    parser = LesionSegLightningTiramisu.add_other_arguments(parser)
    parser = LesionSegDataModuleTrain.add_arguments(parser)
    parser = Mixup.add_arguments(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.link_arguments('n_epochs', 'min_epochs')  # noqa
    parser.link_arguments('n_epochs', 'max_epochs')  # noqa
    unnecessary_options = [
        'checkpoint_callback',
        'logger',
        'min_steps',
        'max_steps',
        'truncated_bptt_steps',
        'weights_save_path',
    ]
    remove_args(parser, unnecessary_options)
    fix_type_funcs(parser)
    return parser


def train(args=None, return_best_model_path=False):
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
    tb_logger = TensorBoardLogger(
        str(root_dir),
        name=name,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_isbi15_score',
        save_top_k=3,
        save_last=True,
        mode='max',
        every_n_val_epochs=args.checkpoint_every_n_epochs,
    )
    trainer = Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
    )
    dict_args = vars(args)
    dm = LesionSegDataModuleTrain.from_csv(
        **dict_args,
    )
    dm.setup()
    model = LesionSegLightningTiramisu(**dict_args)
    logger.debug(model)
    if args.auto_scale_batch_size or args.auto_lr_find:
        tuning_output = trainer.tune(model, datamodule=dm)
        logger.info(tuning_output)
    trainer.fit(model, datamodule=dm)
    best_model_path = get_best_model_path(checkpoint_callback)
    exp_dir = get_experiment_directory(best_model_path)
    logger.info(f"Finished training.")
    if not args.fast_dev_run:
        logger.info(f"Best model path: {best_model_path}")
        config_kwargs = dict(
            exp_dir=exp_dir,
            dict_args=dict_args,
            best_model_path=best_model_path,
        )
        generate_train_config_yaml(**config_kwargs, parser=parser)
        generate_predict_config_yaml(**config_kwargs, parser=predict_parser())
    if return_best_model_path:
        return best_model_path
    else:
        return 0


# flake8: noqa: E501
def predict_parser():
    desc = 'Use a Tiramisu CNN to segment lesions'
    parser = ArgumentParser(
        prog='lesion-predict',
        description=desc,
    )
    parser.add_argument('--config', action=ActionConfigFile,
                        help='path to a configuration file in json or yaml format')
    exp_parser = parser.add_argument_group('Experiment')
    exp_parser.add_argument('-mp', '--model-path', type=file_path(), required=True,
                            help='path to output the trained model')
    exp_parser.add_argument('-sd', '--seed', type=int, default=0,
                            help='set seed for reproducibility')
    exp_parser.add_argument('-v', '--verbosity', action="count", default=0,
                            help="increase output verbosity (e.g., -vv is more than -v)")
    parser = LesionSegLightningTiramisu.add_testing_arguments(parser)
    parser = LesionSegDataModulePredict.add_arguments(parser)
    parser = Trainer.add_argparse_args(parser)
    trainer_options = set(inspect.signature(Trainer).parameters.keys())  # noqa
    unnecessary_options = trainer_options - {'gpus', 'fast_dev_run', 'default_root_dir'}
    remove_args(parser, unnecessary_options)
    fix_type_funcs(parser)
    return parser


def predict(args=None):
    parser = predict_parser()
    if args is None:
        args = parser.parse_args(_skip_check=True)  # noqa
    elif isinstance(args, list):
        args = parser.parse_args(args, _skip_check=True)  # noqa
    args = none_string_to_none(args)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    seed_everything(args.seed, workers=True)
    trainer = Trainer.from_argparse_args(args)
    dict_args = vars(args)
    dm = LesionSegDataModulePredict.from_csv(
        **dict_args,
    )
    dm.setup()
    model = LesionSegLightningTiramisu.load_from_checkpoint(
        args.model_path,
    )
    logger.debug(model)
    trainer.predict(model, datamodule=dm)
    exp_dir = get_experiment_directory(args.model_path)
    logger.info(f"Finished prediction.")
    if not args.fast_dev_run:
        generate_predict_config_yaml(exp_dir, parser, dict_args)
    return 0
