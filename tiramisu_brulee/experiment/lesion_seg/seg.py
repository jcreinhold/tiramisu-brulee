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
from typing import *

from jsonargparse import ArgumentParser, ActionConfigFile
import nibabel as nib
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch import Tensor
from torch.optim import AdamW, RMSprop, lr_scheduler

from tiramisu_brulee.experiment.lightningtiramisu import LightningTiramisu
from tiramisu_brulee.experiment.lesion_seg.data import *
from tiramisu_brulee.experiment.lesion_seg.lesion_tools import *
from tiramisu_brulee.experiment.lesion_seg.parse import *
from tiramisu_brulee.experiment.lesion_seg.util import *
from tiramisu_brulee.loss import (
    binary_combo_loss,
    l1_segmentation_loss,
    mse_segmentation_loss,
)

EXPERIMENT_NAME = 'lesion_tiramisu_experiment'


class LesionSegLightningTiramisu(LightningTiramisu):

    def __init__(self,
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
                 lr: float = 1e-3,
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
                 **kwargs):
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
            lr,
            betas,
            weight_decay,
        )
        # self.combo_weight = combo_weight
        # self.decay_after = decay_after
        # self.hparams.loss_function = loss_function
        # self.hparams.rmsprop = rmsprop
        # self.soft_labels = soft_labels
        # self.hparams.threshold = threshold
        # self.min_lesion_size = min_lesion_size
        # self.fill_holes = fill_holes
        # self.hparams.mixup = mixup
        # self.hparams.mixup_alpha = mixup_alpha
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if self.hparams.loss_function == 'combo':
            self.criterion = partial(
                binary_combo_loss,
                weight=self.hparams.combo_weight
            )
        elif self.hparams.loss_function == 'mse':
            self.criterion = mse_segmentation_loss
        elif self.hparams.loss_function == 'l1':
            self.criterion = l1_segmentation_loss
        else:
            raise ValueError(f'{self.hparams.loss_function} not a supported loss function.')
        if self.hparams.mixup:
            self._mix = Mixup(self.hparams.mixup_alpha)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        src, tgt = batch
        if self.hparams.mixup:
            src, tgt = self._mix(src, tgt)
        pred = self(src)
        loss = self.criterion(pred, tgt)
        tensorboard_logs = dict(train_loss=loss)
        return dict(loss=loss, log=tensorboard_logs)

    def validation_step(self, batch: dict, batch_idx: int) -> Tuple[dict, dict]:
        src, tgt = batch
        pred = self(src)
        loss = self.criterion(pred, tgt)
        with torch.no_grad():
            pred_seg = torch.sigmoid(pred) > self.hparams.threshold
        isbi15_score = almost_isbi15_score(pred_seg, tgt)
        metrics = dict(
            val_loss=loss,
            val_isbi15_score=isbi15_score,
        )
        images = dict(
            truth=tgt,
            pred=pred,
        )
        for i, field in enumerate(self._input_fields):
            images[field] = src[:, i:i + 1, ...]
        return metrics, images

    def validation_step_end(self, val_step_outputs: dict) -> dict:
        metrics, images = val_step_outputs
        with torch.no_grad():
            n = self.current_epoch
            mid_slice = None
            for field in self._input_fields + ('truth', 'pred'):
                if mid_slice is None:
                    mid_slice = images[field].shape[-1] // 2
                img = images[field][..., mid_slice]
                if self.hparams.soft_labels and field == 'pred':
                    img = torch.sigmoid(img)
                elif field == 'pred':
                    img = torch.sigmoid(img) > self.hparams.threshold
                elif field == 'truth':
                    img = img > 0.
                else:
                    img = minmax_scale_batch(img)
                self.logger.experiment.add_images(field, img, n, dataformats='NCHW')
        return metrics

    def validation_epoch_end(self, outputs: dict) -> dict:
        avg_loss = extract_and_average(outputs, 'val_loss')
        avg_isbi15_score = extract_and_average(outputs, 'val_isbi15_score')
        tensorboard_logs = dict(
            avg_val_loss=avg_loss,
            avg_val_isbi15_score=avg_isbi15_score,
        )
        return dict(val_loss=avg_loss, log=tensorboard_logs)

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None) -> dict:
        src = batch['src']
        bbox = BoundingBox3D.from_batch(src)
        src = bbox(src)
        pred = self(src)
        pred_seg = torch.sigmoid(pred) > self.hparams.threshold
        pred_seg = bbox.uncrop_batch(pred_seg)
        pred_seg = pred_seg.float()
        return pred_seg

    def on_predict_batch_end(self, pred_step_outputs: Tensor, batch: dict, batch_idx: int, dataloader_idx: int):
        pred_step_outputs = to_np(pred_step_outputs).squeeze()
        for pred_seg in pred_step_outputs:
            clean_segmentation(pred_seg, )
        affine_matrices = batch['affine']
        out_fns = batch['out']
        for pred, affine, fn in zip(pred_step_outputs, affine_matrices, out_fns):
            logging.info(f'Saving {fn}.')
            nib.Nifti1Image(pred, affine).to_filename(fn)

    def configure_optimizers(self):
        if self.hparams.rmsprop:
            momentum, alpha = self.hparams.betas
            optimizer = RMSprop(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=momentum,
                alpha=alpha,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                betas=self.hparams.betas,
                weight_decay=self.hparams.weight_decay,
            )

        def lambda_rule(epoch):
            numerator = max(0, epoch - self.hparams.decay_after)
            denominator = float(self.hparams.n_epochs + 1)
            lr = 1.0 - numerator / denominator
            return lr

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return [optimizer], [scheduler]

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

    @staticmethod
    def add_training_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Training")
        parser.add_argument('-bt', '--betas', type=positive_float(), default=[0.9, 0.99], nargs=2,
                            help='AdamW momentum parameters (or, for RMSprop, momentum and alpha params)')
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

    @staticmethod
    def add_other_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Other")
        parser.add_argument('-th', '--threshold', type=probability_float(), default=0.5,
                            help='probability threshold for segmentation')
        return parent_parser

    @staticmethod
    def add_testing_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Testing")
        parser.add_argument('-mls', '--min-lesion-size', type=nonnegative_int(), default=3,
                            help='in testing, remove lesions smaller in voxels than this')
        parser.add_argument('-fh', '--fill-holes', action='store_true', default=False,
                            help='in testing, preform binary hole filling')
        return parent_parser


def train_parser():
    desc = 'Train a 3D Tiramisu CNN to segment lesions'
    parser = ArgumentParser(
        prog='lesion-train',
        description=desc,
    )
    parser.add_argument('--config', action=ActionConfigFile)
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
    remove_args(parser, ['logger'])
    return parser


def train(args=None):
    if args is None:
        parser = train_parser()
        args = parser.parse_args()
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    seed_everything(args.seed, workers=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_isbi15_score',
        save_top_k=3,
        save_last=True,
        mode='max',
    )
    tb_logger = TensorBoardLogger(
        args.default_root_dir,
        name=EXPERIMENT_NAME,
    )
    trainer = Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
    )
    dict_args = vars(args)
    dm = LesionSegDataModuleTrain.from_csv(
        args.train_csv,
        args.valid_csv,
        **dict_args,
    )
    dm.setup()
    model = LesionSegLightningTiramisu(**dict_args)
    logger.debug(model)
    trainer.fit(model, dm)
    best_model_path = get_best_model_path(checkpoint_callback)
    exp_dir = get_experiment_directory(best_model_path)
    logger.info(f"Finished training.")
    logger.info(f"Best model path: {best_model_path}")
    generate_train_config_yaml(exp_dir, dict_args)
    generate_predict_config_yaml(exp_dir, dict_args, best_model_path)
    return 0


def predict_parser():
    desc = 'Use a Tiramisu CNN to segment lesions'
    parser = ArgumentParser(
        prog='lesion-predict',
        description=desc,
    )
    parser.add_argument('--config', action=ActionConfigFile)
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
    return parser


def predict(args=None):
    if args is None:
        parser = predict_parser()
        args = parser.parse_args()
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    seed_everything(args.seed, workers=True)
    trainer = Trainer.from_argparse_args(args)
    dict_args = vars(args)
    dm = LesionSegDataModulePredict.from_csv(
        args.predict_csv,
        **dict_args,
    )
    dm.setup()
    model = LesionSegLightningTiramisu.load_from_checkpoint(
        args.model_path,
    )
    logger.debug(model)
    trainer.predict(model, dm)
    exp_dir = get_experiment_directory(args.model_path)
    logger.info(f"Finished prediction.")
    generate_predict_config_yaml(exp_dir, dict_args)
    return 0
