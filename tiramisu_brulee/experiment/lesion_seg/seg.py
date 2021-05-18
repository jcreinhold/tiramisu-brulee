#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_seg

3D Tiramisu network for FLAIR-based T2-lesion segmentation

This code is unfortunately a huge mess. But, given the CSV files with
the appropriate setup, you can run the below command (starting with
"python -u ...") to generate the network used to generate the
segmentation results in the paper:
"A Structural Causal Model for MR Images of Multiple Sclerosis"
https://arxiv.org/abs/2103.03158.

This script requires:
  1. lesion_seg (https://github.com/jcreinhold/msseg)
  2. lesionqc (https://github.com/jcreinhold/lesionqc)
and the various  other packages listed in the imports.

The CSV file is setup with "flair" and "t1" as headers and then
the full path to the NIfTI files (to the corresponding FLAIR and T1-w
images) are the rows. We used the ISBI 2015 and MICCAI 2016 Challenge
Data, as well as some private labeled data, to train the network.

python -u tiramisu3d_only_flair.py \
    --train-csv csv/all/train_weighted.csv \
    --valid-csv csv/all/valid_weighted.csv \
    -ic 1 \
    --use-multitask \
    --use-mixup \
    --use-aug \
    -db 4 4 4 4 \
    -ub 4 4 4 4 \
    -gr 12 \
    -bnl 4 \
    -lr 0.0002 \
    -bt 0.8 0.99 \
    -bs 7 \
    -ps 64 64 64 \
    -vbs 7 \
    -vps 96 96 96 \
    -da 25 \
    -ne 100 \
    -dr 0.1 \
    -wd 0.000001 \
    -ma 0.4 \
    -mm 0.8 \
    -spv 10 \
    -v

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 14, 2021
"""

__all__ = [
    'LesionSegLightningTiramisu',
]

from typing import *

import argparse
from functools import partial
import logging
from os.path import join
import sys
import warnings

import nibabel as nib
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import Tensor
from torch.optim import AdamW, RMSprop, lr_scheduler
from torch.utils.data import DataLoader
import torchio as tio

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
                 mixup_alpha: float = 0.4):
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
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if self.loss_function == 'combo':
            self.criterion = partial(binary_combo_loss, weight=self.combo_weight)
        elif self.loss_function == 'mse':
            self.criterion = mse_segmentation_loss
        elif self.loss_function == 'l1':
            self.criterion = l1_segmentation_loss
        else:
            raise ValueError(f'{self.loss_function} not a supported loss function.')
        if self.mixup:
            self._mix = Mixup(self.mixup_alpha)

    def training_step(self, batch: Dict[Tensor], batch_idx: int) -> dict:
        src, tgt = batch
        if self.mixup:
            src, tgt = self._mix(src, tgt)
        pred = self(src)
        loss = self.criterion(pred, tgt)
        tensorboard_logs = dict(train_loss=loss)
        return dict(loss=loss, log=tensorboard_logs)

    def validation_step(self, batch: Dict[Tensor], batch_idx: int) -> Tuple[dict, dict]:
        src, tgt = batch
        pred = self(src)
        loss = self.criterion(pred, tgt)
        with torch.no_grad():
            pred_seg = torch.sigmoid(pred) > self.threshold
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
                if self.soft_labels and field == 'pred':
                    img = torch.sigmoid(img)
                elif field == 'pred':
                    img = torch.sigmoid(img) > self.threshold
                elif field == 'truth':
                    img = img > 0.
                else:
                    img = minmax_scale_batch(img)
                self.logger.experiment.add_images(field, img, n, dataformats='NCHW')
        return metrics

    def validation_epoch_end(self, outputs: dict) -> dict:
        avg_loss = self._extract_and_average(outputs, 'val_loss')
        avg_isbi15_score = self._extract_and_average(outputs, 'val_isbi15_score')
        tensorboard_logs = dict(
            avg_val_loss=avg_loss,
            avg_val_isbi15_score=avg_isbi15_score,
        )
        return dict(val_loss=avg_loss, log=tensorboard_logs)

    def configure_optimizers(self):
        if self.rmsprop:
            momentum, alpha = self.betas
            optimizer = RMSprop(
                self.parameters(),
                lr=self.lr,
                momentum=momentum,
                alpha=alpha,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = AdamW(
                self.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - self.decay_after) / float(self.n_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return [optimizer], [scheduler]

    def process_whole_img(self, sample):
        fl_fn = str(sample['flair'].path)
        fl = nib.load(fl_fn).get_fdata(dtype=np.float32)
        xs = [fl]
        if self._use_pd:
            pd_fn = str(sample['pd'].path)
            pd = nib.load(pd_fn).get_fdata(dtype=np.float32)
            xs.append(pd)
        out = torch.from_numpy(np.zeros_like(fl))
        h1, h2, w1, w2, d1, d2 = bbox3D(fl > 0.)
        xs = [x[h1:h2, w1:w2, d1:d2] for x in xs]
        x = np.stack(xs)[None, ...]
        x = torch.from_numpy(x).to(self.device)
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probits = torch.sigmoid(logits)
        out[h1:h2, w1:w2, d1:d2] = probits.detach().cpu()
        torch.cuda.empty_cache()
        return out

    def process_img_patches(self, sample, patch_overlap=None):
        if patch_overlap is None:
            patch_overlap = self.patch_size // 2
        grid_sampler = tio.GridSampler(
            sample,
            self.patch_size,
            patch_overlap,
            padding_mode='replicate'
        )
        patch_loader = torch.utils.data.DataLoader(
            grid_sampler,
            batch_size=self.batch_size
        )
        aggregator = tio.GridAggregator(grid_sampler)
        self.eval()
        with torch.no_grad():
            for patches_batch in patch_loader:
                fl = patches_batch['flair'][tio.DATA]
                xs = [fl]
                if self._use_pd:
                    pd_fn = str(sample['pd'].path)
                    pd = nib.load(pd_fn).get_fdata(dtype=np.float32)
                    xs.append(pd)
                x = torch.cat(xs, 1).to(self.device)
                locations = patches_batch[tio.LOCATION]
                logits = self(x)
                probits = torch.sigmoid(logits)
                aggregator.add_batch(probits, locations)
        out = aggregator.get_output_tensor().detach().cpu()
        torch.cuda.empty_cache()
        return out

    @staticmethod
    def add_model_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument('-ic', '--in-channels', type=positive_int, default=1,
                            help='number of input channels')
        parser.add_argument('-oc', '--out-channels', type=positive_int, default=1,
                            help='number of output channels')
        parser.add_argument('-dr', '--dropout-rate', type=positive_float, default=0.1,
                            help='dropout rate/probability')
        parser.add_argument('-it', '--init-type', type=ascii, default='he_uniform',
                            choices=('normal', 'xavier_normal', 'he_normal', 'he_uniform', 'orthogonal'),
                            help='use this type of initialization for the network')
        parser.add_argument('-ig', '--init-gain', type=positive_float, default=0.2,
                            help='use this initialization gain for initialization')
        parser.add_argument('-db', '--down-blocks', type=positive_int, default=(4, 4, 4, 4, 4), nargs='+',
                            help='tiramisu down block specification')
        parser.add_argument('-ub', '--up-blocks', type=positive_int, default=(4, 4, 4, 4, 4), nargs='+',
                            help='tiramisu up block specification')
        parser.add_argument('-bl', '--bottleneck-layers', type=positive_int, default=4,
                            help='tiramisu bottleneck specification')
        parser.add_argument('-gr', '--growth-rate', type=positive_int, default=12,
                            help='tiramisu growth rate specification')
        parser.add_argument('-fcoc', '--first-conv-out-channels', type=positive_int, default=48,
                            help='number of output channels in first conv')
        return parser

    @staticmethod
    def add_training_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Training")
        parser.add_argument('-bt', '--betas', type=positive_float, default=(0.9, 0.999), nargs=2,
                            help='AdamW momentum parameters (or, for RMSprop, momentum and alpha params)')
        parser.add_argument('-cw', '--combo-weight', type=positive_float, default=0.6,
                            help='weight of positive class in combo loss')
        parser.add_argument('-da', '--decay-after', type=positive_int, default=8,
                            help='decay learning rate after this number of epochs')
        parser.add_argument('-lr', '--learning-rate', type=positive_float, default=3e-4,
                            help='learning rate for the optimizer')
        parser.add_argument('-lf', '--loss-function', type=ascii, default='combo',
                            choices=('combo', 'l1', 'mse'),
                            help='loss function to train the network')
        parser.add_argument('-ne', '--n-epochs', type=positive_int, default=64,
                            help='number of epochs')
        parser.add_argument('-rp', '--rmsprop', action='store_true', default=False,
                            help="use rmsprop instead of adam")
        parser.add_argument('-wd', '--weight-decay', type=positive_float, default=1e-5,
                            help="weight decay parameter for adamw")
        parser.add_argument('-sl', '--soft-labels', action='store_true', default=False,
                            help="use soft labels (i.e., non-binary labels) for training")
        return parser

    @staticmethod
    def add_other_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Other")
        parser.add_argument('-th', '--threshold', type=threshold, default=0.5,
                            help='probability threshold for segmentation')
        return parser

    @staticmethod
    def add_testing_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Testing")
        parser.add_argument('-mls', '--min-lesion-size', type=nonnegative_int, default=3,
                            help='in testing, remove lesions smaller in voxels than this')
        parser.add_argument('-nfh', '--no-fill-holes', action='store_false', default=True,
                            help='in testing, do not preform binary hole filling')
        return parser


def arg_parser():
    desc = 'Train and/or use a Tiramisu CNN for segmentation of lesions'
    parser = argparse.ArgumentParser(
        prog='lesion-seg',
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    exp_parser = parser.add_argument_group('Experiment')
    exp_parser.add_argument('--trained-model-path', type=file, default=None,
                            help='path to output the trained model')
    exp_parser.add_argument('-mg', '--multigpu', action='store_true', default=False,
                            help='use multiple gpus')
    exp_parser.add_argument('-rs', '--resume', type=file, default=None,
                            help='resume from this path')
    exp_parser.add_argument('-sd', '--seed', type=int, default=0,
                            help='set seed for reproducibility')
    exp_parser.add_argument('-v', '--verbosity', action="count", default=0,
                            help="increase output verbosity (e.g., -vv is more than -v)")
    parser = LesionSegLightningTiramisu.add_model_arguments(parser)
    parser = LesionSegLightningTiramisu.add_training_arguments(parser)
    parser = LesionSegLightningTiramisu.add_other_arguments(parser)
    parser = LesionSegLightningTiramisu.add_testing_arguments(parser)
    parser = LesionSegDataModule.add_arguments(parser)
    parser = Mixup.add_arguments(parser)
    return parser


def main(args=None):
    if args is None:
        parser = arg_parser()
        args = parser.parse_args()
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    seed_everything(args.seed)
    model = LesionSegLightningTiramisu(
        args
    )
    logger.info(model)
        gpu_kwargs = dict(gpus=2, distributed_backend='dp') if args.multigpu else \
            dict(gpus=[1])
        checkpoint_callback = ModelCheckpoint(
            monitor='avg_val_isbi15_score',
            save_top_k=1,
            save_last=True,
            mode='max'
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = Trainer(
                benchmark=True,
                check_val_every_n_epoch=1,
                accumulate_grad_batches=1,
                min_epochs=args.n_epochs,
                max_epochs=args.n_epochs,
                checkpoint_callback=checkpoint_callback,
                resume_from_checkpoint=args.resume,
                fast_dev_run=False,
                **gpu_kwargs
            )
            torch.cuda.empty_cache()
            trainer.fit(model)
    if args.test_csv is not None:
        device = torch.device('cuda:0')
        if model is None:
            logger.info(f"Loading: {args.trained_model_path}")
            model = MSLightningTiramisu.load_from_checkpoint(
                args.trained_model_path
            )
        model.to(device)
        torch.cuda.empty_cache()
        test_subject_list = csv_to_subjectlist(args.test_csv)
        for test_subj in test_subject_list:
            name = test_subj.name
            logger.info(f'Processing: {name}')
            if args.patch_size == [0, 0, 0]:
                output = model.process_whole_img(test_subj)
            else:
                output = model.process_img_patches(test_subj)
            prob_data = output.numpy().squeeze()
            seg_data = prob_data > args.threshold
            seg_data = clean_segmentation(
                seg_data,
                args.binary_hole_fill,
                args.min_lesion_size
            ).astype(np.float32)
            prob_fn = join(args.out_path, name + '_prob.nii.gz')
            seg_fn = join(args.out_path, name + '_seg.nii.gz')
            in_fn = str(test_subj['flair'].path)
            in_nii = nib.load(in_fn)
            in_data = in_nii.get_fdata()
            assert in_data.shape == prob_data.shape, f"In: {in_data.shape} != Out: {prob_data.shape}"
            prob_nii = nib.Nifti1Image(
                prob_data,
                in_nii.affine,
                in_nii.header)
            prob_nii.to_filename(prob_fn)
            seg_nii = nib.Nifti1Image(
                seg_data,
                in_nii.affine,
                in_nii.header)
            seg_nii.to_filename(seg_fn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
