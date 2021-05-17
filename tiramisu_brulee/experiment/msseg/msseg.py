#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.msseg

3D Tiramisu network for FLAIR-based T2-lesion segmentation

This code is unfortunately a huge mess. But, given the CSV files with
the appropriate setup, you can run the below command (starting with
"python -u ...") to generate the network used to generate the
segmentation results in the paper:
"A Structural Causal Model for MR Images of Multiple Sclerosis"
https://arxiv.org/abs/2103.03158.

This script requires:
  1. msseg (https://github.com/jcreinhold/msseg)
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

__all__ = []

from typing import *

from argparse import ArgumentParser
from functools import partial
import logging
from os.path import join
import sys
import warnings

import nibabel as nib
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW, RMSprop, lr_scheduler
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import torchio
from torchio.transforms import (
    Compose,
    OneOf,
    RandomAffine,
    RandomElasticDeformation
)

from tiramisu_brulee.experiment.lightningtiramisu import LightningTiramisu
from tiramisu_brulee.experiment.msseg.lesion_tools import *
from tiramisu_brulee.experiment.msseg.util import *
from tiramisu_brulee.data import csv_to_subjectlist
from tiramisu_brulee.loss import binary_combo_loss


class MSLightningTiramisu(LightningTiramisu):

    def __init__(self,
                 *args,
                 train_subject_list: List[torchio.Subject] = None,
                 valid_subject_list: List[torchio.Subject] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if self._use_multitask_w_head:
            hparams['network_params']['out_channels'] = out_channels
            self.syn_head = Head(hparams['misc_params']['head_size'], 1)
            self.seg_head = Head(hparams['misc_params']['head_size'], 1)
        self.train_subject_list = train_subject_list
        self.valid_subject_list = valid_subject_list
        self.isbiscore = ISBIScore('isbi15_score')

    @classmethod
    def from_csv(cls, train_csv: str, valid_csv: str, *args, **kwargs):
        tsl = csv_to_subjectlist(train_csv)
        vsl = csv_to_subjectlist(valid_csv)
        return cls(*args, train_subject_list=tsl, valid_subject_list=vsl, **kwargs)

    def setup_loss(self, name: str, combo_weight: float = None):
        if name == 'combo':
            self.seg_loss = binary_combo_loss
            self.train_criterion = partial(self.seg_loss, weight=combo_weight)
            self.valid_criterion = partial(self.seg_loss, weight=combo_weight)
        elif name == 'mse':
            self.seg_loss = mse_segmentation_loss
            self.train_criterion = self.seg_loss
            self.valid_criterion = self.seg_loss
        elif name == 'l1':
            self.seg_loss = l1_segmentation_loss
            self.train_criterion = self.seg_loss
            self.valid_criterion = self.seg_loss
        else:
            raise ValueError(f'{name} not a supported loss function.')

    @property
    def _use_multitask_w_head(self):
        return self.use_multitask and self.head_size > 0

    def configure_optimizers(self):
        if self.use_rmsprop:
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

    def _setup_train_dataset(self):
        if self.use_aug:
            spatial = OneOf(
                {RandomAffine(): 0.8,
                 RandomElasticDeformation(): 0.2},
                p=0.75,
            )
            transforms = [spatial]
            transform = Compose(transforms)
            subjects_dataset = torchio.SubjectsDataset(
                self.train_subject_list,
                transform=transform,
            )
        else:
            subjects_dataset = torchio.SubjectsDataset(
                self.train_subject_list,
            )
        return subjects_dataset

    def _setup_sampler(self):
        if self.use_label_sampler:
            sampler = torchio.data.LabelSampler(
                self.patch_size,
            )
        else:
            sampler = torchio.data.UniformSampler(
                self.patch_size,
            )
        return sampler

    def train_dataloader(self):
        subjects_dataset = self._setup_train_dataset()
        sampler = self._setup_sampler()
        patches_queue = torchio.Queue(
            subjects_dataset,
            self.queue_length,
            self.samples_per_volume,
            sampler,
            num_workers=self.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True
        )
        train_dataloader = DataLoader(
            patches_queue,
            batch_size=self.batch_size
        )
        return train_dataloader

    def _setup_val_dataset(self):
        subjects_dataset = torchio.SubjectsDataset(
            self.valid_subject_list,
            transform=torchio.CropOrPad(
                self.valid_patch_size
            )
        )
        return subjects_dataset

    def val_dataloader(self):
        subjects_dataset = self._setup_val_dataset()
        val_dataloader = DataLoader(
            subjects_dataset,
            batch_size=self.valid_batch_size
        )
        return val_dataloader

    def _collate_batch(self, batch):
        fl = batch['flair'][torchio.DATA]
        y = batch['label'][torchio.DATA]
        if self._use_pd:
            pd = batch['pd'][torchio.DATA]
        if self._use_weight:
            w = batch['weight']
        with torch.no_grad():
            if self._use_pd:
                x = torch.cat((fl, pd), 1)
            else:
                x = fl
            if 'div' in batch:
                x /= batch['div'].view(-1, 1, 1, 1, 1)
        out = (x, y, w) if self._use_weight else (x, y)
        return out

    def _collate_multitask_batch(self, batch):
        t1 = batch['t1'][torchio.DATA]
        fl = batch['flair'][torchio.DATA]
        y = batch['label'][torchio.DATA]
        if self._use_pd:
            pd = batch['pd'][torchio.DATA]
        if self._use_weight:
            w = batch['weight']
        with torch.no_grad():
            if self._use_pd:
                x = torch.cat((fl, pd), 1)
            else:
                x = fl
            if 'div' in batch:
                x /= batch['div'].view(-1, 1, 1, 1, 1)
        out = (x, y, t1, w) if self._use_weight else (x, y, t1)
        return out

    def _mixup_beta(self, n):
        alpha = self._mixup_alpha
        m = torch.distributions.beta.Beta(alpha, alpha)
        return m.sample((n, 1, 1, 1, 1))

    def _asymmetric_mixup_threshold(self, y, b):
        if self._mixup_margin > 0.:
            choose_orig = b > self._mixup_margin
            choose_perm = (1. - b) > self._mixup_margin
            mask = (y == (1. - b)) * choose_perm + (y == b) * choose_orig
            mask = (y == 1.) | mask
            y *= 0.
            y[mask] = 1.
        return y

    def mix(self, x, y, w=None):
        with torch.no_grad():
            if self._use_mixup:
                n = x.size(0)
                rp = torch.randperm(n)
                b = self._mixup_beta(n).to(x.device)
                x_perm = x[rp].clone()
                x = b * x + (1 - b) * x_perm
                y = y.float()
                y_perm = y[rp].clone()
                y = b * y + (1 - b) * y_perm
                y = self._asymmetric_mixup_threshold(y, b)
                if w is not None:
                    w_perm = w[rp].clone()
                    w = b * w + (1 - b) * w_perm
            if self._mix_to_one:
                x, y = x[0:1], y[0:1]
        return (x, y) if w is None else (x, y, w)

    def multitask_mix(self, x, seg, syn, w=None):
        with torch.no_grad():
            if self._use_mixup:
                n = x.size(0)
                rp = torch.randperm(n)
                b = self._mixup_beta(n).to(x.device)
                x_perm = x[rp].clone()
                x = b * x + (1 - b) * x_perm
                seg = seg.float()
                seg_perm = seg[rp].clone()
                seg = b * seg + (1 - b) * seg_perm
                syn_perm = syn[rp].clone()
                syn = b * syn + (1 - b) * syn_perm
                seg = self._asymmetric_mixup_threshold(seg, b)
                if w is not None:
                    w_perm = w[rp].clone()
                    w = b * w + (1 - b) * w_perm
                if self._mix_to_one:
                    x, seg, syn = x[0:1], seg[0:1], syn[0:1]
        out = (x, seg, syn) if w is None else (x, seg, syn, w)
        return out

    def _training_step(self, batch):
        batch = self._collate_batch(batch)
        if self._use_weight:
            x, y, w = self.mix(*batch)
        else:
            x, y = self.mix(*batch)
        y_hat = self(x)
        loss = self.train_criterion(y_hat, y, reduction='none')
        if self._use_weight:
            mean_dims = tuple(range(1 - len(x.shape), 0))
            loss = torch.mean(loss, dim=mean_dims)
            loss *= w
        loss = loss.mean()
        tensorboard_logs = dict(
            train_loss=loss)
        return {'loss': loss, 'log': tensorboard_logs}

    def _multitask_training_step(self, batch):
        batch = self._collate_multitask_batch(batch)
        if self._use_weight:
            x, y, t1, w = self.multitask_mix(*batch)
        else:
            x, y, t1 = self.multitask_mix(*batch)
        if self._use_multitask_w_head():
            inter_x = self(x)
            y_hat = self.seg_head(inter_x)
            t1_hat = self.syn_head(inter_x)
        else:
            y_hat, t1_hat = torch.chunk(self(x), 2, dim=1)
        seg_loss = self.train_criterion(y_hat, y, reduction='none')
        syn_loss = self.train_syn_criterion(t1_hat, t1, reduction='none')
        loss = seg_loss + self._syn_weight * syn_loss
        if self._use_weight:
            mean_dims = tuple(range(1 - len(x.shape), 0))
            loss = torch.mean(loss, dim=mean_dims)
            loss *= w
        loss = loss.mean()
        seg_loss = seg_loss.mean()
        syn_loss = syn_loss.mean()
        tensorboard_logs = dict(
            train_loss=loss,
            train_seg_loss=seg_loss,
            train_syn_loss=syn_loss)
        return {'loss': loss, 'log': tensorboard_logs}

    def training_step(self, batch, batch_idx):
        if self._use_multitask:
            out_dict = self._multitask_training_step(batch)
        else:
            out_dict = self._training_step(batch)
        return out_dict

    def _validation_step(self, batch):
        if self._use_weight:
            x, y, w = self._collate_batch(batch)
        else:
            x, y = self._collate_batch(batch)
        y_hat = self(x)
        loss = self.valid_criterion(y_hat, y, reduction='none')
        if self._use_weight:
            mean_dims = tuple(range(1 - len(x.shape), 0))
            loss = torch.mean(loss, dim=mean_dims)
            loss *= w
        loss = loss.mean()
        with torch.no_grad():
            y_hat_ = torch.sigmoid(y_hat) > self._threshold
        isbiscore = self.isbiscore(y_hat_, y)
        isbiscore = isbiscore.to(loss.device)
        out_dict = dict(
            val_loss=loss,
            val_isbi15_score=isbiscore)
        return out_dict, (x, y, y_hat)

    def _multitask_validation_step(self, batch):
        if self._use_weight:
            x, y, t1, w = self._collate_multitask_batch(batch)
        else:
            x, y, t1 = self._collate_multitask_batch(batch)
        if self._use_multitask_w_head():
            inter_x = self(x)
            y_hat = self.seg_head(inter_x)
            t1_hat = self.syn_head(inter_x)
        else:
            y_hat, t1_hat = torch.chunk(self(x), 2, dim=1)
        seg_loss = self.valid_criterion(y_hat, y, reduction='none')
        syn_loss = self.valid_syn_criterion(t1_hat, t1, reduction='none')
        loss = seg_loss + self._syn_weight * syn_loss
        if self._use_weight:
            mean_dims = tuple(range(1 - len(x.shape), 0))
            loss = torch.mean(loss, dim=mean_dims)
            loss *= w
        loss = loss.mean()
        seg_loss = seg_loss.mean()
        syn_loss = syn_loss.mean()
        with torch.no_grad():
            y_hat_ = torch.sigmoid(y_hat) > self._threshold
        isbiscore = self.isbiscore(y_hat_, y)
        isbiscore = isbiscore.to(loss.device)
        out_dict = dict(
            val_loss=loss,
            val_isbi15_score=isbiscore,
            val_seg_loss=seg_loss,
            val_syn_loss=syn_loss)
        return out_dict, (x, y, t1, y_hat, t1_hat)

    def validation_step(self, batch, batch_idx):
        if self._use_multitask:
            out_dict, imgs = self._multitask_validation_step(batch)
            x, y, t1, y_hat, t1_hat = imgs
            if batch_idx == 0:
                with torch.no_grad():
                    mid_slice = x.shape[-1] // 2
                    fl_ = minmax_scale_batch(x[:, 0:1, :, :, mid_slice])
                    if self._use_pd:
                        pd_ = minmax_scale_batch(x[:, 1:2, :, :, mid_slice])
                    t1_ = minmax_scale_batch(t1[..., mid_slice])
                    t1_hat_ = minmax_scale_batch(t1_hat[..., mid_slice])
                    if self._softmask:
                        y_hat_ = minmax_scale_batch(torch.sigmoid(y_hat[..., mid_slice]))
                        y_ = minmax_scale_batch(y[..., mid_slice])
                    else:
                        y_hat_ = torch.sigmoid(y_hat[..., mid_slice]) > self._threshold
                        y_ = y[..., mid_slice] > 0.
                    n = self.current_epoch
                self.logger.experiment.add_images('t1', t1_, n, dataformats='NCHW')
                self.logger.experiment.add_images('flair', fl_, n, dataformats='NCHW')
                self.logger.experiment.add_images('y', y_, n, dataformats='NCHW')
                self.logger.experiment.add_images('t1_hat', t1_hat_, n, dataformats='NCHW')
                self.logger.experiment.add_images('y_hat', y_hat_, n, dataformats='NCHW')
                if self._use_pd:
                    self.logger.experiment.add_images('pd', pd_, n, dataformats='NCHW')
        else:
            out_dict, imgs = self._validation_step(batch)
            x, y, y_hat = imgs
            if batch_idx == 0:
                with torch.no_grad():
                    mid_slice = x.shape[-1] // 2
                    t1_ = minmax_scale_batch(x[:, 0:1, :, :, mid_slice])
                    fl_ = minmax_scale_batch(x[:, 2:3, :, :, mid_slice])
                    if self._use_pd:
                        pd_ = minmax_scale_batch(x[:, 3:4, :, :, mid_slice])
                    if self._softmask:
                        y_hat_ = minmax_scale_batch(torch.sigmoid(y_hat[..., mid_slice]))
                        y_ = minmax_scale_batch(y[..., mid_slice])
                    else:
                        y_hat_ = torch.sigmoid(y_hat[..., mid_slice]) > self._threshold
                        y_ = y[..., mid_slice] > 0.
                    n = self.current_epoch
                self.logger.experiment.add_images('t1', t1_, n, dataformats='NCHW')
                self.logger.experiment.add_images('flair', fl_, n, dataformats='NCHW')
                self.logger.experiment.add_images('pred', y_hat_, n, dataformats='NCHW')
                self.logger.experiment.add_images('truth', y_, n, dataformats='NCHW')
                if self._use_pd:
                    self.logger.experiment.add_images('pd', pd_, n, dataformats='NCHW')
        return out_dict

    @staticmethod
    def _cat(x: Tensor) -> Tensor:
        try:
            return torch.cat(x)
        except RuntimeError:
            return torch.tensor(x)

    def validation_epoch_end(self, outputs):
        avg_loss = self._cat([x['val_loss'] for x in outputs]).mean()
        avg_isbi15_score = self._cat([x['val_isbi15_score'] for x in outputs]).mean()
        if self._use_multitask:
            avg_seg_loss = self._cat([x['val_seg_loss'] for x in outputs]).mean()
            avg_syn_loss = self._cat([x['val_syn_loss'] for x in outputs]).mean()
            avg_isbi15_score_minus_loss = self._isbi15score_weight * avg_isbi15_score - avg_seg_loss
            tensorboard_logs = {'avg_val_loss': avg_loss,
                                'avg_val_isbi15_score': avg_isbi15_score,
                                'avg_val_isbi15_score_minus_loss': avg_isbi15_score_minus_loss,
                                'avg_val_seg_loss': avg_seg_loss,
                                'avg_val_syn_loss': avg_syn_loss}
        else:
            avg_isbi15_score_minus_loss = self._isbi15score_weight * avg_isbi15_score - avg_loss
            tensorboard_logs = {'avg_val_loss': avg_loss,
                                'avg_val_isbi15_score': avg_isbi15_score,
                                'avg_val_isbi15_score_minus_loss': avg_isbi15_score_minus_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

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
            if self._use_multitask:
                if self._use_multitask_w_head():
                    logits = self.seg_head(self(x))
                else:
                    logits, _ = torch.chunk(self(x), 2, dim=1)
            else:
                logits = self(x)
            probits = torch.sigmoid(logits)
        out[h1:h2, w1:w2, d1:d2] = probits.detach().cpu()
        torch.cuda.empty_cache()
        return out

    def process_img_patches(self, sample, patch_overlap=None):
        patch_size = self.patch_size
        batch_size = self.batch_size
        if patch_overlap is None:
            patch_overlap = patch_size // 2
        grid_sampler = torchio.inference.GridSampler(
            sample,
            patch_size,
            patch_overlap,
            padding_mode='replicate'
        )
        patch_loader = torch.utils.data.DataLoader(
            grid_sampler,
            batch_size=batch_size
        )
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        self.eval()
        with torch.no_grad():
            for patches_batch in patch_loader:
                fl = patches_batch['flair'][torchio.DATA]
                xs = [fl]
                if self._use_pd:
                    pd_fn = str(sample['pd'].path)
                    pd = nib.load(pd_fn).get_fdata(dtype=np.float32)
                    xs.append(pd)
                x = torch.cat(xs, 1).to(self.device)
                locations = patches_batch[torchio.LOCATION]
                if self._use_multitask:
                    if self._use_multitask_w_head():
                        logits = self.seg_head(self(x))
                    else:
                        logits, _ = torch.chunk(self(x), 2, dim=1)
                else:
                    logits = self(x)
                probits = torch.sigmoid(logits)
                aggregator.add_batch(probits, locations)
        out = aggregator.get_output_tensor().detach().cpu()
        torch.cuda.empty_cache()
        return out


################################### Main #######################################
def main(args=None):
    if args is None:
        parser = arg_parser()
        args = parser.parse_args()
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    seed_everything(args.seed)
    model = None
    if args.train_csv is not None and args.valid_csv is not None:
        train_subject_list = csv_to_subjectlist(args.train_csv)
        valid_subject_list = csv_to_subjectlist(args.valid_csv)
        model = MSLightningTiramisu(
            exp_config,
            train_subject_list,
            valid_subject_list)
        model.setup_loss(args.loss_function, args.combo_weight)
        if args.use_multitask:
            model.syn_loss = F.l1_loss
            model.train_syn_criterion = model.syn_loss
            model.valid_syn_criterion = model.syn_loss
        n_epochs = exp_config.lightning_params['n_epochs']
        logger.info(model)
        gpu_kwargs = dict(gpus=2, distributed_backend='dp') if args.multigpu else \
            dict(gpus=[1])
        checkpoint_callback = ModelCheckpoint(
            monitor='avg_val_isbi15_score_minus_loss',
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
            if args.use_morph_gac:
                logger.info('Starting morphological geodesic active contour')
                fl_fn = str(test_subj['flair'].path)
                fl = nib.load(fl_fn).get_fdata(dtype=np.float32)
                flg = inverse_gaussian_gradient(fl)
                seg_gac = morphological_geodesic_active_contour(
                    flg, 100, init_level_set=seg_data)
                seg_gac_fn = join(args.out_path, name + '_seg_gac.nii.gz')
                seg_gac_nii = nib.Nifti1Image(
                    seg_gac,
                    in_nii.affine,
                    in_nii.header)
                seg_gac_nii.to_filename(seg_gac_fn)
            if args.use_morph_acwe:
                logger.info('Starting morphological Chan-Vese')
                fl_fn = str(test_subj['flair'].path)
                fl = nib.load(fl_fn).get_fdata(dtype=np.float32)
                seg_acwe = morphological_chan_vese(
                    fl, 100, init_level_set=seg_data)
                seg_acwe_fn = join(args.out_path, name + '_seg_acwe.nii.gz')
                seg_acwe_nii = nib.Nifti1Image(
                    seg_acwe,
                    in_nii.affine,
                    in_nii.header)
                seg_acwe_nii.to_filename(seg_acwe_fn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
