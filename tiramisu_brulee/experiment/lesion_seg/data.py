#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_seg.data

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 17, 2021
"""

__all__ = [
    'LesionSegDataModule',
    'Mixup',
]

from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.distributions as D
import torchio as tio

from tiramisu_brulee.data import csv_to_subjectlist
from tiramisu_brulee.experiment.lesion_seg.parse import file, positive_float, positive_int
from tiramisu_brulee.experiment.lesion_seg.util import reshape_for_broadcasting


class LesionSegDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_subject_list: List[tio.Subject],
        val_subject_list: List[tio.Subject],
        train_patch_size: List[int],
        val_patch_size: List[int],
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        queue_length: int,
        samples_per_volume: int,
        label_sampler: bool = False,
        spatial_augmentation: bool = False,
    ):
        super().__init__()
        self.train_subject_list = train_subject_list
        self.val_subject_list = val_subject_list
        self.train_patch_size = train_patch_size
        self.val_patch_size = val_patch_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume
        self.label_sampler = label_sampler
        self.spatial_augmentation = spatial_augmentation

    @classmethod
    def from_csv(cls, train_csv: str, valid_csv: str, *args, **kwargs):
        tsl = csv_to_subjectlist(train_csv)
        vsl = csv_to_subjectlist(valid_csv)
        return cls(tsl, vsl, *args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        self._determine_input()
        self._setup_train_dataset()
        self._setup_val_dataset()

    def train_dataloader(self) -> DataLoader:
        sampler = self._get_train_sampler()
        patches_queue = tio.Queue(
            self.train_dataset,
            self.queue_length,
            self.samples_per_volume,
            sampler,
            num_workers=self.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        train_dataloader = DataLoader(
            patches_queue,
            batch_size=self.train_batch_size,
            collate_fn=self._collate_fn,
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
        return val_dataloader

    def _get_train_augmentation(self):
        transform = None
        if self.spatial_augmentation:
            spatial = tio.OneOf(
                {tio.RandomAffine(): 0.8,
                 tio.RandomElasticDeformation(): 0.2},
                p=0.75,
            )
            transforms = [spatial]
            transform = tio.Compose(transforms)
        return transform

    def _get_train_sampler(self):
        ps = self.train_patch_size
        sampler = tio.LabelSampler(ps) if self.label_sampler else tio.UniformSampler(ps)
        return sampler

    def _setup_train_dataset(self):
        transform = self._get_train_augmentation()
        subjects_dataset = tio.SubjectsDataset(
            self.train_subject_list,
            transform=transform,
        )
        self.train_dataset = subjects_dataset

    def _get_val_augmentation(self):
        transform = tio.CropOrPad(
            self.val_patch_size
        )
        return transform

    def _setup_val_dataset(self):
        transform = self._get_val_augmentation()
        subjects_dataset = tio.SubjectsDataset(
            self.val_subject_list,
            transform=transform
        )
        self.val_dataset = subjects_dataset

    def _determine_input(self):
        """ assume all columns except `name`, `label`, `div`, or `weight` is an image type """
        exclude = ('name', 'label', 'div', 'weight')
        train_subject = self.train_subject_list[0]  # arbitrarily pick the first element
        valid_subject = self.val_subject_list[0]
        inputs = []
        for key in train_subject:
            if key not in exclude:
                inputs.append(key)
        if len(inputs) == 0:
            msg = ('No inputs detected in training CSV. Expect columns like '
                   '`t1` with corresponding paths to NIfTI files.')
            raise ValueError(msg)
        # if len(inputs) != self.in_channels:
        #    msg = f'Detected {len(inputs)} ({inputs}) but expected {self.in_channels}.'
        #    raise ValueError(msg)
        for key in inputs:
            if key not in valid_subject:
                msg = f'Validation CSV has different fields than training CSV.'
                raise ValueError(msg)
        if 'label' not in train_subject or 'label' not in valid_subject:
            raise ValueError('`label` field expected in both training and validation CSV.')
        if ('div' in train_subject) ^ ('div' in valid_subject):
            msg = ('If `div` present in one of the training or validation CSVs, '
                   'it is expected in both training and validation CSV.')
            raise ValueError(msg)
        if 'div' in train_subject and len(inputs) > 1:
            raise ValueError(f'If using `div`, expect only 1 input. Got {inputs}.')
        self._use_div = 'div' in train_subject and 'div' in valid_subject
        self._input_fields = tuple(sorted(inputs))

    def _div_batch(self, batch: Tensor, div: Tensor) -> Tensor:
        with torch.no_grad():
            batch /= reshape_for_broadcasting(div, batch.ndim)
        return batch

    def _collate_fn(self, batch: Dict[Tensor]) -> Tuple[Tensor, Tensor]:
        inputs = []
        for field in self._input_fields:
            inputs.append(batch[field][tio.DATA])
        src = torch.cat(inputs, dim=1)
        if self._use_div:
            src = self._div_batch(src, batch['div'])
        tgt = batch['label'][tio.DATA]
        return src, tgt

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument('--train-csv', type=file, default=None,
                            help='path to csv with training images')
        parser.add_argument('--valid-csv', type=file, default=None,
                            help='path to csv with validation images')
        parser.add_argument('--test-csv', type=file, default=None,
                            help='path to csv with test images')
        parser.add_argument('-tbs', '--train-batch-size', type=positive_int, default=2,
                            help='training/test batch size')
        parser.add_argument('-vbs', '--val-batch-size', type=positive_int, default=2,
                            help='validation batch size')
        parser.add_argument('-tps', '--train-patch-size', type=positive_int, nargs=3, default=(96, 96, 96),
                            help='training/test patch size extracted from image')
        parser.add_argument('-vps', '--val-patch-size', type=positive_int, nargs=3, default=(128, 128, 128),
                            help='validation patch size extracted from image')
        parser.add_argument('-nw', '--num-workers', type=positive_int, default=16,
                            help='number of CPU processors to use')
        parser.add_argument('-ql', '--queue-length', type=positive_int, default=200,
                            help='queue length for torchio sampler')
        parser.add_argument('-spv', '--samples-per-volume', type=positive_int, default=10,
                            help='samples per volume for torchio sampler')
        parser.add_argument('-ls', '--label-sampler', action='store_true', default=False,
                            help="use label sampler instead of uniform")
        parser.add_argument('-sa', '--spatial-augmentation', action='store_true', default=False,
                            help='use spatial (affine and elastic) data augmentation')
        return parser


class Mixup:

    def __init__(self, alpha: float):
        self.alpha = alpha

    def _mixup_dist(self, device: torch.device) -> D.Distribution:
        alpha = torch.tensor(self.alpha, device=device)
        dist = D.Beta(alpha, alpha)
        return dist

    def _mixup_coef(self, batch_size: int, device: torch.device) -> Tensor:
        dist = self._mixup_dist(device)
        return dist.sample(batch_size)

    def __call__(self, src: Tensor, tgt: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            batch_size = src.shape[0]
            perm = torch.randperm(batch_size)
            lam = self._mixup_coef(batch_size, src.device)
            lam = reshape_for_broadcasting(lam, src.ndim)
            src = lam * src + (1 - lam) * src[perm]
            tgt = tgt.float()
            tgt = lam * tgt + (1 - lam) * tgt[perm]
        return src, tgt

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Mixup")
        parser.add_argument('-mu', '--mixup', action='store_true', default=False,
                            help='use mixup during training')
        parser.add_argument('-ma', '--mixup-alpha', type=positive_float, default=0.4,
                            help='mixup alpha parameter for beta dist.')
        return parser
