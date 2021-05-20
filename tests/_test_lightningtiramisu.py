#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.tests._test_lightningtiramisu

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jul 03, 2020
"""

__all__ = [
    "create_test_csv",
    "LightningTiramisuTester",
    "n_dirname",
]

import os
from os.path import join
from typing import List

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from pytorch_lightning.utilities.parsing import AttributeDict

import torchio
from torchio.transforms import Compose, OneOf, RandomAffine, RandomElasticDeformation

from tiramisu_brulee.experiment.lightningtiramisu import LightningTiramisu


class LightningTiramisuTester(LightningTiramisu):
    def __init__(
        self,
        network_dim: int,
        hparams: AttributeDict,
        subject_list: List[torchio.Subject],
    ):
        super().__init__(network_dim, **hparams["network_params"])
        self.criterion = F.binary_cross_entropy_with_logits
        self.subject_list = subject_list
        self.my_hparams = hparams

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return AdamW(self.parameters(), **self.my_hparams.optim_params)

    def train_dataloader(self):
        spatial = OneOf({RandomAffine(): 0.8, RandomElasticDeformation(): 0.2}, p=0.75,)
        transforms = [spatial]
        transform = Compose(transforms)

        subjects_dataset = torchio.SubjectsDataset(
            self.subject_list, transform=transform
        )

        sampler = torchio.data.UniformSampler(self.my_hparams.data_params["patch_size"])
        patches_queue = torchio.Queue(
            subjects_dataset,
            self.my_hparams.data_params["queue_length"],
            self.my_hparams.data_params["samples_per_volume"],
            sampler,
            num_workers=self.my_hparams.data_params["num_workers"],
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        train_dataloader = DataLoader(
            patches_queue, batch_size=self.my_hparams.data_params["batch_size"]
        )
        return train_dataloader

    def val_dataloader(self):
        subjects_dataset = torchio.SubjectsDataset(self.subject_list)
        sampler = torchio.data.UniformSampler(self.my_hparams.data_params["patch_size"])
        patches_queue = torchio.Queue(
            subjects_dataset,
            self.my_hparams.data_params["queue_length"],
            self.my_hparams.data_params["samples_per_volume"],
            sampler,
            num_workers=self.my_hparams.data_params["num_workers"],
            shuffle_subjects=False,
            shuffle_patches=False,
        )
        val_dataloader = DataLoader(
            patches_queue, batch_size=self.my_hparams.data_params["batch_size"]
        )
        return val_dataloader


def create_test_csv(path_to_csv: str, data_dir: str, weight: bool = False):
    t1 = join(data_dir, "img.nii.gz")
    label = join(data_dir, "mask.nii.gz")
    headers = "subject,label,t1"
    filenames = f"subj1,{t1},{label}"
    if weight:
        headers += ",weight\n"
        filenames += ",0.9"
    else:
        headers += "\n"
    with open(path_to_csv, "w") as f:
        f.write(headers)
        f.write(filenames)


def n_dirname(path: str, n: int) -> str:
    """ return n-th dirname from basename """
    dirname = path
    for _ in range(n):
        dirname = os.path.dirname(dirname)
    return dirname
