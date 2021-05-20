#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.tests.test_tiramisu3d

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 13, 2021
"""

from os.path import join
import warnings

import pytest
from pytorch_lightning import Trainer, seed_everything
import torchio

import tiramisu_brulee
from tiramisu_brulee.experiment.lesion_seg.data import csv_to_subjectlist
from tiramisu_brulee.loss import binary_combo_loss

from ._test_configs import test_lightningtiramisu2d_config
from ._test_lightningtiramisu import (
    create_test_csv,
    LightningTiramisuTester,
    n_dirname,
)

seed_everything(1337)

tiramisu_brulee_dir = n_dirname(tiramisu_brulee.__file__, 2)
DATA_DIR = join(tiramisu_brulee_dir, "tests/test_data/")


class LightningTiramisu2d(LightningTiramisuTester):
    def __init__(self, hparams, subject_list):
        super().__init__(2, hparams, subject_list)

    def training_step(self, batch, batch_idx):
        x = batch["t1"][torchio.DATA].squeeze()
        y = batch["label"][torchio.DATA].squeeze()[:, 1:2, ...]
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x = batch["t1"][torchio.DATA].squeeze()
        y = batch["label"][torchio.DATA].squeeze()[:, 1:2, ...]
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return {"val_loss": loss}


@pytest.fixture(scope="session")
def temp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("out")


@pytest.fixture
def subject_list(temp_dir):
    csv = join(temp_dir, "data.csv")
    create_test_csv(csv, DATA_DIR)
    subject_list = csv_to_subjectlist(csv)
    return subject_list


@pytest.fixture
def net(subject_list):
    net = LightningTiramisu2d(test_lightningtiramisu2d_config, subject_list)
    return net


def test_tiramisu2d(net, temp_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer = Trainer(
            default_root_dir=temp_dir, fast_dev_run=True, progress_bar_refresh_rate=0
        )
        trainer.fit(net)


def test_weight(net, temp_dir):
    csv = join(temp_dir, "data.csv")
    create_test_csv(csv, DATA_DIR, weight=True)
    subject_list = csv_to_subjectlist(csv)
    net = LightningTiramisu2d(test_lightningtiramisu2d_config, subject_list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer = Trainer(
            default_root_dir=temp_dir, fast_dev_run=True, progress_bar_refresh_rate=0
        )
        trainer.fit(net)


def test_combo_loss(net, temp_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.criterion = binary_combo_loss
        trainer = Trainer(
            default_root_dir=temp_dir, fast_dev_run=True, progress_bar_refresh_rate=0
        )
        trainer.fit(net)
