#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.tests.tiramisu3d

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: May 13, 2021
"""

import contextlib
import os
from os.path import join
import warnings

import pytest
from pytorch_lightning import Trainer, seed_everything

with open(os.devnull, "w") as f:
    with contextlib.redirect_stdout(f):
        import torchio

import tiramisu_brulee
from tiramisu_brulee.loss import binary_combo_loss
from tiramisu_brulee.data import csv_to_subjectlist
from tiramisu_brulee.util import n_dirname

from ._test_configs import test_lightningtiramisu2d_config
from ._test_lightningtiramisu import (
    _create_test_csv,
    LightningTiramisuTester
)

seed_everything(1337)

tiramisu_brulee_dir = n_dirname(tiramisu_brulee.__file__, 2)
DATA_DIR = join(tiramisu_brulee_dir, "tests/test_data/")


class LightningTiramisu3d(LightningTiramisuTester):

    def training_step(self, batch, batch_idx):
        x = batch['t1'][torchio.DATA].squeeze()
        y = batch['label'][torchio.DATA].squeeze()[:, 1:2, ...]
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x = batch['t1'][torchio.DATA].squeeze()
        y = batch['label'][torchio.DATA].squeeze()[:, 1:2, ...]
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return {'val_loss': loss}


@pytest.fixture(scope='session')
def temp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('out')


@pytest.fixture
def subject_list(temp_dir):
    csv = join(temp_dir, "data.csv")
    _create_test_csv(csv, DATA_DIR)
    subject_list = csv_to_subjectlist(csv)
    return subject_list


@pytest.fixture
def net(subject_list):
    net = LightningTiramisu3d(
        test_lightningtiramisu2d_config,
        subject_list)
    return net


def test_tiramisu3d(net, temp_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer = Trainer(
            default_root_dir=temp_dir,
            fast_dev_run=True,
            progress_bar_refresh_rate=0)
        trainer.fit(net)


def test_weight(net, temp_dir):
    csv = join(temp_dir, "data.csv")
    _create_test_csv(csv, DATA_DIR, weight=True)
    subject_list = csv_to_subjectlist(csv)
    net = LightningTiramisu3d(
        test_lightningtiramisu2d_config,
        subject_list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer = Trainer(
            default_root_dir=temp_dir,
            fast_dev_run=True,
            progress_bar_refresh_rate=0)
        trainer.fit(net)


def test_combo_loss(net, temp_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.criterion = binary_combo_loss
        trainer = Trainer(
            default_root_dir=temp_dir,
            fast_dev_run=True,
            progress_bar_refresh_rate=0)
        trainer.fit(net)
