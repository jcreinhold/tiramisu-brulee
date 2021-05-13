#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lightningtiramisu

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 03, 2020
"""

__all__ = ['LightningTiramisu']

from torch import Tensor

import pytorch_lightning as pl

from pytorch_lightning.utilities.parsing import AttributeDict
from tiramisu_brulee.model import Tiramisu2d, Tiramisu3d
from tiramisu_brulee.util import init_weights


class LightningTiramisu(pl.LightningModule):

    def __init__(self, hparams: AttributeDict):
        super().__init__()
        hparams = self._hparams_to_attributedict(hparams)
        self.network_dim = hparams.lightning_params["network_dim"]
        if self._use_2d_network:
            self.net = Tiramisu2d(**hparams.network_params)
        elif self._use_3d_network:
            self.net = Tiramisu3d(**hparams.network_params)
        else:
            raise self._invalid_network_dim
        init_weights(self.net, **hparams.lightning_params["init_params"])
        self.save_hyperparameters(hparams)

    @staticmethod
    def _hparams_to_attributedict(hparams):
        if not isinstance(hparams, AttributeDict):
            return AttributeDict(hparams)
        return hparams

    @property
    def _use_2d_network(self):
        return self.network_dim == 2

    @property
    def _use_3d_network(self):
        return self.network_dim == 3

    @property
    def _invalid_network_dim(self):
        err_msg = f"Network dim. {self.network_dim} invalid."
        return ValueError(err_msg)

    @staticmethod
    def criterion(x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
