#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lightningtiramisu

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jul 03, 2020
"""

__all__ = ["LightningTiramisu"]

from typing import List, Tuple

import pytorch_lightning as pl
from torch import Tensor

from tiramisu_brulee.model import Tiramisu2d, Tiramisu3d
from tiramisu_brulee.util import init_weights


class LightningTiramisu(pl.LightningModule):
    def __init__(
        self,
        network_dim: int,
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
        learning_rate: float = 3e-4,
        betas: Tuple[int, int] = (0.9, 0.99),
        weight_decay: float = 1e-7,
    ):
        super().__init__()
        self.save_hyperparameters()
        if self._use_2d_network:
            net = Tiramisu2d
        elif self._use_3d_network:
            net = Tiramisu3d
        else:
            raise self._invalid_network_dim
        self.net = net(
            self.hparams.in_channels,
            self.hparams.out_channels,
            self.hparams.down_blocks,
            self.hparams.up_blocks,
            self.hparams.bottleneck_layers,
            self.hparams.growth_rate,
            self.hparams.first_conv_out_channels,
            self.hparams.dropout_rate,
        )
        init_weights(self.net, self.hparams.init_type, self.hparams.gain)

    @property
    def _use_2d_network(self):
        return self.hparams.network_dim == 2

    @property
    def _use_3d_network(self):
        return self.hparams.network_dim == 3

    @property
    def _invalid_network_dim(self):
        err_msg = f"Network dim. {self.hparams.network_dim} invalid."
        return ValueError(err_msg)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
