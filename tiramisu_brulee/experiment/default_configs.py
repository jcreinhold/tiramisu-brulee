#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.default_configs

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 03, 2020
"""

__all__ = ['default_lightningtiramisu2d_config',
           'default_lightningtiramisu3d_config']

from pytorch_lightning.utilities.parsing import AttributeDict as adict

default_lightningtiramisu2d_config = adict(
    data_params=adict(
        batch_size=16,
        num_workers=8,
        patch_size=(3, 128, 128),
        queue_length=100,
        samples_per_volume=10
    ),
    lightning_params=adict(
        init_params=adict(
            init_type='normal',
            gain=0.02
        ),
        n_epochs=1,
        network_dim=2,
    ),
    network_params=adict(
        in_channels=3,
        out_channels=1,
        down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5),
        bottleneck_layers=5,
        growth_rate=16,
        out_chans_first_conv=48,
        dropout_rate=0.2,
    ),
    optim_params=adict(
        lr=1e3,
        betas=(0.9, 0.99),
        weight_decay=1e-7,
    )
)

default_lightningtiramisu3d_config = adict(
    data_params=adict(
        batch_size=8,
        num_workers=8,
        patch_size=64,
        queue_length=100,
        samples_per_volume=10
    ),
    lightning_params=adict(
        init_params=adict(
            init_type='normal',
            gain=0.02
        ),
        n_epochs=1,
        network_dim=3,
    ),
    network_params=adict(
        in_channels=1,
        out_channels=1,
        down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4),
        bottleneck_layers=4,
        growth_rate=16,
        out_chans_first_conv=48,
        dropout_rate=0.2
    ),
    optim_params=adict(
        lr=1e3,
        betas=(0.9, 0.99),
        weight_decay=1e-7,
    )
)
