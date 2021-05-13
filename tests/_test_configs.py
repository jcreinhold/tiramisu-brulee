#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.tests._test_configs

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 03, 2020
"""

__all__ = ['test_lightningtiramisu2d_config',
           'test_lightningtiramisu3d_config']

from pytorch_lightning.utilities.parsing import AttributeDict as adict

test_lightningtiramisu2d_config = adict(
    data_params=adict(
        batch_size=4,
        num_workers=0,
        patch_size=(3, 32, 32),
        queue_length=4,
        samples_per_volume=4
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
        down_blocks=(2, 2),
        up_blocks=(2, 2),
        bottleneck_layers=2,
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

test_lightningtiramisu3d_config = adict(
    data_params=adict(
        batch_size=4,
        num_workers=0,
        patch_size=(16, 16, 16),
        queue_length=4,
        samples_per_volume=4
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
        down_blocks=(2, 2),
        up_blocks=(2, 2),
        bottleneck_layers=2,
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
