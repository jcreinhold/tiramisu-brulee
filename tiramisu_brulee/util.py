#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.util

miscellaneous functions

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 01, 2020
"""

__all__ = ['init_weights',
           'n_dirname']

import os

from torch.nn import init


def _is_conv(layer):
    classname = layer.__class__.__name__
    return (hasattr(layer, 'weight') and 'Conv' in classname)


def _is_norm(layer):
    classname = layer.__class__.__name__
    return (hasattr(layer, 'weight') and 'Norm' in classname)


def init_weights(net, init_type: str = 'normal', gain: float = 0.02):
    def init_func(layer):
        if _is_conv(layer):
            if init_type == 'normal':
                init.normal_(layer.weight.data, 0.0, gain)
            elif init_type == 'xavier_normal':
                init.xavier_normal_(layer.weight.data, gain=gain)
            elif init_type == 'he_normal':
                init.kaiming_normal_(layer.weight.data, a=0., mode='fan_in')
            elif init_type == 'he_uniform':
                init.kaiming_uniform_(layer.weight.data, a=0., mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(layer.weight.data, gain=gain)
            else:
                err_msg = f'initialization type [{init_type}] is not implemented'
                raise NotImplementedError(err_msg)
            if hasattr(layer, 'bias') and layer.bias is not None:
                init.constant_(layer.bias.data, 0.0)
        elif _is_norm(layer):
            init.normal_(layer.weight.data, 1.0, gain)
            init.constant_(layer.bias.data, 0.0)

    net.apply(init_func)


def n_dirname(path: str, n: int) -> str:
    """ return n-th dirname from basename """
    dirname = path
    for _ in range(n):
        dirname = os.path.dirname(dirname)
    return dirname


if __name__ == "__main__":
    from torch import nn

    net = nn.Sequential(nn.Conv1d(1, 1, 1),
                        nn.BatchNorm1d(1))
    init_weights(net)
