#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.util

miscellaneous functions

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: Jul 01, 2020
"""

__all__ = [
    "init_weights",
]

from torch.nn import init, Module


def _is_conv(layer: Module) -> bool:
    classname = layer.__class__.__name__
    return hasattr(layer, "weight") and "Conv" in classname


def _is_norm(layer: Module) -> bool:
    classname = layer.__class__.__name__
    return hasattr(layer, "weight") and "Norm" in classname


def init_weights(net: Module, init_type: str = "normal", gain: float = 0.02):
    def init_func(layer):
        if _is_conv(layer):
            if init_type == "normal":
                init.normal_(layer.weight.data, 0.0, gain)
            elif init_type == "xavier_normal":
                init.xavier_normal_(layer.weight.data, gain=gain)
            elif init_type == "he_normal":
                init.kaiming_normal_(layer.weight.data, a=0.0, mode="fan_in")
            elif init_type == "he_uniform":
                init.kaiming_uniform_(layer.weight.data, a=0.0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(layer.weight.data, gain=gain)
            else:
                err_msg = f"initialization type [{init_type}] not implemented"
                raise NotImplementedError(err_msg)
            if hasattr(layer, "bias") and layer.bias is not None:
                init.constant_(layer.bias.data, 0.0)
        elif _is_norm(layer):
            init.normal_(layer.weight.data, 1.0, gain)
            init.constant_(layer.bias.data, 0.0)

    net.apply(init_func)
