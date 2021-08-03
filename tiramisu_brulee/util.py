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

from torch import Tensor
from torch.nn import init, Module


def _is_conv(layer: Module) -> bool:
    classname = layer.__class__.__name__
    return hasattr(layer, "weight") and "Conv" in classname


def _is_norm(layer: Module) -> bool:
    classname = layer.__class__.__name__
    return hasattr(layer, "weight") and "Norm" in classname


def init_weights(net: Module, init_type: str = "normal", gain: float = 0.02) -> None:
    def init_func(layer: Module) -> None:
        assert isinstance(layer.weight.data, Tensor)
        weight_data = layer.weight.data
        assert weight_data is layer.weight.data
        has_bias = hasattr(layer, "bias") and layer.bias is not None
        if has_bias:
            assert isinstance(layer.bias.data, Tensor)
            bias_data = layer.bias.data
            assert bias_data is layer.bias.data
        if _is_conv(layer):
            if init_type == "normal":
                init.normal_(weight_data, 0.0, gain)
            elif init_type == "xavier_normal":
                init.xavier_normal_(weight_data, gain=gain)
            elif init_type == "he_normal":
                init.kaiming_normal_(weight_data, a=0.0, mode="fan_in")
            elif init_type == "he_uniform":
                init.kaiming_uniform_(weight_data, a=0.0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(weight_data, gain=gain)
            else:
                err_msg = f"initialization type [{init_type}] not implemented"
                raise NotImplementedError(err_msg)
            if has_bias:
                init.constant_(bias_data, 0.0)
        elif _is_norm(layer):
            init.normal_(weight_data, 1.0, gain)
            if has_bias:
                init.constant_(bias_data, 0.0)

    net.apply(init_func)
