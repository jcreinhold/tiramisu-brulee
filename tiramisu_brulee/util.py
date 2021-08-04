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
        is_conv = _is_conv(layer)
        is_norm = _is_norm(layer)
        if not is_conv and not is_norm:
            return
        assert isinstance(layer.weight, Tensor)
        weight = layer.weight
        assert weight is layer.weight
        has_bias = hasattr(layer, "bias") and layer.bias is not None
        if has_bias:
            assert isinstance(layer.bias, Tensor)
            bias = layer.bias
            assert bias is layer.bias
        if is_conv:
            if init_type == "normal":
                init.normal_(weight, 0.0, gain)
            elif init_type == "xavier_normal":
                init.xavier_normal_(weight, gain=gain)
            elif init_type == "he_normal":
                init.kaiming_normal_(weight, a=0.0, mode="fan_in")
            elif init_type == "he_uniform":
                init.kaiming_uniform_(weight, a=0.0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(weight, gain=gain)
            else:
                err_msg = f"initialization type [{init_type}] not implemented"
                raise NotImplementedError(err_msg)
            if has_bias:
                # noinspection PyUnboundLocalVariable
                init.constant_(bias, 0.0)
        elif is_norm:
            init.normal_(weight, 1.0, gain)
            if has_bias:
                # noinspection PyUnboundLocalVariable
                init.constant_(bias, 0.0)

    net.apply(init_func)
