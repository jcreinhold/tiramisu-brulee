"""Miscellaneous functions
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: Jul 01, 2020
"""

__all__ = ["init_weights"]

import builtins

import torch
import torch.nn as nn


def _is_conv(layer: nn.Module) -> builtins.bool:
    classname = layer.__class__.__name__
    return hasattr(layer, "weight") and "Conv" in classname


def _is_norm(layer: nn.Module) -> builtins.bool:
    classname = layer.__class__.__name__
    return hasattr(layer, "weight") and "Norm" in classname


def init_weights(
    net: nn.Module, *, init_type: builtins.str = "normal", gain: builtins.float = 0.02
) -> None:
    def init_func(layer: nn.Module) -> None:
        is_conv = _is_conv(layer)
        is_norm = _is_norm(layer)
        if not is_conv and not is_norm:
            return
        assert isinstance(layer.weight, torch.Tensor)
        weight = layer.weight
        assert weight is layer.weight
        has_bias = hasattr(layer, "bias") and layer.bias is not None
        if has_bias:
            assert isinstance(layer.bias, torch.Tensor)
            bias = layer.bias
            assert bias is layer.bias
        if is_conv:
            if init_type == "normal":
                nn.init.normal_(weight, 0.0, gain)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(weight, gain=gain)
            elif init_type == "he_normal":
                nn.init.kaiming_normal_(weight, a=0.0, mode="fan_in")
            elif init_type == "he_uniform":
                nn.init.kaiming_uniform_(weight, a=0.0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(weight, gain=gain)
            else:
                err_msg = f"initialization type [{init_type}] not implemented"
                raise NotImplementedError(err_msg)
            if has_bias:
                # noinspection PyUnboundLocalVariable
                nn.init.constant_(bias, 0.0)
        elif is_norm:
            nn.init.normal_(weight, 1.0, gain)
            if has_bias:
                # noinspection PyUnboundLocalVariable
                nn.init.constant_(bias, 0.0)

    net.apply(init_func)
