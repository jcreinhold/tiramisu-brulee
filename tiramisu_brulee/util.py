"""Miscellaneous functions
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: Jul 01, 2020
"""

__all__ = ["InitType", "init_weights"]

import builtins
import enum

import torch
import torch.nn as nn


def is_conv(layer: nn.Module) -> builtins.bool:
    classname = layer.__class__.__name__
    return hasattr(layer, "weight") and "Conv" in classname


def is_norm(layer: nn.Module) -> builtins.bool:
    classname = layer.__class__.__name__
    return hasattr(layer, "weight") and "Norm" in classname


@enum.unique
class InitType(enum.Enum):
    NORMAL = "normal"
    XAVIER_NORMAL = "xavier_normal"
    HE_NORMAL = "he_normal"
    HE_UNIFORM = "he_uniform"
    ORTHOGONAL = "orthogonal"

    @classmethod
    def from_string(cls, string: builtins.str) -> "InitType":
        if string.lower() == "normal":
            return cls.NORMAL
        elif string.lower() == "xavier_normal":
            return cls.XAVIER_NORMAL
        elif string.lower() == "he_normal":
            return cls.HE_NORMAL
        elif string.lower() == "he_uniform":
            return cls.HE_UNIFORM
        elif string.lower() == "orthogonal":
            return cls.ORTHOGONAL
        else:
            raise ValueError("Invalid init type.")


def init_weights(
    net: nn.Module,
    *,
    init_type: InitType = InitType.NORMAL,
    gain: builtins.float = 0.02,
) -> None:
    def init_func(layer: nn.Module) -> None:
        _is_conv = is_conv(layer)
        _is_norm = is_norm(layer)
        if not _is_conv and not _is_norm:
            return
        assert isinstance(layer.weight, torch.Tensor)
        weight = layer.weight
        assert weight is layer.weight
        has_bias = hasattr(layer, "bias") and layer.bias is not None
        if has_bias:
            assert isinstance(layer.bias, torch.Tensor)
            bias = layer.bias
            assert bias is layer.bias
        if _is_conv:
            if init_type == InitType.NORMAL:
                nn.init.normal_(weight, 0.0, gain)
            elif init_type == InitType.XAVIER_NORMAL:
                nn.init.xavier_normal_(weight, gain=gain)
            elif init_type == InitType.HE_NORMAL:
                nn.init.kaiming_normal_(weight, a=0.0, mode="fan_in")
            elif init_type == InitType.HE_UNIFORM:
                nn.init.kaiming_uniform_(weight, a=0.0, mode="fan_in")
            elif init_type == InitType.ORTHOGONAL:
                nn.init.orthogonal_(weight, gain=gain)
            else:
                err_msg = f"initialization type [{init_type}] not implemented"
                raise NotImplementedError(err_msg)
            if has_bias:
                # noinspection PyUnboundLocalVariable
                nn.init.constant_(bias, 0.0)
        elif _is_norm:
            nn.init.normal_(weight, 1.0, gain)
            if has_bias:
                # noinspection PyUnboundLocalVariable
                nn.init.constant_(bias, 0.0)

    net.apply(init_func)
