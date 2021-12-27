#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.cli.to_onnx

command-line interface functions for converting
a trained Tiramisu neural network to ONNX

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 27 Dec 2021
"""

__all__ = [
    "to_onnx",
]

import argparse
import builtins
import logging

import torch

from tiramisu_brulee.experiment.cli.common import check_patch_size, pseudo3d_dims_setup
from tiramisu_brulee.experiment.parse import parse_unknown_to_dict
from tiramisu_brulee.experiment.seg import LesionSegLightningTiramisu
from tiramisu_brulee.experiment.type import (
    ArgParser,
    ArgType,
    ModelNum,
    file_path,
    nonnegative_int_or_none_or_all,
    positive_odd_int_or_none,
)
from tiramisu_brulee.experiment.util import setup_log, split_filename


def arg_parser() -> ArgParser:
    """argument parser for using a Tiramisu CNN for prediction"""
    desc = "Convert a Tiramisu CNN to the ONNX format"
    parser = argparse.ArgumentParser(prog="tiramisu-to-onnx", description=desc)
    parser.add_argument(
        "-mp",
        "--model-path",
        type=file_path(),
        nargs="+",
        required=True,
        default=["SET ME!"],
        help="path to the trained model",
    )
    parser.add_argument(
        "-op",
        "--onnx-path",
        type=str,
        required=True,
        default="SET ME!",
        help="path to output the onnx model",
    )
    parser.add_argument(
        "-p3d",
        "--pseudo3d-dim",
        type=nonnegative_int_or_none_or_all(),
        nargs="+",
        choices=(0, 1, 2, "all"),
        default=None,
        help="dim on which to concatenate the images for input "
        "to a 2D network. If provided, either provide 1 value"
        "to be used for each CSV or provide N values "
        "corresponding to the N CSVs. If not provided, "
        "use 3D network.",
    )
    parser.add_argument(
        "-p3s",
        "--pseudo3d-size",
        type=positive_odd_int_or_none(),
        default=None,
        help="size of the pseudo3d dimension (if -p3d provided)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="opset version for ONNX (see torch.onnx.export for details)",
    )
    parser.add_argument(
        "--do-constant-folding",
        action="store_true",
        help="do constant folding for ONNX (see torch.onnx.export for details)",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="increase output verbosity (e.g., -vv is more than -v)",
    )
    return parser


def to_onnx(args: ArgType = None) -> builtins.int:
    """Convert a Tiramisu CNN to ONNX for inference"""
    parser = arg_parser()
    if args is None:
        args, unknown = parser.parse_known_args()
    elif isinstance(args, list):
        args, unknown = parser.parse_known_args()
    else:
        raise RuntimeError(f"Expected args to be None or a list. Got {type(args)}.")
    modalities = list(sorted(parse_unknown_to_dict(unknown, names_only=True).keys()))
    n_inputs = len(modalities)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    n_models = len(args.model_path)
    pseudo3d_dims = pseudo3d_dims_setup(args.pseudo3d_dim, n_models, "predict")
    predict_iter_data = zip(args.model_path, pseudo3d_dims)
    for i, (model_path, p3d) in enumerate(predict_iter_data, 1):
        model_num = ModelNum(num=i, out_of=n_models)
        if n_models == 1:
            onnx_path = args.onnx_path
        else:
            root, base, ext = split_filename(args.onnx_path)
            onnx_path = root / (base + f"_{i}" + ext)
        nth_model = f" ({i}/{n_models})"
        model = LesionSegLightningTiramisu.load_from_checkpoint(
            str(model_path),
            _model_num=model_num,
        )
        n_channels = n_inputs if p3d is None else (args.pseudo3d_size * n_inputs)
        input_shape = (1, n_channels) + (128,) * (3 if p3d is None else 2)
        input_sample = torch.randn(input_shape)
        axes = {0: "batch_size", 2: "h", 3: "w"}
        if p3d is None:
            axes.update({4: "d"})
        model.to_onnx(
            file_path=onnx_path,
            input_sample=input_sample,
            export_params=True,
            verbose=args.verbosity >= 2,
            opset_version=args.opset_version,
            do_constant_folding=args.do_constant_folding,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": axes, "output": axes},
        )
        logger.info("Finished converting" + nth_model)
    return 0
