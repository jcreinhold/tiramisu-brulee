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
import contextlib
import io
import logging
import os
import sys
import tempfile
import typing
import warnings

import onnxruntime as ort
import onnxruntime.quantization as ortq
import torch
import torch.nn.utils.prune as prune

from tiramisu_brulee.experiment.cli.common import pseudo3d_dims_setup
from tiramisu_brulee.experiment.parse import parse_unknown_to_dict
from tiramisu_brulee.experiment.seg import LesionSegLightningTiramisu
from tiramisu_brulee.experiment.type import (
    ArgParser,
    ArgType,
    ModelNum,
    PathLike,
    file_path,
    nonnegative_int_or_none_or_all,
    positive_odd_int_or_none,
    probability_float,
)
from tiramisu_brulee.experiment.util import setup_log, split_filename
from tiramisu_brulee.util import is_conv


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
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--quantize", action="store_true", help="dynamic quantization")
    quantization.add_argument("--float16", action="store_true", help="use float16")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="apply global pruning before saving the model",
    )
    parser.add_argument(
        "--prune-amount",
        type=probability_float(),
        default=0.25,
        help="saving the model",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="simplify onnx model with onnx-simplifier",
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
        if args.prune:
            parameters_to_prune: typing.List[typing.Tuple[torch.nn.Module, str]] = []
            for module in model.modules():
                if is_conv(module):
                    parameters_to_prune.append((module, "weight"))
            n_params = len(parameters_to_prune)
            msg = f"Pruning {n_params} convolutional layers with {args.prune_amount}"
            logger.info(msg + nth_model)
            prune.global_unstructured(
                parameters_to_prune, prune.L1Unstructured, amount=args.prune_amount
            )
        n_channels = n_inputs if p3d is None else (args.pseudo3d_size * n_inputs)
        input_shape = (1, n_channels) + (128,) * (3 if p3d is None else 2)
        input_sample = torch.randn(input_shape)
        axes = {0: "batch_size", 2: "h", 3: "w"}
        if p3d is None:
            axes.update({4: "d"})
        with tempfile.NamedTemporaryFile("w") as f:
            save_as_ort = str(onnx_path).endswith(".ort")
            file_path = f.name if save_as_ort else onnx_path
            to_onnx_kwds = dict(
                file_path=file_path,
                input_sample=input_sample,
                export_params=True,
                verbose=args.verbosity >= 2,
                opset_version=args.opset_version,
                do_constant_folding=args.do_constant_folding,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": axes, "output": axes},
            )
            logger.info("Exporting model to ONNX" + nth_model)
            if args.verbosity >= 3:
                model.to_onnx(**to_onnx_kwds)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with io.BytesIO() as buffer:
                        with stderr_redirected(buffer):
                            model.to_onnx(**to_onnx_kwds)
            if args.quantize:
                logger.info("Quantizing model (uint8)" + nth_model)
                ortq.quantize_dynamic(
                    file_path,
                    file_path,
                    activation_type=ortq.QuantType.QUInt8,
                    weight_type=ortq.QuantType.QUInt8,
                )
            elif args.float16:
                logger.info("Converting model to half-precision (float16)" + nth_model)
                to_float16(file_path)
            if args.simplify:
                logger.info("Simplifying model" + nth_model)
                simplify(file_path, input_shape)
            if save_as_ort:
                logger.info("Saving as ORT" + nth_model)
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                )
                sess_options.optimized_model_filepath = onnx_path
                ort.InferenceSession(file_path, sess_options=sess_options)
        logger.info("Finished converting" + nth_model)
    return 0


def simplify(
    onnx_model_path: PathLike, input_shape: typing.Tuple[builtins.int, ...]
) -> None:
    try:
        import onnx
        import onnxsim
    except (ImportError, ModuleNotFoundError):
        warnings.warn("Cannot import onnx or onnx-simplifier.")
        return
    model = onnx.load_model(onnx_model_path)
    model_simp, check = onnxsim.simplify(
        model, input_shapes={None: input_shape}, dynamic_input_shape=True
    )
    if not check:
        warnings.warn("Simplified ONNX model could not be validated.")
        return
    onnx.save_model(model_simp, onnx_model_path)


def to_float16(onnx_model_path: PathLike) -> None:
    try:
        from onnxmltools.utils import save_model
        from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
    except (ImportError, ModuleNotFoundError):
        warnings.warn("Cannot import onnxmltools.")
        return
    model = convert_float_to_float16_model_path(onnx_model_path)
    save_model(model, onnx_model_path)


# https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
# https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/
@contextlib.contextmanager
def stderr_redirected(stream):  # type: ignore[no-untyped-def]
    # The original fd stderr points to. Usually 2 on POSIX systems.
    original_stderr_fd = sys.stderr.fileno()
    # Create a temporary file and redirect stderr to it
    with tempfile.TemporaryFile(mode="w+b") as tf:
        with os.fdopen(os.dup(original_stderr_fd), "wb") as saved_stderr_fd:
            try:
                sys.stderr.flush()
                # Make original_stderr_fd point to the temporary file
                os.dup2(tf.fileno(), original_stderr_fd)
                # Yield to caller
                yield
                # Copy contents of temporary file to the given stream
                tf.flush()
                tf.seek(0, io.SEEK_SET)
                stream.write(tf.read())
            finally:
                sys.stderr.flush()
                # Restore stderr to it's original value
                os.dup2(saved_stderr_fd.fileno(), original_stderr_fd)
