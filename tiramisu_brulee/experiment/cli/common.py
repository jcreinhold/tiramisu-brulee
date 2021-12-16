"""Common functions for predict and train CLIs
Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: Jul 30, 2021
"""

__all__ = [
    "check_patch_size",
    "EXPERIMENT_NAME",
    "handle_fast_dev_run",
    "pseudo3d_dims_setup",
    "tiramisu_brulee_info",
]

import builtins
import pathlib
import subprocess  # nosec
import sys
import typing

from tiramisu_brulee.experiment.type import TiramisuBruleeInfo

EXPERIMENT_NAME = "lesion_tiramisu_experiment"


def check_patch_size(
    patch_size: typing.List[builtins.int], use_pseudo3d: builtins.bool
) -> None:
    n_patch_elems = len(patch_size)
    if n_patch_elems != 2 and use_pseudo3d:
        raise ValueError(
            f"Number of patch size elements must be 2 for "
            f"pseudo-3D (2D) network. Got {len(patch_size)}."
        )
    elif n_patch_elems != 3 and not use_pseudo3d:
        raise ValueError(
            f"Number of patch size elements must be 3 for "
            f"a 3D network. Got {len(patch_size)}."
        )


def handle_fast_dev_run(
    unnecessary_args: typing.Set[builtins.str],
) -> typing.Set[builtins.str]:
    """fast_dev_run is problematic with py36 so remove it"""
    py_version = sys.version_info
    assert py_version.major == 3
    if py_version.minor == 6:
        unnecessary_args.add("fast_dev_run")
    return unnecessary_args


def pseudo3d_dims_setup(
    pseudo3d_dim: typing.Optional[typing.List[builtins.int]],
    n_models: builtins.int,
    stage: builtins.str,
) -> typing.Union[typing.List[None], typing.List[builtins.int]]:
    assert stage in ("train", "predict")
    if stage == "predict":
        stage = "us"
    n_p3d = 0 if pseudo3d_dim is None else len(pseudo3d_dim)
    pseudo3d_dims: typing.Union[typing.List[None], typing.List[builtins.int]]
    if n_p3d == 1 and pseudo3d_dim is not None:
        pseudo3d_dims = pseudo3d_dim * n_models
    elif n_p3d == n_models and pseudo3d_dim is not None:
        pseudo3d_dims = pseudo3d_dim
    elif pseudo3d_dim is None:
        pseudo3d_dims = [None] * n_models
    else:
        raise ValueError(
            "pseudo3d_dim must be None (for 3D network), 1 value to be used "
            f"across all models to be {stage}ed, or N values corresponding to each "
            f"of the N models to be {stage}ed. Got {n_p3d} != {n_models}."
        )
    return pseudo3d_dims


def tiramisu_brulee_info(*, short: builtins.bool = True) -> TiramisuBruleeInfo:
    """get the git commit hash and version for tiramisu-brulee"""
    import tiramisu_brulee

    tb_path = str(pathlib.Path(tiramisu_brulee.__file__).parents[1])
    cmd = ["git", "rev-parse", "HEAD"]
    if short:
        cmd.insert(2, "--short")
    out = subprocess.run(  # nosec
        cmd, cwd=tb_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if out.returncode == 0:
        commit = out.stdout.decode("ascii").strip()
    else:
        commit = "unknown"
    return TiramisuBruleeInfo(version=tiramisu_brulee.__version__, commit=commit)
