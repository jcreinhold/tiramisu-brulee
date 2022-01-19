#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.tests.test_lesion_seg

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 18, 2021
"""

import builtins
import pathlib
import sys
import typing

import pytest

from tiramisu_brulee.experiment.cli.predict import predict, predict_image
from tiramisu_brulee.experiment.cli.train import train


@pytest.fixture
def file() -> pathlib.Path:
    return pathlib.Path(__file__).resolve()


@pytest.fixture
def cwd(file: pathlib.Path) -> pathlib.Path:
    return file.parent


@pytest.fixture
def data_dir(cwd: pathlib.Path) -> pathlib.Path:
    return cwd / "test_data"


@pytest.fixture(scope="session")
def temp_dir(tmpdir_factory) -> pathlib.Path:  # type: ignore[no-untyped-def]
    return pathlib.Path(tmpdir_factory.mktemp("out"))


def _create_csv(
    temp_dir: pathlib.Path, data_dir: pathlib.Path, stage: builtins.str
) -> pathlib.Path:
    csv_path = temp_dir / f"{stage}.csv"
    image_path = data_dir / "img.nii.gz"
    label_path = data_dir / "mask.nii.gz"
    out_path = temp_dir / "out.nii.gz"
    headers = "subject,label,t1,t2"
    filenames = [f"subj{i},{label_path},{image_path},{image_path}" for i in range(2)]
    if stage == "predict":
        headers += ",out\n"
        filenames = [fns + f",{out_path}\n" for fns in filenames]
    else:
        headers += "\n"
        filenames = [fns + "\n" for fns in filenames]
    with open(csv_path, "w") as f:
        f.write(headers)
        for fns in filenames:
            f.write(fns)
    return csv_path


@pytest.fixture
def train_csv(temp_dir: pathlib.Path, data_dir: pathlib.Path) -> pathlib.Path:
    return _create_csv(temp_dir, data_dir, "train")


@pytest.fixture
def predict_csv(temp_dir: pathlib.Path, data_dir: pathlib.Path) -> pathlib.Path:
    return _create_csv(temp_dir, data_dir, "predict")


@pytest.fixture
def cli_train_args(temp_dir: pathlib.Path) -> typing.List[builtins.str]:
    args = []
    args += f"--default_root_dir {temp_dir}".split()
    args += "--enable_progress_bar false".split()
    args += "--num-input 2".split()
    args += "--batch-size 2".split()
    args += "--queue-length 1".split()
    args += "--samples-per-volume 1".split()
    args += "--n-epochs 2".split()
    args += "--down-blocks 2 2".split()
    args += "--up-blocks 2 2".split()
    args += "--bottleneck-layers 2".split()
    args += "--first-conv-out-channels 2".split()
    args += "--num-workers 0".split()
    args += "--pos-weight 2.0".split()
    return args


@pytest.fixture
def cli_predict_args(
    temp_dir: pathlib.Path, predict_csv: pathlib.Path
) -> typing.List[builtins.str]:
    args = []
    args += f"--default_root_dir {temp_dir}".split()
    args += f"--predict-csv {predict_csv}".split()
    args += "--enable_progress_bar false".split()
    args += "--num-workers 0".split()
    return args


def _handle_fast_dev_run(
    predict_args: typing.List[builtins.str],
) -> typing.List[builtins.str]:
    """py36-compatible pytorch-lightning has problem parsing fast_dev_run"""
    py_version = sys.version_info
    assert py_version.major == 3
    if py_version.minor > 6:
        predict_args += ["--fast_dev_run"]
    return predict_args


def _get_and_format_best_model_paths(args: typing.List[builtins.str]) -> builtins.str:
    best_model_paths = train(args, return_best_model_paths=True)
    assert isinstance(best_model_paths, list)
    best_model_paths_strlist = [str(bmp) for bmp in best_model_paths]
    best_model_paths_str = " ".join(best_model_paths_strlist)
    return best_model_paths_str


def test_cli(
    cli_train_args: typing.List[builtins.str],
    cli_predict_args: typing.List[builtins.str],
    train_csv: pathlib.Path,
) -> None:
    csv_ = " ".join([str(csv) for csv in [train_csv] * 2])
    cli_train_args += f"--train-csv {csv_}".split()
    cli_train_args += f"--valid-csv {csv_}".split()
    cli_train_args += "--patch-size 8 8 8".split()
    cli_train_args += "--track-metric dice".split()
    best_model_paths = _get_and_format_best_model_paths(cli_train_args)
    cli_predict_args += f"--model-path {best_model_paths}".split()
    cli_predict_args = _handle_fast_dev_run(cli_predict_args)
    retcode = predict(cli_predict_args)
    assert retcode == 0


def test_reorient_cli(
    cli_train_args: typing.List[builtins.str],
    cli_predict_args: typing.List[builtins.str],
    train_csv: pathlib.Path,
) -> None:
    csv_ = " ".join([str(csv) for csv in [train_csv] * 2])
    cli_train_args += f"--train-csv {csv_}".split()
    cli_train_args += f"--valid-csv {csv_}".split()
    cli_train_args += "--patch-size 8 8 8".split()
    cli_train_args += ["--reorient-to-canonical"]
    best_model_paths = _get_and_format_best_model_paths(cli_train_args)
    cli_predict_args += f"--model-path {best_model_paths}".split()
    cli_predict_args += ["--reorient-to-canonical"]
    cli_predict_args = _handle_fast_dev_run(cli_predict_args)
    retcode = predict(cli_predict_args)
    assert retcode == 0


def test_mixup_train_cli(
    cli_train_args: typing.List[builtins.str], train_csv: pathlib.Path
) -> None:
    csv_ = " ".join([str(csv) for csv in [train_csv] * 2])
    cli_train_args += f"--train-csv {csv_}".split()
    cli_train_args += f"--valid-csv {csv_}".split()
    cli_train_args += "--patch-size 8 8 8".split()
    cli_train_args += "--mixup".split()
    retcode = train(cli_train_args, return_best_model_paths=False)
    assert retcode == 0


def test_mlflow_train_cli(
    cli_train_args: typing.List[builtins.str],
    train_csv: pathlib.Path,
    temp_dir: pathlib.Path,
) -> None:
    csv_ = " ".join([str(csv) for csv in [train_csv] * 2])
    cli_train_args += f"--train-csv {csv_}".split()
    cli_train_args += f"--valid-csv {csv_}".split()
    cli_train_args += "--patch-size 8 8 8".split()
    for i in range(len(cli_train_args)):
        if cli_train_args[i] == "--n-epochs":
            cli_train_args[i + 1] = "4"
            break
    cli_train_args += f"--tracking-uri file:./{temp_dir}/ml-runs".split()
    retcode = train(cli_train_args, return_best_model_paths=False)
    assert retcode == 0


def test_multiclass_train_cli(
    cli_train_args: typing.List[builtins.str], train_csv: pathlib.Path
) -> None:
    csv_ = " ".join([str(csv) for csv in [train_csv] * 2])
    cli_train_args += f"--train-csv {csv_}".split()
    cli_train_args += f"--valid-csv {csv_}".split()
    cli_train_args += "--patch-size 8 8 8".split()
    cli_train_args += "--num-classes 2".split()
    retcode = train(cli_train_args, return_best_model_paths=False)
    assert retcode == 0


def test_patch_prediction_cli(
    cli_train_args: typing.List[builtins.str],
    cli_predict_args: typing.List[builtins.str],
    train_csv: pathlib.Path,
) -> None:
    csv_ = " ".join([str(csv) for csv in [train_csv] * 2])
    cli_train_args += f"--train-csv {csv_}".split()
    cli_train_args += f"--valid-csv {csv_}".split()
    cli_train_args += "--patch-size 8 8 8".split()
    cli_train_args += "--pos-sampling-weight 0.8".split()
    cli_train_args += ["--label-sampler"]
    best_model_paths = _get_and_format_best_model_paths(cli_train_args)
    cli_predict_args += f"--model-path {best_model_paths}".split()
    cli_predict_args += "--patch-size 32 32 32".split()
    cli_predict_args += "--patch-overlap 0 0 0".split()
    retcode = predict(cli_predict_args)
    assert retcode == 0


@pytest.fixture
def cli_predict_image_args(
    temp_dir: pathlib.Path, data_dir: pathlib.Path
) -> typing.List[builtins.str]:
    image_path = data_dir / "img.nii.gz"
    out_path = temp_dir / "out.nii.gz"
    args = []
    args += f"--default_root_dir {temp_dir}".split()
    args += f"--t1 {image_path}".split()
    args += f"--t2 {image_path}".split()
    args += f"--out {out_path}".split()
    args += "--enable_progress_bar false".split()
    args += "--num-workers 0".split()
    return args


def test_predict_image_cli(
    cli_train_args: typing.List[builtins.str],
    cli_predict_image_args: typing.List[builtins.str],
    train_csv: pathlib.Path,
) -> None:
    csv_ = " ".join([str(csv) for csv in [train_csv] * 2])
    cli_train_args += f"--train-csv {csv_}".split()
    cli_train_args += f"--valid-csv {csv_}".split()
    cli_train_args += "--patch-size 8 8 8".split()
    best_model_paths = _get_and_format_best_model_paths(cli_train_args)
    cli_predict_image_args += f"--model-path {best_model_paths}".split()
    cli_predict_image_args = _handle_fast_dev_run(cli_predict_image_args)
    retcode = predict_image(cli_predict_image_args)
    assert retcode == 0


def test_pseudo3d_cli(
    cli_train_args: typing.List[builtins.str],
    cli_predict_args: typing.List[builtins.str],
    train_csv: pathlib.Path,
) -> None:
    csv_ = " ".join([str(csv) for csv in [train_csv] * 3])
    cli_train_args += f"--train-csv {csv_}".split()
    cli_train_args += f"--valid-csv {csv_}".split()
    cli_train_args += "--patch-size 8 8".split()
    cli_train_args += "--pseudo3d-dim 0 1 2".split()
    cli_train_args += "--pseudo3d-size 31".split()
    best_model_paths = _get_and_format_best_model_paths(cli_train_args)
    cli_predict_args += f"--model-path {best_model_paths}".split()
    cli_predict_args += "--patch-size None None".split()
    cli_predict_args += "--pseudo3d-dim 0 1 2".split()
    cli_predict_args += "--pseudo3d-size 31".split()
    cli_predict_args += "--patch-overlap 0 0 0".split()
    retcode = predict(cli_predict_args)
    assert retcode == 0


def test_union_aggregate_cli(
    cli_train_args: typing.List[builtins.str],
    cli_predict_args: typing.List[builtins.str],
    train_csv: pathlib.Path,
) -> None:
    csv_ = " ".join([str(csv) for csv in [train_csv] * 2])
    cli_train_args += f"--train-csv {csv_}".split()
    cli_train_args += f"--valid-csv {csv_}".split()
    cli_train_args += "--patch-size 8 8".split()
    cli_train_args += "--pseudo3d-dim 0 1".split()
    cli_train_args += "--pseudo3d-size 31".split()
    best_model_paths = _get_and_format_best_model_paths(cli_train_args)
    cli_predict_args += f"--model-path {best_model_paths}".split()
    cli_predict_args += "--patch-size None None".split()
    cli_predict_args += "--pseudo3d-dim 0 1".split()
    cli_predict_args += "--pseudo3d-size 31".split()
    cli_predict_args += "--patch-overlap 0 0 0".split()
    cli_predict_args += "--aggregation-type union".split()
    retcode = predict(cli_predict_args)
    assert retcode == 0


def test_vote_aggregate_cli(
    cli_train_args: typing.List[builtins.str],
    cli_predict_args: typing.List[builtins.str],
    train_csv: pathlib.Path,
) -> None:
    csv_ = " ".join([str(csv) for csv in [train_csv] * 2])
    cli_train_args += f"--train-csv {csv_}".split()
    cli_train_args += f"--valid-csv {csv_}".split()
    cli_train_args += "--patch-size 8 8".split()
    cli_train_args += "--pseudo3d-dim 0 1".split()
    cli_train_args += "--pseudo3d-size 31".split()
    best_model_paths = _get_and_format_best_model_paths(cli_train_args)
    cli_predict_args += f"--model-path {best_model_paths}".split()
    cli_predict_args += "--patch-size None None".split()
    cli_predict_args += "--pseudo3d-dim 0 1".split()
    cli_predict_args += "--pseudo3d-size 31".split()
    cli_predict_args += "--patch-overlap 0 0 0".split()
    cli_predict_args += "--aggregation-type vote".split()
    retcode = predict(cli_predict_args)
    assert retcode == 0


def test_pseudo3d_all_cli(
    cli_train_args: typing.List[builtins.str],
    cli_predict_args: typing.List[builtins.str],
    train_csv: pathlib.Path,
) -> None:
    cli_train_args += f"--train-csv {train_csv}".split()
    cli_train_args += f"--valid-csv {train_csv}".split()
    cli_train_args += "--patch-size 8 8".split()
    cli_train_args += "--pseudo3d-dim all".split()
    cli_train_args += "--pseudo3d-size 31".split()
    cli_train_args += ["--random-validation-patches"]
    best_model_paths = _get_and_format_best_model_paths(cli_train_args)
    cli_predict_args += f"--model-path {best_model_paths}".split()
    cli_predict_args += "--patch-size None None".split()
    cli_predict_args += "--pseudo3d-dim 0".split()
    cli_predict_args += "--pseudo3d-size 31".split()
    cli_predict_args += "--patch-overlap 0 0 0".split()
    retcode = predict(cli_predict_args)
    assert retcode == 0


def test_pseudo3d_all_interp_cli(
    cli_train_args: typing.List[builtins.str],
    cli_predict_args: typing.List[builtins.str],
    train_csv: pathlib.Path,
) -> None:
    cli_train_args += f"--train-csv {train_csv}".split()
    cli_train_args += f"--valid-csv {train_csv}".split()
    cli_train_args += "--patch-size 8 8".split()
    cli_train_args += "--pseudo3d-dim all".split()
    cli_train_args += "--pseudo3d-size 31".split()
    cli_train_args += "--resize-method interpolate".split()
    cli_train_args += ["--random-validation-patches"]
    best_model_paths = _get_and_format_best_model_paths(cli_train_args)
    cli_predict_args += f"--model-path {best_model_paths}".split()
    cli_predict_args += "--patch-size None None".split()
    cli_predict_args += "--pseudo3d-dim 0".split()
    cli_predict_args += "--pseudo3d-size 31".split()
    cli_predict_args += "--patch-overlap 0 0 0".split()
    retcode = predict(cli_predict_args)
    assert retcode == 0
