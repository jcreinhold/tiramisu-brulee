#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.tests.test_lesion_seg

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 18, 2021
"""

from pathlib import Path
from typing import List

import pytest

from tiramisu_brulee.experiment.lesion_seg.seg import train, predict, predict_image


@pytest.fixture
def file() -> Path:
    return Path(__file__).resolve()


@pytest.fixture
def cwd(file: Path) -> Path:
    return file.parent


@pytest.fixture
def data_dir(cwd: Path) -> Path:
    return cwd / "test_data"


@pytest.fixture(scope="session")
def temp_dir(tmpdir_factory) -> Path:
    return Path(tmpdir_factory.mktemp("out"))


def _create_csv(temp_dir: Path, data_dir: Path, stage: str) -> Path:
    csv_path = temp_dir / f"{stage}.csv"
    image_path = data_dir / "img.nii.gz"
    label_path = data_dir / "mask.nii.gz"
    out_path = temp_dir / "out.nii.gz"
    headers = "subject,label,t1"
    filenames = f"subj1,{label_path},{image_path}"
    if stage == "predict":
        headers += ",out\n"
        filenames += f",{out_path}"
    else:
        headers += "\n"
    with open(csv_path, "w") as f:
        f.write(headers)
        f.write(filenames)
    return csv_path


@pytest.fixture
def train_csv(temp_dir: Path, data_dir: Path) -> Path:
    return _create_csv(temp_dir, data_dir, "train")


@pytest.fixture
def predict_csv(temp_dir: Path, data_dir: Path) -> Path:
    return _create_csv(temp_dir, data_dir, "predict")


@pytest.fixture
def cli_train_args(temp_dir: Path, train_csv: Path) -> List[str]:
    csv_ = " ".join([str(csv) for csv in [train_csv, train_csv]])
    args = []
    args += f"--default_root_dir {temp_dir}".split()
    args += f"--train-csv {csv_}".split()
    args += f"--valid-csv {csv_}".split()
    args += "--batch-size 1".split()
    args += "--patch-size 16 16 16".split()
    args += "--queue-length 1".split()
    args += "--samples-per-volume 1".split()
    args += "--n-epochs 10".split()
    args += "--down-blocks 2 2".split()
    args += "--up-blocks 2 2".split()
    args += "--bottleneck-layers 2".split()
    args += "--first-conv-out-channels 2".split()
    args += "--num-workers 0".split()
    return args


@pytest.fixture
def cli_predict_args(temp_dir: Path, predict_csv: Path) -> List[str]:
    args = []
    args += f"--default_root_dir {temp_dir}".split()
    args += f"--predict-csv {predict_csv}".split()
    args += "--num-workers 0".split()
    args += ["--fast_dev_run"]
    return args


def test_cli(cli_train_args, cli_predict_args):
    best_model_paths = train(cli_train_args, True)
    best_model_paths = " ".join([str(bmp) for bmp in best_model_paths])
    cli_predict_args += f"--model-path {best_model_paths}".split()
    retcode = predict(cli_predict_args)
    assert retcode == 0


@pytest.fixture
def cli_predict_image_args(temp_dir: Path, data_dir: Path) -> List[str]:
    image_path = data_dir / "img.nii.gz"
    out_path = temp_dir / "out.nii.gz"
    args = []
    args += f"--default_root_dir {temp_dir}".split()
    args += f"--t1 {image_path}".split()
    args += f"--out {out_path}".split()
    args += "--num-workers 0".split()
    args += ["--fast_dev_run"]
    return args


def test_predict_image_cli(cli_train_args, cli_predict_image_args):
    best_model_paths = train(cli_train_args, True)
    best_model_paths = " ".join([str(bmp) for bmp in best_model_paths])
    cli_predict_image_args += f"--model-path {best_model_paths}".split()
    retcode = predict_image(cli_predict_image_args)
    assert retcode == 0
