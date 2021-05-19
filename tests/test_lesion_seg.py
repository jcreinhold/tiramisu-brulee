#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.tests.test_lesion_seg

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: May 18, 2021
"""

from pathlib import Path
from typing import List

import pytest

from tiramisu_brulee.experiment.lesion_seg.seg import train, predict


@pytest.fixture
def file() -> Path:
    return Path(__file__).resolve()


@pytest.fixture
def cwd(file: Path) -> Path:
    return file.parent


@pytest.fixture
def data_dir(cwd: Path) -> Path:
    return cwd / 'test_data'


@pytest.fixture(scope='session')
def temp_dir(tmpdir_factory) -> Path:
    return Path(tmpdir_factory.mktemp('out'))


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
    args = []
    args += f'--default_root_dir {temp_dir}'.split()
    args += f'--train-csv {train_csv}'.split()
    args += f'--valid-csv {train_csv}'.split()
    return args


@pytest.fixture
def cli_predict_args(temp_dir: Path, predict_csv: Path) -> List[str]:
    args = []
    args += f'--default_root_dir {temp_dir}'.split()
    args += f'--predict-csv {train_csv}'.split()
    args += f'--fast_dev {train_csv}'.split()
    return args


def test_cli(cli_train_args, cli_predict_args):
    retcode = train()
    assert retcode == 0
