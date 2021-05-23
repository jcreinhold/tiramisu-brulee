#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_seg.data

load and process data for lesion segmentation

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 17, 2021
"""

__all__ = [
    "csv_to_subjectlist",
    "glob_ext",
    "LesionSegDataModulePredict",
    "LesionSegDataModuleTrain",
    "Mixup",
]

from glob import glob
from logging import getLogger
from os.path import join
from typing import List, Optional, Tuple

from jsonargparse import ArgumentParser
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.distributions as D
import torchio as tio

from tiramisu_brulee.experiment.lesion_seg.parse import (
    file_path,
    nonnegative_int,
    positive_float,
    positive_int,
)
from tiramisu_brulee.experiment.lesion_seg.util import reshape_for_broadcasting

VALID_NAMES = ("ct", "flair", "pd", "t1", "t1c", "t2", "label", "weight", "div", "out")

logger = getLogger(__name__)


class LesionSegDataModuleBase(pl.LightningDataModule):
    def _determine_input(
        self,
        subject_list: List[tio.Subject],
        other_subject_list: Optional[List[tio.Subject]] = None,
    ):
        """
        assume all columns except:
            `name`, `label`, `div`, `weight`, or `out`
        are some type of non-categorical image
        """
        exclude = ("name", "label", "div", "weight", "out")
        subject = subject_list[0]  # arbitrarily pick the first element
        inputs = []
        for key in subject:
            if key not in exclude:
                inputs.append(key)
        if len(inputs) == 0:
            msg = (
                "No inputs detected in CSV. Expect columns like "
                "`t1` with corresponding paths to NIfTI files."
            )
            raise ValueError(msg)
        if other_subject_list is not None:
            other_subject = other_subject_list[0]
            for key in inputs:
                if key not in other_subject:
                    msg = "Validation CSV fields not the same as training CSV"
                    raise ValueError(msg)
            if "label" not in subject or "label" not in other_subject:
                msg = "`label` field expected in both " "training and validation CSV."
                raise ValueError(msg)
            if ("div" in subject) ^ ("div" in other_subject):
                msg = (
                    "If `div` present in one of the training "
                    "or validation CSVs, it is expected in "
                    "both training and validation CSV."
                )
                raise ValueError(msg)
        if "div" in subject and len(inputs) > 1:
            msg = f"If using `div`, expect only 1 input. Got {inputs}."
            raise ValueError(msg)
        self._use_div = "div" in subject
        self._input_fields = tuple(sorted(inputs))

    def _div_batch(self, batch: Tensor, div: Tensor) -> Tensor:
        with torch.no_grad():
            batch /= reshape_for_broadcasting(div, batch.ndim)
        return batch

    def _collate_fn(self, batch: dict) -> Tensor:
        if isinstance(batch, list):
            batch = default_collate(batch)
        inputs = []
        for field in self._input_fields:
            inputs.append(batch[field][tio.DATA])
        src = torch.cat(inputs, dim=1)
        if self._use_div:
            src = self._div_batch(src, batch["div"])
        return src


class LesionSegDataModuleTrain(LesionSegDataModuleBase):
    def __init__(
        self,
        train_subject_list: List[tio.Subject],
        val_subject_list: List[tio.Subject],
        batch_size: int = 2,
        patch_size: List[int] = (96, 96, 96),
        queue_length: int = 200,
        samples_per_volume: int = 10,
        num_workers: int = 16,
        label_sampler: bool = False,
        spatial_augmentation: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.train_subject_list = train_subject_list
        self.val_subject_list = val_subject_list
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume
        self.num_workers = num_workers
        self.label_sampler = label_sampler
        self.spatial_augmentation = spatial_augmentation

    @classmethod
    def from_csv(cls, train_csv: str, valid_csv: str, *args, **kwargs):
        tsl = csv_to_subjectlist(train_csv)
        vsl = csv_to_subjectlist(valid_csv)
        return cls(tsl, vsl, *args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        self._determine_input(self.train_subject_list, self.val_subject_list)
        self._setup_train_dataset()
        self._setup_val_dataset()

    def train_dataloader(self) -> DataLoader:
        sampler = self._get_train_sampler()
        patches_queue = tio.Queue(
            self.train_dataset,
            self.queue_length,
            self.samples_per_volume,
            sampler,
            num_workers=self.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        train_dataloader = DataLoader(
            patches_queue, batch_size=self.batch_size, collate_fn=self._collate_fn,
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
        return val_dataloader

    def _get_train_augmentation(self):
        transforms = [LabelToFloat()]
        if self.spatial_augmentation:
            spatial = tio.OneOf(
                {tio.RandomAffine(): 0.8, tio.RandomElasticDeformation(): 0.2}, p=0.75,
            )
            transforms.insert(0, spatial)
        transform = tio.Compose(transforms)
        return transform

    def _get_train_sampler(self):
        ps = self.patch_size
        if self.label_sampler:
            return tio.LabelSampler(ps)
        else:
            return tio.UniformSampler(ps)

    def _setup_train_dataset(self):
        transform = self._get_train_augmentation()
        subjects_dataset = tio.SubjectsDataset(
            self.train_subject_list, transform=transform,
        )
        self.train_dataset = subjects_dataset

    def _get_val_augmentation(self):
        crop = tio.CropOrPad(self.patch_size)
        transform = tio.Compose([crop, LabelToFloat()])
        return transform

    def _setup_val_dataset(self):
        transform = self._get_val_augmentation()
        subjects_dataset = tio.SubjectsDataset(
            self.val_subject_list, transform=transform
        )
        self.val_dataset = subjects_dataset

    def _collate_fn(self, batch: dict) -> Tuple[Tensor, Tensor]:
        batch = default_collate(batch)
        src = super()._collate_fn(batch)
        tgt = batch["label"][tio.DATA]
        return src, tgt

    @staticmethod
    def add_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument(
            "--train-csv",
            type=file_path(),
            nargs="+",
            required=True,
            default=["SET ME!"],
            help="path(s) to csv(s) with training images",
        )
        parser.add_argument(
            "--valid-csv",
            type=file_path(),
            nargs="+",
            required=True,
            default=["SET ME!"],
            help="path(s) to csv(s) with validation images",
        )
        parser.add_argument(
            "-bs",
            "--batch-size",
            type=positive_int(),
            default=2,
            help="training/validation batch size",
        )
        parser.add_argument(
            "-ps",
            "--patch-size",
            type=positive_int(),
            nargs=3,
            default=[96, 96, 96],
            help="training/validation patch size extracted from image",
        )
        parser.add_argument(
            "-nw",
            "--num-workers",
            type=nonnegative_int(),
            default=16,
            help="number of CPUs to use for loading data",
        )
        parser.add_argument(
            "-ql",
            "--queue-length",
            type=positive_int(),
            default=200,
            help="queue length for torchio sampler",
        )
        parser.add_argument(
            "-spv",
            "--samples-per-volume",
            type=positive_int(),
            default=10,
            help="samples per volume for torchio sampler",
        )
        parser.add_argument(
            "-ls",
            "--label-sampler",
            action="store_true",
            default=False,
            help="use label sampler instead of uniform",
        )
        parser.add_argument(
            "-sa",
            "--spatial-augmentation",
            action="store_true",
            default=False,
            help="use spatial (affine and elastic) data augmentation",
        )
        return parent_parser


class LesionSegDataModulePredict(LesionSegDataModuleBase):
    def __init__(
        self,
        subject_list: List[tio.Subject],
        batch_size: int = 1,
        num_workers: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.subject_list = subject_list
        self.batch_size = batch_size
        self.num_workers = num_workers

    @classmethod
    def from_csv(cls, predict_csv: str, *args, **kwargs):
        subject_list = csv_to_subjectlist(predict_csv)
        return cls(subject_list, *args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        self._determine_input(self.subject_list)
        self._setup_predict_dataset()

    def predict_dataloader(self) -> DataLoader:
        pred_dataloader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
        return pred_dataloader

    def _setup_predict_dataset(self):
        subjects_dataset = tio.SubjectsDataset(
            self.subject_list, transform=LabelToFloat(),
        )
        self.predict_dataset = subjects_dataset

    def _collate_fn(self, batch: dict) -> Tensor:
        batch = default_collate(batch)
        src = super()._collate_fn(batch)
        # assume affine matrices same across modalities
        # so arbitrarily choose first
        field = self._input_fields[0]
        out = dict(
            src=src,
            affine=batch[field][tio.AFFINE],
            out=batch["out"],  # path to save the prediction
        )
        return out

    @staticmethod
    def add_arguments(
        parent_parser: ArgumentParser, add_csv: bool = True
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Data")
        if add_csv:
            parser.add_argument(
                "--predict-csv",
                type=file_path(),
                required=True,
                default="SET ME!",
                help="path to csv of prediction images",
            )
        parser.add_argument(
            "-bs",
            "--batch-size",
            type=positive_int(),
            default=1,
            help="number of images to run at a time",
        )
        parser.add_argument(
            "-nw",
            "--num-workers",
            type=nonnegative_int(),
            default=16,
            help="number of CPUs to use for loading data",
        )
        return parent_parser


class Mixup:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def _mixup_dist(self, device: torch.device) -> D.Distribution:
        alpha = torch.tensor(self.alpha, device=device)
        dist = D.Beta(alpha, alpha)
        return dist

    def _mixup_coef(self, batch_size: int, device: torch.device) -> Tensor:
        dist = self._mixup_dist(device)
        return dist.sample(batch_size)

    def __call__(self, src: Tensor, tgt: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            batch_size = src.shape[0]
            perm = torch.randperm(batch_size)
            lam = self._mixup_coef(batch_size, src.device)
            lam = reshape_for_broadcasting(lam, src.ndim)
            src = lam * src + (1 - lam) * src[perm]
            tgt = tgt.float()
            tgt = lam * tgt + (1 - lam) * tgt[perm]
        return src, tgt

    @staticmethod
    def add_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Mixup")
        parser.add_argument(
            "-mu",
            "--mixup",
            action="store_true",
            default=False,
            help="use mixup during training",
        )
        parser.add_argument(
            "-ma",
            "--mixup-alpha",
            type=positive_float(),
            default=0.4,
            help="mixup alpha parameter for beta distribution",
        )
        return parent_parser


def _to_float(tensor: Tensor) -> Tensor:
    """ create separate func b/c lambda not pickle-able """
    return tensor.float()


def LabelToFloat() -> tio.Transform:
    """ cast a label image (usually uint8) to a float """
    return tio.Lambda(_to_float, types_to_apply=[tio.LABEL],)


def _get_type(name: str):
    name_ = name.lower()
    if name_ == "label":
        type_ = tio.LABEL
    elif name_ == "weight" or name_ == "div":
        type_ = "float"
    elif name_ == "out":
        type_ = "path"
    elif name_ in VALID_NAMES:
        type_ = tio.INTENSITY
    else:
        logger.warning(
            f"{name} not in known {VALID_NAMES}. " f"Assuming an non-label image type."
        )
        type_ = tio.INTENSITY
    return type_


def glob_ext(path: str, ext: str = "*.nii*") -> List[str]:
    """ grab all `ext` files in a directory and sort them for consistency """
    fns = sorted(glob(join(path, ext)))
    return fns


def csv_to_subjectlist(filename: str) -> List[tio.Subject]:
    """ Convert a csv file to a list of torchio subjects

    Args:
        filename: Path to csv file formatted with
            `subject` in a column, describing the
            id/name of the subject (must be unique).
            Row will fill in the filenames per type.
            Other columns headers must be one of:
            ct, flair, label, pd, t1, t1c, t2, weight, div
            (`label` should correspond to a
             segmentation mask)
            (`weight` and `div` should correspond to a float)

    Returns:
        subject_list (List[torchio.Subject]): list of torchio Subjects
    """
    df = pd.read_csv(filename, index_col="subject")
    names = df.columns.to_list()
    if any([name not in VALID_NAMES for name in names]):
        raise ValueError(f"Column name needs to be in {VALID_NAMES}")

    subject_list = []
    for row in df.iterrows():
        subject_name = row[0]
        data = {}
        for name in names:
            val_type = _get_type(name)
            val = row[1][name]
            if val_type == "float":
                data[name] = torch.tensor(val, dtype=torch.float32)
            elif val_type == "path":
                data[name] = val
            else:
                data[name] = tio.Image(val, type=val_type)
        subject = tio.Subject(name=subject_name, **data)
        subject_list.append(subject)

    return subject_list
