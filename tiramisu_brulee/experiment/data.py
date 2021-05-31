#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.data

load and process data for training/prediction
for segmentation tasks

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 17, 2021
"""

__all__ = [
    "csv_to_subjectlist",
    "LesionSegDataModulePredictBase",
    "LesionSegDataModuleTrain",
    "Mixup",
]

from logging import getLogger
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

from jsonargparse import ArgumentParser
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.distributions as D
import torchio as tio

from tiramisu_brulee.experiment.type import (
    file_path,
    nonnegative_int,
    PatchShapeOption,
    PatchShape,
    positive_float,
    positive_int,
    positive_odd_int,
    positive_int_or_none,
)
from tiramisu_brulee.experiment.util import reshape_for_broadcasting

VALID_NAMES = ("ct", "flair", "pd", "t1", "t1c", "t2", "label", "weight", "div", "out")

logger = getLogger(__name__)


class LesionSegDataModuleBase(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        patch_size: Optional[PatchShape] = None,
        num_workers: int = 16,
        pseudo3d_dim: Optional[int] = None,
        pseudo3d_size: Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pseudo3d_dim = pseudo3d_dim
        self.pseudo3d_size = pseudo3d_size
        if self._use_pseudo3d and self.pseudo3d_size is None:
            raise ValueError(
                "If pseudo3d_dim provided, pseudo3d_size must be provided."
            )
        self.patch_size = self._determine_patch_size(patch_size)

    def _determine_input(
        self,
        subjects: Union[tio.Subject, List[tio.Subject]],
        other_subjects: Optional[List[tio.Subject]] = None,
    ):
        """
        assume all columns except:
            `name`, `label`, `div`, `weight`, or `out`
        are some type of non-categorical image
        """
        exclude = ("name", "label", "div", "weight", "out")
        if isinstance(subjects, list):
            subject = subjects[0]  # arbitrarily pick the first element
        else:
            subject = subjects
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
        if other_subjects is not None:
            other_subject = other_subjects[0]
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
        self._use_div = "div" in subject
        self._input_fields = tuple(sorted(inputs))

    @staticmethod
    def _div_batch(batch: Tensor, div: Tensor) -> Tensor:
        with torch.no_grad():
            batch /= reshape_for_broadcasting(div, batch.ndim)
        return batch

    def _default_collate_fn(self, batch: dict, cat_dim: Optional[int] = None) -> Tensor:
        if isinstance(batch, list):
            batch = default_collate(batch)
        inputs = []
        for field in self._input_fields:
            inputs.append(batch[field][tio.DATA])
        cat_dim_ = cat_dim or 1
        src = torch.cat(inputs, dim=cat_dim_)
        if cat_dim is not None:
            # if axis is not None, use pseudo3d images
            src.swapaxes_(1, cat_dim_)
            src.squeeze_()
            if src.ndim == 3:  # batch size of 1
                src.unsqueeze_(0)
            assert src.ndim == 4
        if self._use_div:
            src = self._div_batch(src, batch["div"])
        return src

    @staticmethod
    def _pseudo3d_label(label: Tensor, pseudo3d_dim: int):
        assert label.ndim == 5, "expects label with shape NxCxHxWxD"
        if pseudo3d_dim == 0:
            mid_channel = label.shape[2] // 2
            label = label[..., mid_channel, :, :]
        elif pseudo3d_dim == 1:
            mid_channel = label.shape[3] // 2
            label = label[..., :, mid_channel, :]
        elif pseudo3d_dim == 2:
            mid_channel = label.shape[4] // 2
            label = label[..., :, :, mid_channel]
        else:
            raise ValueError(f"pseudo3d_dim must be 0, 1, or 2. Got {pseudo3d_dim}.")
        return label

    def _determine_patch_size(self, patch_size: PatchShapeOption) -> PatchShapeOption:
        if patch_size is None:
            return patch_size
        patch_size = list(patch_size)
        if self._use_pseudo3d and len(patch_size) != 2:
            raise ValueError(
                "If using pseudo3d, patch size must contain only 2 values."
            )
        if self._use_pseudo3d:
            patch_size.insert(self.pseudo3d_dim, self.pseudo3d_size)
        return tuple(patch_size)

    @property
    def _use_pseudo3d(self) -> bool:
        return self.pseudo3d_dim is not None


class LesionSegDataModuleTrain(LesionSegDataModuleBase):
    """Data module for training and validation for lesion segmentation

    Args:
        train_subject_list (List[tio.Subject]):
            list of torchio.Subject for training
        val_subject_list (List[tio.Subject]):
            list of torchio.Subject for validation
        batch_size (int): batch size for training/validation
        patch_size (PatchShape): patch size for training/validation
        queue_length (int): Maximum number of patches that can be
            stored in the queue. Using a large number means that
            the queue needs to be filled less often, but more CPU
            memory is needed to store the patches.
        samples_per_volume (int): Number of patches to extract from
            each volume. A small number of patches ensures a large
            variability in the queue, but training will be slower.
        num_workers (int): number of subprocesses for data loading
        label_sampler (bool): sample patches centered on positive labels
        spatial_augmentation (bool): use random affine and elastic
            data augmentation for training
        pseudo3d_dim (Optional[int]): concatenate images along this
            axis and swap it for channel dimension
    """

    def __init__(
        self,
        train_subject_list: List[tio.Subject],
        val_subject_list: List[tio.Subject],
        batch_size: int = 2,
        patch_size: PatchShape = (96, 96, 96),
        queue_length: int = 200,
        samples_per_volume: int = 10,
        num_workers: int = 16,
        label_sampler: bool = False,
        spatial_augmentation: bool = False,
        pseudo3d_dim: Optional[int] = None,
        pseudo3d_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            batch_size, patch_size, num_workers, pseudo3d_dim, pseudo3d_size,
        )
        self.train_subject_list = train_subject_list
        self.val_subject_list = val_subject_list
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume
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
        if self._use_pseudo3d:
            sampler = self._get_val_sampler()
            patches_queue = tio.Queue(
                self.val_dataset,
                self.queue_length,
                self.samples_per_volume,
                sampler,
                num_workers=self.num_workers,
                shuffle_subjects=False,
                shuffle_patches=False,
            )
            val_dataloader = DataLoader(
                patches_queue, batch_size=self.batch_size, collate_fn=self._collate_fn,
            )
        else:
            val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self._collate_fn,
            )
        return val_dataloader

    def _get_train_augmentation(self):
        transforms = [label_to_float()]
        if self.spatial_augmentation:
            spatial = tio.OneOf(
                {tio.RandomAffine(): 0.8, tio.RandomElasticDeformation(): 0.2}, p=0.75,
            )
            transforms.insert(0, spatial)
        transform = tio.Compose(transforms)
        return transform

    def _get_train_sampler(self):
        if self.label_sampler:
            return tio.LabelSampler(self.patch_size)
        else:
            return tio.UniformSampler(self.patch_size)

    def _setup_train_dataset(self):
        transform = self._get_train_augmentation()
        subjects_dataset = tio.SubjectsDataset(
            self.train_subject_list, transform=transform,
        )
        self.train_dataset = subjects_dataset

    def _get_val_augmentation(self):
        transforms = [label_to_float()]
        if not self._use_pseudo3d:
            transforms.insert(0, tio.CropOrPad(self.patch_size))
        transform = tio.Compose(transforms)
        return transform

    def _get_val_sampler(self):
        return tio.LabelSampler(self.patch_size)

    def _setup_val_dataset(self):
        transform = self._get_val_augmentation()
        subjects_dataset = tio.SubjectsDataset(
            self.val_subject_list, transform=transform,
        )
        self.val_dataset = subjects_dataset

    def _collate_fn(self, batch: dict) -> Tuple[Tensor, Tensor]:
        batch = default_collate(batch)
        # p3d is the offset by batch/channel dims if pseudo3d used
        p3d = self.pseudo3d_dim + 2 if self._use_pseudo3d else None
        src = self._default_collate_fn(batch, p3d)
        tgt = batch["label"][tio.DATA]
        if self.pseudo3d_dim is not None:
            tgt = self._pseudo3d_label(tgt, self.pseudo3d_dim)
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
            help="path(s) to CSV(s) with training images",
        )
        parser.add_argument(
            "--valid-csv",
            type=file_path(),
            nargs="+",
            required=True,
            default=["SET ME!"],
            help="path(s) to CSV(s) with validation images",
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
            nargs="+",
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
        parser.add_argument(
            "-p3d",
            "--pseudo3d-dim",
            type=nonnegative_int(),
            nargs="+",
            choices=(0, 1, 2),
            default=None,
            help="dim on which to concatenate the images for input "
            "to a 2D network. If provided, either provide 1 value"
            "to be used for all train/valid CSVs or provide N values "
            "corresponding to the N train/valid CSVs. If not provided, "
            "use 3D network.",
        )
        parser.add_argument(
            "-p3s",
            "--pseudo3d-size",
            type=positive_odd_int(),
            default=None,
            help="size of the pseudo3d dimension (if -p3d provided)",
        )
        return parent_parser


class LesionSegDataModulePredictBase(LesionSegDataModuleBase):
    def __init__(
        self,
        subjects: Union[tio.Subject, List[tio.Subject]],
        batch_size: int,
        patch_size: Optional[PatchShape] = None,
        num_workers: int = 16,
        pseudo3d_dim: Optional[int] = None,
        pseudo3d_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            batch_size, patch_size, num_workers, pseudo3d_dim, pseudo3d_size
        )
        self.subjects = subjects

    def setup(self, stage: Optional[str] = None):
        self._determine_input(self.subjects)
        self._setup_predict_dataset()

    def predict_dataloader(self) -> DataLoader:
        pred_dataloader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
        self.total_batches = len(pred_dataloader)
        return pred_dataloader

    def _setup_predict_dataset(self):
        self.predict_dataset = None
        raise NotImplementedError

    def _collate_fn(self, batch: dict) -> Tensor:
        raise NotImplementedError

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
            help="number of patches to run at a time",
        )
        parser.add_argument(
            "-ps",
            "--patch-size",
            type=positive_int_or_none(),
            nargs="+",
            default=None,
            help="shape of patches (None -> crop image to foreground)",
        )
        parser.add_argument(
            "-po",
            "--patch-overlap",
            type=nonnegative_int(),
            nargs=3,
            default=None,
            help="patches will overlap by this much (None -> patch-size // 2)",
        )
        parser.add_argument(
            "-nw",
            "--num-workers",
            type=nonnegative_int(),
            default=16,
            help="number of CPUs to use for loading data",
        )
        parser.add_argument(
            "-p3d",
            "--pseudo3d-dim",
            type=nonnegative_int(),
            nargs="+",
            choices=(0, 1, 2),
            default=None,
            help="dim on which to concatenate the images for input "
            "to a 2D network. If provided, either provide 1 value"
            "to be used for all train/valid CSVs or provide N values "
            "corresponding to the N train/valid CSVs. If not provided, "
            "use 3D network.",
        )
        parser.add_argument(
            "-p3s",
            "--pseudo3d-size",
            type=positive_odd_int(),
            default=None,
            help="size of the pseudo3d dimension (if -p3d provided)",
        )
        return parent_parser


class LesionSegDataModulePredictWhole(LesionSegDataModulePredictBase):
    """Data module for whole-image prediction for lesion segmentation

    Args:
        subjects (List[tio.Subject]):
            list of torchio.Subject for prediction
        batch_size (int): number of images to predict at a time
        num_workers (int):
            number of subprocesses to use for data loading
    """

    def __init__(
        self,
        subjects: Union[tio.Subject, List[tio.Subject]],
        batch_size: int,
        num_workers: int = 16,
        **kwargs,
    ):
        super().__init__(
            subjects, batch_size, None, num_workers, None, None,
        )

    @classmethod
    def from_csv(cls, predict_csv: str, *args, **kwargs):
        subject_list = csv_to_subjectlist(predict_csv)
        return cls(subject_list, *args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        self._determine_input(self.subjects)
        self._setup_predict_dataset()

    def _setup_predict_dataset(self):
        subjects_dataset = tio.SubjectsDataset(
            self.subjects, transform=label_to_float(),
        )
        self.predict_dataset = subjects_dataset

    def _collate_fn(self, batch: dict) -> Tensor:
        batch = default_collate(batch)
        src = self._default_collate_fn(batch)
        # assume affine matrices same across modalities
        # so arbitrarily choose first
        field = self._input_fields[0]
        out = dict(
            src=src,
            affine=batch[field][tio.AFFINE],
            out=batch["out"],  # path to save the prediction
        )
        return out


class LesionSegDataModulePredictPatches(LesionSegDataModulePredictBase):
    """Data module for patch-based prediction for lesion segmentation

    Args:
        subject (tio.Subject):
            a torchio.Subject for prediction
        batch_size (int): number of patches to predict at a time
        patch_size (OptionalPatchShape): patch size for training/validation
            if any element is None, use the corresponding image dim
        patch_overlap (Optional[Tuple[int, int, int]]):
            overlap of each patch, if None then patch_size // 2
        num_workers (int):
            number of subprocesses to use for data loading
        pseudo3d_dim (Optional[int]): concatenate images along this
            axis and swap it for channel dimension
        pseudo3d_size (Optional[int]): number of slices to concatenate
            if pseudo3d_dim provided, must be an odd (usually small) integer
    """

    def __init__(
        self,
        subject: tio.Subject,
        batch_size: int = 1,
        patch_size: PatchShapeOption = (96, 96, 96),
        patch_overlap: Optional[PatchShape] = None,
        num_workers: int = 16,
        pseudo3d_dim: Optional[int] = None,
        pseudo3d_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            subject, batch_size, patch_size, num_workers, pseudo3d_dim, pseudo3d_size,
        )
        ps = self.patch_size  # result from _determine_patch_size
        self._set_patch_size(subject, ps)
        self.patch_overlap = patch_overlap or self._default_overlap(ps)

    def _set_patch_size(
        self, subject: tio.Subject, patch_size: PatchShapeOption
    ) -> PatchShapeOption:
        if any([ps is None for ps in patch_size]):
            image_dim = subject.spatial_shape
            patch_size = [ps or dim for ps, dim in zip(patch_size, image_dim)]
        if len(patch_size) != 3:
            raise ValueError(
                "Patch size must have length 3 here. "
                f"Got {len(patch_size)}. Something went wrong."
            )
        self.patch_size = tuple(patch_size)

    def _default_overlap(self, patch_size: PatchShape) -> PatchShape:
        patch_overlap = []
        for i, ps in enumerate(patch_size):
            if ps is None:
                patch_overlap.append(0)
                continue
            if i == self.pseudo3d_dim:
                patch_overlap.append(self.pseudo3d_size - 1)
                continue
            overlap = ps // 2
            if overlap % 2:
                overlap += 1
            patch_overlap.append(overlap)
        return patch_overlap

    def _setup_predict_dataset(self):
        grid_sampler = tio.GridSampler(
            self.subjects, self.patch_size, self.patch_overlap, padding_mode="edge",
        )
        # need to create aggregator in LesionSegLightning* module, which expects
        # the grid sampler we don't want to send the whole sampler over though,
        # so create a makeshift object with the relevant attributes that duck types
        self.grid_obj = SimpleNamespace(
            subject=SimpleNamespace(spatial_shape=grid_sampler.subject.spatial_shape),
            padding_mode=grid_sampler.padding_mode,
            patch_overlap=grid_sampler.patch_overlap,
        )
        self.predict_dataset = grid_sampler

    def _collate_fn(self, batch: dict) -> Tensor:
        batch = default_collate(batch)
        # p3d is the offset by batch/channel dims if pseudo3d used
        p3d = self.pseudo3d_dim + 2 if self._use_pseudo3d else None
        src = self._default_collate_fn(batch, p3d)
        # assume affine matrices same across modalities
        # so arbitrarily choose first
        field = self._input_fields[0]
        out = dict(
            src=src,
            affine=batch[field][tio.AFFINE],
            out=batch["out"],  # path to save the prediction
            locations=batch[tio.LOCATION],
            grid_obj=self.grid_obj,
            pseudo3d_dim=self.pseudo3d_dim,
            total_batches=self.total_batches,
        )
        return out


class Mixup:
    """mixup for data augmentation

    See Also:
        Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization."
        arXiv preprint arXiv:1710.09412 (2017).

    Args:
        alpha (float): parameter for beta distribution
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def _mixup_dist(self, device: torch.device) -> D.Distribution:
        alpha = torch.tensor(self.alpha, device=device)
        dist = D.Beta(alpha, alpha)
        return dist

    def _mixup_coef(self, batch_size: int, device: torch.device) -> Tensor:
        dist = self._mixup_dist(device)
        return dist.sample((batch_size,))

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


def label_to_float() -> tio.Transform:
    """ cast a label image (usually uint8) to a float """
    return tio.Lambda(_to_float, types_to_apply=[tio.LABEL])


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


def csv_to_subjectlist(filename: str) -> List[tio.Subject]:
    """Convert a csv file to a list of torchio subjects

    Args:
        filename: Path to csv file formatted with
            `subject` in a column, describing the
            id/name of the subject (must be unique).
            Row will fill in the filenames per type.
            Other columns headers must be one of:
            ct, flair, label, pd, t1, t1c, t2, weight, div
            (`label` should correspond to a segmentation mask)
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
