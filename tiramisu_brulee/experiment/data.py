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
    "LesionSegDataModulePredictPatches",
    "LesionSegDataModulePredictWhole",
    "LesionSegDataModuleTrain",
    "Mixup",
]

from logging import getLogger
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from jsonargparse import ArgumentParser
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.distributions as D
import torchio as tio

from tiramisu_brulee.experiment.type import (
    Batch,
    file_path,
    nonnegative_int,
    nonnegative_int_or_none_or_all,
    PatchShapeOption,
    PatchShape,
    positive_float,
    positive_int,
    positive_odd_int_or_none,
    positive_int_or_none,
)
from tiramisu_brulee.experiment.util import reshape_for_broadcasting

RECOGNIZED_NAMES = (
    "ct",
    "flair",
    "pd",
    "t1",
    "t1c",
    "t2",
    "label",
    "weight",
    "div",
    "out",
)

logger = getLogger(__name__)

TrainDataModule = TypeVar("TrainDataModule", bound="LesionSegDataModuleTrain")
PredictDataModule = TypeVar("PredictDataModule", bound="LesionSegDataModulePredictBase")


class LesionSegDataModuleBase(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        patch_size: Optional[PatchShapeOption] = None,
        num_workers: int = 16,
        pseudo3d_dim: Optional[int] = None,
        pseudo3d_size: Optional[int] = None,
        reorient_to_canonical: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pseudo3d_dim = pseudo3d_dim
        self._pseudo3d_dim_internal = (
            pseudo3d_dim if isinstance(pseudo3d_dim, int) else 0
        )
        self.pseudo3d_size = pseudo3d_size
        if self._use_pseudo3d and self.pseudo3d_size is None:
            raise ValueError(
                "If pseudo3d_dim provided, pseudo3d_size must be provided."
            )
        self.patch_size = self._determine_patch_size(patch_size)
        self.reorient_to_canonical = reorient_to_canonical

    def _determine_input(
        self,
        subjects: Union[tio.Subject, List[tio.Subject]],
        other_subjects: Optional[List[tio.Subject]] = None,
    ) -> None:
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
    def _div_image_batch(image_batch: Tensor, div: Tensor) -> Tensor:
        with torch.no_grad():
            image_batch /= reshape_for_broadcasting(div, image_batch.ndim)
        return image_batch

    def _default_collate_fn(
        self, batch: Batch, cat_dim: Optional[int] = None,
    ) -> Tensor:
        if isinstance(batch, list):
            batch = default_collate(batch)
        inputs: List[Tensor] = []
        for field in self._input_fields:
            inputs.append(batch[field][tio.DATA])
        cat_dim_ = cat_dim or 1
        src: Tensor = torch.cat(inputs, dim=cat_dim_)
        if cat_dim is not None:
            # if axis is not None, use pseudo3d images
            src.swapaxes_(1, cat_dim_)
            src.squeeze_()
            if src.ndim == 3:  # batch size of 1
                src.unsqueeze_(0)
            assert src.ndim == 4
        if self._use_div:
            assert isinstance(batch["div"], Tensor)
            div_factor: Tensor = batch["div"]
            src = self._div_image_batch(src, div_factor)
        return src

    @staticmethod
    def _pseudo3d_label(label: Tensor, pseudo3d_dim: int) -> Tensor:
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

    def _determine_patch_size(
        self, patch_size: Optional[PatchShapeOption]
    ) -> Optional[PatchShapeOption]:
        if patch_size is None:
            return None
        patch_size_list = list(patch_size)
        if self._use_pseudo3d and len(patch_size_list) != 2:
            raise ValueError(
                "If using pseudo3d, patch size must contain only 2 values."
            )
        if self._use_pseudo3d:
            patch_size_list.insert(self._pseudo3d_dim_internal, self.pseudo3d_size)
        return tuple(patch_size_list)  # type: ignore

    @property
    def _use_pseudo3d(self) -> bool:
        return self.pseudo3d_dim is not None

    @staticmethod
    def _add_common_arguments(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "-bs",
            "--batch-size",
            type=positive_int(),
            default=1,
            help="training/validation batch size",
        )
        parser.add_argument(
            "-nw",
            "--num-workers",
            type=nonnegative_int(),
            default=16,
            help="number of CPUs to use for loading data",
        )
        parser.add_argument(
            "-rtc",
            "--reorient-to-canonical",
            action="store_true",
            default=False,
            help="reorient inputs images to canonical orientation "
            "(useful when using data from heterogeneous sources "
            "or using pseudo3d_dim == all; otherwise, e.g., the "
            "axis corresponding to left-right in one image might "
            "be anterior-posterior in another.)",
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
            "-nsa",
            "--non-strict-affine",
            dest="strict_affine",
            action="store_false",
            default=True,
            help="if images have different affine matrices, "
            "resample the images to be consistent; avoid using"
            "this by coregistering your images within a subject.",
        )
        parser.add_argument(
            "-cd",
            "--check-dicom",
            action="store_true",
            default=False,
            help="check DICOM images to see if they have uniform "
            "spacing between slices; warn the user if not.",
        )
        return parser


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

    # noinspection PyUnusedLocal
    def __init__(  # type: ignore
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
        reorient_to_canonical: bool = True,
        num_classes: int = 1,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            patch_size=patch_size,
            num_workers=num_workers,
            pseudo3d_dim=pseudo3d_dim,
            pseudo3d_size=pseudo3d_size,
            reorient_to_canonical=reorient_to_canonical,
        )
        self.train_subject_list = train_subject_list
        self.val_subject_list = val_subject_list
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume
        self.label_sampler = label_sampler
        self.spatial_augmentation = spatial_augmentation
        self.num_classes = num_classes

    @classmethod
    def from_csv(  # type: ignore
        cls: Type[TrainDataModule], train_csv: str, valid_csv: str, *args, **kwargs,
    ) -> TrainDataModule:
        strict_affine = kwargs.get("strict_affine", True)
        check_dicom = kwargs.get("check_dicom", False)
        tsl = csv_to_subjectlist(train_csv, strict_affine, check_dicom)
        vsl = csv_to_subjectlist(valid_csv, strict_affine, check_dicom)
        return cls(tsl, vsl, *args, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
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

    def _get_train_augmentation(self) -> Callable:
        transforms = []
        if self.reorient_to_canonical:
            transforms.append(tio.ToCanonical())
        if self.num_classes >= 1:
            transforms.append(image_to_float())
        else:
            msg = f"num_classes must be positive. Got {self.num_classes}."
            raise ValueError(msg)
        if self.spatial_augmentation:
            spatial = tio.OneOf(
                {tio.RandomAffine(): 0.8, tio.RandomElasticDeformation(): 0.2}, p=0.75,
            )
            transforms.insert(1, spatial)
            # noinspection PyTypeChecker
            flip = tio.RandomFlip(axes=("LR",))
            transforms.append(flip)
        if self.pseudo3d_dim == "all":
            transforms.insert(1, RandomTranspose())
            transforms.append(RandomRot90())
        transform: Callable = tio.Compose(transforms)
        return transform

    def _get_train_sampler(self) -> Union[tio.LabelSampler, tio.UniformSampler]:
        if self.label_sampler:
            return tio.LabelSampler(self.patch_size)
        else:
            return tio.UniformSampler(self.patch_size)

    def _setup_train_dataset(self) -> None:
        transform = self._get_train_augmentation()
        subjects_dataset = tio.SubjectsDataset(
            self.train_subject_list, transform=transform,
        )
        self.train_dataset = subjects_dataset

    def _get_val_augmentation(self) -> Callable:
        transforms = []
        if self.reorient_to_canonical:
            transforms.append(tio.ToCanonical())
        if self.num_classes >= 1:
            transforms.append(image_to_float())
        else:
            msg = f"num_classes must be positive. Got {self.num_classes}."
            raise ValueError(msg)
        if not self._use_pseudo3d:
            transforms.insert(1, tio.CropOrPad(self.patch_size))
        if self.pseudo3d_dim == "all":
            transforms.insert(1, RandomTranspose())
        transform: Callable = tio.Compose(transforms)
        return transform

    def _get_val_sampler(self) -> tio.LabelSampler:
        return tio.LabelSampler(self.patch_size)

    def _setup_val_dataset(self) -> None:
        transform = self._get_val_augmentation()
        subjects_dataset = tio.SubjectsDataset(
            self.val_subject_list, transform=transform,
        )
        self.val_dataset = subjects_dataset

    def _collate_fn(self, batch: List[tio.Subject]) -> Tuple[Tensor, Tensor]:
        collated_batch: Batch = default_collate(batch)
        p3d = self._pseudo3d_dim_internal if self._use_pseudo3d else None
        # offset by batch/channel dims if pseudo3d used
        p3d_with_offset = p3d + 2 if self._use_pseudo3d else None  # type: ignore
        src = self._default_collate_fn(collated_batch, p3d_with_offset)
        tgt = collated_batch["label"][tio.DATA]
        if self._use_pseudo3d:
            tgt = self._pseudo3d_label(tgt, self._pseudo3d_dim_internal)
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
            "-ps",
            "--patch-size",
            type=positive_int(),
            nargs="+",
            default=[96, 96, 96],
            help="training/validation patch size extracted from image",
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
        LesionSegDataModuleBase._add_common_arguments(parser)
        return parent_parser


class WholeImagePredictBatch:
    def __init__(
        self,
        src: Tensor,
        affine: Tensor,
        path: List[str],
        out: List[str],
        reorient: bool,
    ):
        self.src = src
        self.affine = affine
        self.path = path
        self.out = out
        self.reorient = reorient
        self.validate()

    def validate(self) -> None:
        assert len(self.src) == len(self.affine)
        assert len(self.affine) == len(self.path)
        assert len(self.path) == len(self.out)
        assert all([Path(path).is_file() for path in self.path])
        assert isinstance(self.reorient, bool)


class PatchesImagePredictBatch:
    def __init__(
        self,
        src: Tensor,
        affine: Tensor,
        path: str,
        out: str,
        locations: Tensor,
        grid_obj: SimpleNamespace,
        pseudo3d_dim: Optional[int],
        total_batches: int,
        reorient: bool,
    ):
        self.src = src
        self.affine = affine
        self.path = path
        self.out = out
        self.locations = locations
        self.grid_obj = grid_obj
        self.pseudo3d_dim = pseudo3d_dim
        self.total_batches = total_batches
        self.reorient = reorient
        self.validate()

    def validate(self) -> None:
        assert len(self.src) == len(self.affine)
        assert Path(self.path).is_file()
        assert self.pseudo3d_dim is None or (0 <= self.pseudo3d_dim <= 2)
        assert self.total_batches >= 1
        assert hasattr(self.grid_obj, "subject")
        assert hasattr(self.grid_obj, "padding_mode")
        assert hasattr(self.grid_obj, "patch_overlap")
        assert hasattr(self.grid_obj.subject, "spatial_shape")
        assert isinstance(self.reorient, bool)


class LesionSegDataModulePredictBase(LesionSegDataModuleBase):
    def __init__(  # type: ignore
        self,
        subjects: Union[tio.Subject, List[tio.Subject]],
        batch_size: int,
        patch_size: Optional[PatchShapeOption] = None,
        num_workers: int = 16,
        pseudo3d_dim: Optional[int] = None,
        pseudo3d_size: Optional[int] = None,
        reorient_to_canonical: bool = True,
        **kwargs,
    ):
        super().__init__(
            batch_size,
            patch_size,
            num_workers,
            pseudo3d_dim,
            pseudo3d_size,
            reorient_to_canonical,
        )
        self.predict_dataset: tio.SubjectsDataset
        self.subjects = subjects

    def setup(self, stage: Optional[str] = None) -> None:
        self._determine_input(self.subjects)
        self._setup_predict_dataset()

    def predict_dataloader(self) -> DataLoader:
        pred_dataloader: DataLoader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
        self.total_batches = len(pred_dataloader)
        return pred_dataloader

    def _setup_predict_dataset(self) -> None:
        raise NotImplementedError

    def _collate_fn(
        self, batch: List[tio.Subject]
    ) -> Union[WholeImagePredictBatch, PatchesImagePredictBatch]:
        raise NotImplementedError

    @staticmethod
    def add_arguments(
        parent_parser: ArgumentParser, add_csv: bool = True,
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
        LesionSegDataModuleBase._add_common_arguments(parser)
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

    # noinspection PyUnusedLocal
    def __init__(  # type: ignore
        self,
        subjects: Union[tio.Subject, List[tio.Subject]],
        batch_size: int,
        num_workers: int = 16,
        reorient_to_canonical: bool = True,
        **kwargs,
    ):
        super().__init__(
            subjects=subjects,
            batch_size=batch_size,
            patch_size=None,
            num_workers=num_workers,
            pseudo3d_dim=None,
            pseudo3d_size=None,
            reorient_to_canonical=reorient_to_canonical,
        )

    @classmethod
    def from_csv(  # type: ignore
        cls: Type[PredictDataModule], predict_csv: str, *args, **kwargs,
    ) -> PredictDataModule:
        strict_affine = kwargs.get("strict_affine", True)
        check_dicom = kwargs.get("check_dicom", False)
        subject_list = csv_to_subjectlist(predict_csv, strict_affine, check_dicom)
        return cls(subject_list, *args, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        self._determine_input(self.subjects)
        self._setup_predict_dataset()

    def _setup_predict_dataset(self) -> None:
        transforms = []
        if self.reorient_to_canonical:
            transforms.append(tio.ToCanonical())
        transforms.append(image_to_float())
        transform = tio.Compose(transforms)
        subjects_dataset = tio.SubjectsDataset(self.subjects, transform=transform)
        self.predict_dataset = subjects_dataset

    def _collate_fn(self, batch: List[tio.Subject]) -> WholeImagePredictBatch:
        collated_batch: Batch = default_collate(batch)
        src: Tensor = self._default_collate_fn(collated_batch)
        # assume all input images are co-registered
        # so arbitrarily choose first
        field: str = self._input_fields[0]
        first_field = collated_batch[field]
        assert isinstance(first_field, dict)
        affine: Tensor = first_field[tio.AFFINE]
        path: List[str] = [str(filepath) for filepath in first_field["path"]]
        out_path = collated_batch["out"]  # path to save the prediction
        assert isinstance(out_path, list)
        out = WholeImagePredictBatch(
            src=src,
            affine=affine,
            path=path,
            out=out_path,
            reorient=self.reorient_to_canonical,
        )
        return out


class LesionSegDataModulePredictPatches(LesionSegDataModulePredictBase):
    """Data module for patch-based prediction for lesion segmentation

    Args:
        subject (tio.Subject):
            a torchio.Subject for prediction
        batch_size (int): number of patches to predict at a time
        patch_size (PatchShapeOption): patch size for training/validation
            if any element is None, use the corresponding image dim
        patch_overlap (Optional[PatchShape]):
            overlap of each patch, if None then patch_size // 2
        num_workers (int):
            number of subprocesses to use for data loading
        pseudo3d_dim (Optional[int]): concatenate images along this
            axis and swap it for channel dimension
        pseudo3d_size (Optional[int]): number of slices to concatenate
            if pseudo3d_dim provided, must be an odd (usually small) integer
    """

    # noinspection PyUnusedLocal
    def __init__(  # type: ignore
        self,
        subject: tio.Subject,
        batch_size: int = 1,
        patch_size: PatchShapeOption = (96, 96, 96),
        patch_overlap: Optional[PatchShape] = None,
        num_workers: int = 16,
        pseudo3d_dim: Optional[int] = None,
        pseudo3d_size: Optional[int] = None,
        reorient_to_canonical: bool = True,
        **kwargs,
    ):
        super().__init__(
            subjects=subject,
            batch_size=batch_size,
            patch_size=patch_size,
            num_workers=num_workers,
            pseudo3d_dim=pseudo3d_dim,
            pseudo3d_size=pseudo3d_size,
            reorient_to_canonical=reorient_to_canonical,
        )
        assert self.patch_size is not None
        # self.patch_size is the result from _determine_patch_size
        ps: PatchShapeOption = self.patch_size
        self._set_patch_size(subject, ps)
        self.patch_overlap: PatchShape = patch_overlap or self._default_overlap(ps)

    def _set_patch_size(
        self, subject: tio.Subject, patch_size: PatchShapeOption,
    ) -> None:
        if len(patch_size) != 3:
            raise ValueError(
                "Patch size must have length 3 here. "
                f"Got {len(patch_size)}. Something went wrong."
            )
        image_dim = subject.spatial_shape
        if len(image_dim) != 3:
            raise ValueError(
                "Input image must be three-dimensional. "
                f"Got image dim of {len(image_dim)}."
            )
        patch_size_no_none = [ps or dim for ps, dim in zip(patch_size, image_dim)]
        ps_x, ps_y, ps_z = patch_size_no_none
        self.patch_size = (ps_x, ps_y, ps_z)

    def _default_overlap(self, patch_size: PatchShapeOption) -> PatchShape:
        patch_overlap_list = []
        for i, ps in enumerate(patch_size):
            if ps is None:
                patch_overlap_list.append(0)
                continue
            if i == self.pseudo3d_dim:
                assert self.pseudo3d_size is not None
                patch_overlap_list.append(self.pseudo3d_size - 1)
                continue
            overlap = ps // 2
            if overlap % 2:
                overlap += 1
            patch_overlap_list.append(overlap)
        patch_overlap: PatchShape
        if len(patch_size) == 2:
            po_x, po_y = patch_overlap_list
            patch_overlap = (po_x, po_y)
        elif len(patch_size) == 3:
            po_x, po_y, po_z = patch_overlap_list
            patch_overlap = (po_x, po_y, po_z)
        else:
            raise ValueError(
                f"patch_size must have length 2 or 3. Got {len(patch_size)}."
            )
        return patch_overlap

    def _setup_predict_dataset(self) -> None:
        field = self._input_fields[0]
        self.path: str = self.subjects[field]["path"]
        # `subjects` is only one subject in this class
        if self.reorient_to_canonical:
            self.subjects = tio.ToCanonical()(self.subjects)
        self.subjects = image_to_float()(self.subjects)
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

    def _collate_fn(self, batch: List[tio.Subject]) -> PatchesImagePredictBatch:
        collated_batch = default_collate(batch)
        p3d = self._pseudo3d_dim_internal if self._use_pseudo3d else None
        # offset by batch/channel dims if pseudo3d used
        p3d_with_offset = p3d + 2 if self._use_pseudo3d else None  # type: ignore
        src: Tensor = self._default_collate_fn(collated_batch, p3d_with_offset)
        # assume input images are co-registered
        # so arbitrarily choose first
        field: str = self._input_fields[0]
        affine: Tensor = collated_batch[field][tio.AFFINE]
        out_path: str = collated_batch["out"]  # path to save the prediction
        locations: Tensor = collated_batch[tio.LOCATION]
        out = PatchesImagePredictBatch(
            src=src,
            affine=affine,
            path=self.path,
            out=out_path,  # path to save the prediction
            locations=locations,
            grid_obj=self.grid_obj,
            pseudo3d_dim=p3d,
            total_batches=self.total_batches,
            reorient=self.reorient_to_canonical,
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
        lam: Tensor = dist.sample((batch_size,))  # convex combination coef
        return lam

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


def image_to_float() -> tio.Transform:
    """ cast an image from any type (e.g., uint8 or float64) to float32 """
    return tio.Lambda(_to_float, types_to_apply=[tio.INTENSITY, tio.LABEL])


class RandomTranspose(
    tio.transforms.augmentation.RandomTransform, tio.SpatialTransform,
):

    transposes = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        index = self.get_params()
        for image in self.get_images(subject):
            data = image.data.permute(*self.transposes[index])
            image.set_data(data)
        return subject

    @staticmethod
    def get_params() -> int:
        dim = int(torch.randint(0, 3, (1,)).item())
        return dim


class RandomRot90(
    tio.transforms.augmentation.RandomTransform, tio.SpatialTransform,
):
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        k = self.get_params()
        for image in self.get_images(subject):
            assert image.data.ndim == 4
            data = image.data.rot90(k, (-2, -1))
            image.set_data(data)
        return subject

    @staticmethod
    def get_params() -> int:
        n_rot = int(torch.randint(0, 4, size=(1,)).item())
        return n_rot


def _get_type(name: str) -> str:
    name_lower = name.lower()
    _type: str
    if name_lower == "label":
        _type = tio.LABEL
    elif name_lower in ("weight", "div"):
        _type = "float"
    elif name_lower == "out":
        _type = "path"
    elif name_lower in RECOGNIZED_NAMES:
        _type = tio.INTENSITY
    else:
        logger.warning(
            f"{name} not in known {RECOGNIZED_NAMES}. "
            f"Assuming {name} is a non-label image."
        )
        _type = tio.INTENSITY
    return _type


def csv_to_subjectlist(
    filename: str, strict: bool = True, check_dicom: bool = False,
) -> List[tio.Subject]:
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
        strict: if affine matrices are different enough
            (according to torchio tolerance), raise a
            runtime error. Otherwise, resample the images
            of the subject to the first image.
        check_dicom: if true, check dicom images for uniform
            spacing and warn the user about image if there
            is serious non-uniformity in slice distances
    Returns:
        subject_list (List[torchio.Subject]): list of torchio Subjects
    """
    df = pd.read_csv(filename, index_col="subject")
    names = df.columns.to_list()
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
                image_path = Path(val)
                if image_path.is_dir() and check_dicom:
                    _check_spacing_between_dicom_slices(image_path, strict)
                data[name] = tio.Image(image_path, type=val_type)
        subject = tio.Subject(name=subject_name, **data)
        subject = _check_consistent_space_and_resample(subject, strict)
        subject_list.append(subject)
    return subject_list


def _check_consistent_space_and_resample(
    subject: tio.Subject, strict: bool = True,
) -> tio.Subject:
    """Check space of images in subject consistent; if not strict, resample."""
    # spatial shape always needs to be the same
    subject.check_consistent_spatial_shape()
    if strict:
        subject.check_consistent_affine()
    else:
        default_printoptions = np.get_printoptions()
        np.set_printoptions(precision=5, suppress=True)
        try:
            subject.check_consistent_affine()
        except RuntimeError as e:
            logger.warning(f"{subject['name']} has inconsistent affine matrices.")
            logger.info(e)
            logger.info("Attempting to resample the images to be consistent.")
            affine = None
            first_image = None
            first_image_name = None
            iterable = subject.get_images_dict(intensity_only=False).items()
            for image_name, image in iterable:
                if affine is None:
                    affine = image.affine
                    first_image = image
                    first_image_name = image_name
                elif not np.allclose(affine, image.affine, rtol=1e-6, atol=1e-6):
                    aff_mtx_dist = np.linalg.norm(affine - image.affine)
                    logger.info(
                        f"Frobenius dist. between {first_image_name} and {image_name} "
                        f"affine matrices {aff_mtx_dist:0.4e}"
                    )
                    if aff_mtx_dist >= 1e-4:
                        msg = (
                            "Distance between affine matrices is large. "
                            "Consider aborting and registering the images manually."
                        )
                        logger.warning(msg)
                    if image.type == tio.LABEL:
                        resampler = tio.Resample(first_image, "nearest")
                    else:
                        resampler = tio.Resample(first_image)
                    resampled = resampler(image)
                    subject[image_name] = resampled
        np.set_printoptions(**default_printoptions)
    return subject


def _check_spacing_between_dicom_slices(
    dicom_dir: Union[Path, str], strict: bool = True,
) -> None:
    try:
        import pydicom  # type: ignore[import]
    except (ImportError, ModuleNotFoundError):
        logger.warning("pydicom not found. Cannot validate DICOM image.")
        return
    images = [pydicom.dcmread(path) for path in Path(dicom_dir).glob("*.dcm")]
    slice_thickness = float(images[0].SliceThickness)

    def get_stack_position(image: pydicom.dataset.Dataset) -> int:
        stack_position: int = image.InStackPositionNumber
        return stack_position

    sorted_images = sorted(images, key=get_stack_position)
    positions = np.array([img.ImagePositionPatient for img in sorted_images])
    space_between_positions = np.diff(positions, axis=0)
    dist_between_slices = np.linalg.norm(space_between_positions, axis=1)
    diff_in_dist = np.abs(np.diff(dist_between_slices))
    median_dist_between_slices = np.median(dist_between_slices)
    slice_thickness_msg = ""
    if not np.isclose(slice_thickness, median_dist_between_slices):
        slice_thickness_msg = (
            f"Slice thickness: {slice_thickness:0.6f} != "
            f"(Median) computed slice thickness {median_dist_between_slices:0.6f}"
        )
    max_diff_in_dist = diff_in_dist.max()
    inconsistent_dist_msg = ""
    if max_diff_in_dist > 5e-4:
        # TODO: why is max_diff_in_dist different from ITK "Maximum nonuniformity"
        inconsistent_dist_msg = (
            "Maximum difference in distance between slices: "
            f"{max_diff_in_dist:0.5e}."
        )
    if slice_thickness_msg or inconsistent_dist_msg:
        msg = (
            (slice_thickness_msg + "\n" + inconsistent_dist_msg)
            if slice_thickness_msg and inconsistent_dist_msg
            else (slice_thickness_msg or inconsistent_dist_msg)
        )
        if strict:
            raise RuntimeError(msg)
        else:
            logger.warning(msg)
