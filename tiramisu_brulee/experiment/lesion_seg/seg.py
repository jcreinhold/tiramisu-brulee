#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.lesion_seg.seg

3D Tiramisu network for lesion segmentation

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 14, 2021
"""

__all__ = [
    "LesionSegLightningTiramisu",
]

from collections import namedtuple
from functools import partial
import logging
from typing import List, Optional, Tuple

from jsonargparse import ArgumentParser
import nibabel as nib
import numpy as np

import torch
from torch import Tensor
from torch.optim import AdamW, RMSprop, lr_scheduler

from tiramisu_brulee.experiment.lightningtiramisu import LightningTiramisu
from tiramisu_brulee.experiment.lesion_seg.data import Mixup
from tiramisu_brulee.experiment.lesion_seg.lesion_tools import (
    almost_isbi15_score,
    clean_segmentation,
)
from tiramisu_brulee.experiment.lesion_seg.parse import (
    nonnegative_int,
    positive_float,
    positive_int,
    probability_float,
)
from tiramisu_brulee.experiment.lesion_seg.util import (
    append_num_to_filename,
    BoundingBox3D,
    minmax_scale_batch,
    to_np,
)
from tiramisu_brulee.loss import (
    binary_combo_loss,
    l1_segmentation_loss,
    mse_segmentation_loss,
)

ModelNum = namedtuple("ModelNum", ["num", "out_of"])


class LesionSegLightningTiramisu(LightningTiramisu):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        down_blocks: List[int] = (4, 4, 4, 4, 4),
        up_blocks: List[int] = (4, 4, 4, 4, 4),
        bottleneck_layers: int = 4,
        growth_rate: int = 16,
        first_conv_out_channels: int = 48,
        dropout_rate: float = 0.2,
        init_type: str = "normal",
        gain: float = 0.02,
        n_epochs: int = 1,
        learning_rate: float = 1e-3,
        betas: Tuple[int, int] = (0.9, 0.99),
        weight_decay: float = 1e-7,
        combo_weight: float = 0.6,
        decay_after: int = 8,
        loss_function: str = "combo",
        rmsprop: bool = False,
        soft_labels: bool = False,
        threshold: float = 0.5,
        min_lesion_size: int = 3,
        fill_holes: bool = True,
        predict_probability: bool = False,
        mixup: bool = False,
        mixup_alpha: float = 0.4,
        _model_num: ModelNum = ModelNum(1, 1),
        **kwargs,
    ):
        """
        3D Tiramisu-based PyTorch-Lightning module for lesion segmentation [1,2]

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            down_blocks (List[int]): number of layers in each block in down path
            up_blocks (List[int]): number of layers in each block in up path
            bottleneck_layers (int): number of layers in the bottleneck
            growth_rate (int): number of channels to grow by in each layer
            first_conv_out_channels (int): number of output channels in first conv
            dropout_rate (float): dropout rate/probability
            init_type (str): method to initialize the weights of network
            gain (float): gain parameter for initialization
            n_epochs (int): number of epochs to train the network
            learning_rate (float): learning rate for the optimizer
            betas (Tuple[float, float]): momentum parameters for adam
            weight_decay (float): weight decay for optimizer
            combo_weight (float): weight by which to balance BCE and Dice loss
            decay_after (int): decay learning rate linearly after this many epochs
            loss_function (str): loss function to use in training
            rmsprop (bool): use rmsprop instead of adamw
            soft_labels (bool): use non-binary labels for training
            threshold (float): threshold by which to decide on positive class
            min_lesion_size (int): minimum lesion size in voxels in output prediction
            fill_holes (bool): use binary fill holes operation on label
            predict_probability (bool): save a probability image instead of a binary one
            mixup (bool): use mixup in training
            mixup_alpha (float): mixup parameter for beta distribution
            _model_num (ModelNum): internal param for ith of n models

        References:
            [1] Jégou, Simon, et al. "The one hundred layers tiramisu:
                Fully convolutional densenets for semantic segmentation."
                CVPR. 2017.
            [2] Zhang, Huahong, et al. "Multiple sclerosis lesion segmentation
                with Tiramisu and 2.5D stacked slices." International Conference
                on Medical Image Computing and Computer-Assisted Intervention.
                Springer, Cham, 2019.
        """
        network_dim = 3  # only support 3D input
        super().__init__(
            network_dim,
            in_channels,
            out_channels,
            down_blocks,
            up_blocks,
            bottleneck_layers,
            growth_rate,
            first_conv_out_channels,
            dropout_rate,
            init_type,
            gain,
            n_epochs,
            learning_rate,
            betas,
            weight_decay,
        )
        self._model_num = _model_num
        self.save_hyperparameters(ignore="_model_num")

    def setup(self, stage: Optional[str] = None):
        if self.hparams.loss_function == "combo":
            self.criterion = partial(
                binary_combo_loss, combo_weight=self.hparams.combo_weight
            )
        elif self.hparams.loss_function == "mse":
            self.criterion = mse_segmentation_loss
        elif self.hparams.loss_function == "l1":
            self.criterion = l1_segmentation_loss
        else:
            raise ValueError(f"{self.hparams.loss_function} not supported.")
        if self.hparams.mixup:
            self._mix = Mixup(self.hparams.mixup_alpha)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        src, tgt = batch
        if self.hparams.mixup:
            src, tgt = self._mix(src, tgt)
        pred = self(src)
        loss = self.criterion(pred, tgt)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        src, tgt = batch
        pred = self(src)
        loss = self.criterion(pred, tgt)
        with torch.no_grad():
            pred_seg = torch.sigmoid(pred) > self.hparams.threshold
        isbi15_score = almost_isbi15_score(pred_seg, tgt)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
        )
        self.log(
            "val_isbi15_score",
            isbi15_score,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        if batch_idx == 0:
            images = dict(truth=tgt, pred=pred,)
            for i in range(src.shape[1]):
                images[f"input_channel_{i}"] = src[:, i : i + 1, ...]
        else:
            images = None
        return dict(val_loss=loss, images=images)

    def validation_epoch_end(self, outputs: list):
        self._log_images(outputs[0]["images"])

    def predict_step(
        self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tensor:
        src = batch["src"]
        bbox = BoundingBox3D.from_batch(src, pad=0)
        src = bbox(src)
        pred = self(src)
        pred_seg = torch.sigmoid(pred)
        if not self.hparams.predict_probability:
            pred_seg = pred_seg > self.hparams.threshold
        pred_seg = bbox.uncrop_batch(pred_seg)
        pred_seg = pred_seg.float()
        return pred_seg

    def on_predict_batch_end(
        self,
        pred_step_outputs: Tensor,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int,
    ):
        data = to_np(pred_step_outputs).squeeze()
        if data.ndim == 5:
            raise ValueError("Predictions should only have one channel.")
        elif data.ndim == 3:
            data = data[None, ...]
        if not self.hparams.predict_probability:
            data = [clean_segmentation(seg) for seg in data]
        data = [seg.astype(np.float32) for seg in data]
        affine_matrices = batch["affine"]
        out_fns = batch["out"]
        nifti_attrs = zip(data, affine_matrices, out_fns)
        for data, affine, fn in nifti_attrs:
            if self._model_num != ModelNum(num=1, out_of=1):
                fn = append_num_to_filename(fn, self._model_num.num)
            logging.info(f"Saving {fn}.")
            nib.Nifti1Image(data, affine).to_filename(fn)

    def configure_optimizers(self):
        if self.hparams.rmsprop:
            momentum, alpha = self.hparams.betas
            optimizer = RMSprop(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=momentum,
                alpha=alpha,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                weight_decay=self.hparams.weight_decay,
            )

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.decay_rule)
        return [optimizer], [scheduler]

    def decay_rule(self, epoch: int) -> float:
        numerator = max(0, epoch - self.hparams.decay_after)
        denominator = float(self.hparams.n_epochs + 1)
        lr = 1.0 - numerator / denominator
        return lr

    def _log_images(self, images: dict):
        n = self.current_epoch
        mid_slice = None
        for key, image in images.items():
            with torch.no_grad():
                if mid_slice is None:
                    mid_slice = image.shape[-1] // 2
                image_slice = image[..., mid_slice]
                if self.hparams.soft_labels and key == "pred":
                    image_slice = torch.sigmoid(image_slice)
                elif key == "pred":
                    threshold = self.hparams.threshold
                    image_slice = torch.sigmoid(image_slice) > threshold
                elif key == "truth":
                    image_slice = image_slice > 0.0
                else:
                    image_slice = minmax_scale_batch(image_slice)
            self.logger.experiment.add_images(key, image_slice, n, dataformats="NCHW")

    @staticmethod
    def add_model_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument(
            "-ic",
            "--in-channels",
            type=positive_int(),
            default=1,
            help="number of input channels (should match the number "
            "of non-label/other fields in the input csv)",
        )
        parser.add_argument(
            "-oc",
            "--out-channels",
            type=positive_int(),
            default=1,
            help="number of output channels (usually 1 for segmentation)",
        )
        parser.add_argument(
            "-dr",
            "--dropout-rate",
            type=positive_float(),
            default=0.2,
            help="dropout rate/probability",
        )
        parser.add_argument(
            "-it",
            "--init-type",
            type=str,
            default="he_uniform",
            choices=(
                "normal",
                "xavier_normal",
                "he_normal",
                "he_uniform",
                "orthogonal",
            ),
            help="use this type of initialization for the network",
        )
        parser.add_argument(
            "-ig",
            "--init-gain",
            type=positive_float(),
            default=0.2,
            help="use this initialization gain for initialization",
        )
        parser.add_argument(
            "-db",
            "--down-blocks",
            type=positive_int(),
            default=[4, 4, 4, 4, 4],
            nargs="+",
            help="tiramisu down-sample path specification",
        )
        parser.add_argument(
            "-ub",
            "--up-blocks",
            type=positive_int(),
            default=[4, 4, 4, 4, 4],
            nargs="+",
            help="tiramisu up-sample path specification",
        )
        parser.add_argument(
            "-bl",
            "--bottleneck-layers",
            type=positive_int(),
            default=4,
            help="tiramisu bottleneck specification",
        )
        parser.add_argument(
            "-gr",
            "--growth-rate",
            type=positive_int(),
            default=12,
            help="tiramisu growth rate (number of channels "
            "added between each layer in a dense block)",
        )
        parser.add_argument(
            "-fcoc",
            "--first-conv-out-channels",
            type=positive_int(),
            default=48,
            help="number of output channels in first conv",
        )
        return parent_parser

    @staticmethod
    def add_training_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Training")
        parser.add_argument(
            "-bt",
            "--betas",
            type=positive_float(),
            default=[0.9, 0.99],
            nargs=2,
            help="AdamW momentum parameters (for RMSprop, momentum and alpha)",
        )
        parser.add_argument(
            "-cen",
            "--checkpoint-every-n-epochs",
            type=positive_int(),
            default=1,
            help="save model weights (checkpoint) every n epochs",
        )
        parser.add_argument(
            "-cw",
            "--combo-weight",
            type=positive_float(),
            default=0.6,
            help="weight of positive class in combo loss",
        )
        parser.add_argument(
            "-da",
            "--decay-after",
            type=positive_int(),
            default=8,
            help="decay learning rate after this number of epochs",
        )
        parser.add_argument(
            "-lr",
            "--learning-rate",
            type=positive_float(),
            default=3e-4,
            help="learning rate for the optimizer",
        )
        parser.add_argument(
            "-lf",
            "--loss-function",
            type=str,
            default="combo",
            choices=("combo", "l1", "mse"),
            help="loss function to train the network",
        )
        parser.add_argument(
            "-ne",
            "--n-epochs",
            type=positive_int(),
            default=64,
            help="number of epochs",
        )
        parser.add_argument(
            "-rp",
            "--rmsprop",
            action="store_true",
            default=False,
            help="use rmsprop instead of adam",
        )
        parser.add_argument(
            "-wd",
            "--weight-decay",
            type=positive_float(),
            default=1e-5,
            help="weight decay parameter for adamw",
        )
        parser.add_argument(
            "-sl",
            "--soft-labels",
            action="store_true",
            default=False,
            help="use soft labels (i.e., non-binary labels) for training",
        )
        return parent_parser

    @staticmethod
    def add_other_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Other")
        parser.add_argument(
            "-th",
            "--threshold",
            type=probability_float(),
            default=0.5,
            help="probability threshold for segmentation",
        )
        return parent_parser

    @staticmethod
    def add_testing_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Testing")
        parser.add_argument(
            "-mls",
            "--min-lesion-size",
            type=nonnegative_int(),
            default=3,
            help="in testing, remove lesions smaller in voxels than this",
        )
        parser.add_argument(
            "-fh",
            "--fill-holes",
            action="store_true",
            default=False,
            help="in testing, preform binary hole filling",
        )
        parser.add_argument(
            "-pp",
            "--predict-probability",
            action="store_true",
            default=False,
            help="in testing, store the probability instead of the binary prediction",
        )
        return parent_parser
