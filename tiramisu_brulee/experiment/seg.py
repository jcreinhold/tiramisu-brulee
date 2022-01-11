"""Training and prediction lightning modules

Training and prediction logic for segmentation
(usually lesion segmentation). Also, an
implementation of the Tiramisu network with
the training and prediction logic built-in.

Author: Jacob Reinhold <jcreinhold@gmail.com>
Created on: May 14, 2021
"""

__all__ = [
    "LesionSegLightningBase",
    "LesionSegLightningTiramisu",
]

import builtins
import enum
import functools
import logging
import typing
import warnings

import numpy as np
import pytorch_lightning as pl
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchio as tio
from pytorch_lightning.utilities import AttributeDict
from torch.optim import AdamW, Optimizer, RMSprop
from torch.optim.lr_scheduler import LambdaLR

from tiramisu_brulee.experiment.data import (
    Mixup,
    PatchesImagePredictBatch,
    WholeImagePredictBatch,
)
from tiramisu_brulee.experiment.lesion_tools import (
    almost_isbi15_score,
    clean_segmentation,
)
from tiramisu_brulee.experiment.type import (
    ArgParser,
    ModelNum,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_float_or_none,
    positive_int,
    probability_float,
)
from tiramisu_brulee.experiment.util import (
    BoundingBox3D,
    append_num_to_filename,
    minmax_scale_batch,
)
from tiramisu_brulee.loss import (
    binary_combo_loss,
    combo_loss,
    l1_segmentation_loss,
    mse_segmentation_loss,
)
from tiramisu_brulee.model import ResizeMethod, Tiramisu2d, Tiramisu3d
from tiramisu_brulee.util import InitType, init_weights

PredictBatch = typing.Union[PatchesImagePredictBatch, WholeImagePredictBatch]


@enum.unique
class LossFunction(enum.Enum):
    COMBO: builtins.str = "combo"
    L1: builtins.str = "l1"
    MSE: builtins.str = "mse"

    @classmethod
    def from_string(cls, string: builtins.str) -> "LossFunction":
        if string.lower() == "combo":
            return cls.COMBO
        elif string.lower() == "l1":
            return cls.L1
        elif string.lower() == "mse":
            return cls.MSE
        else:
            msg = f"Only 'combo', 'l1', 'mse' allowed. Got {string}"
            raise ValueError(msg)


class LesionSegLightningBase(pl.LightningModule):
    """PyTorch-Lightning module for lesion segmentation

    Includes framework for both training and prediction,
    just drop in a PyTorch neural network module

    Args:
        network (nn.Module): PyTorch neural network
        n_epochs (int): number of epochs to train the network
        learning_rate (float): learning rate for the optimizer
        betas (typing.Tuple[float, float]): momentum parameters for adam
        weight_decay (float): weight decay for optimizer
        loss_function (str): loss function to use in training
        pos_weight (typing.Optional[float]): weight for positive class
            in focal/bce loss if using combo loss function
        focal_gamma (float): gamma param for focal loss
            if using combo loss function (0. -> BCE)
        combo_weight (float): weight by which to balance focal and Dice
            losses in combo loss function
        decay_after (int): decay learning rate linearly after this many epochs
        rmsprop (bool): use rmsprop instead of adamw
        soft_labels (bool): use non-binary labels for training
        threshold (float): threshold by which to decide on positive class
        min_lesion_size (int): minimum lesion size in voxels in output prediction
        fill_holes (bool): use binary fill holes operation on label
        predict_probability (bool): save a probability image instead of a binary one
        mixup (bool): use mixup in training
        mixup_alpha (float): mixup parameter for beta distribution
        num_input (int): number of different images input to the network,
            differs from in_channels when using pseudo3d
        num_classes (int): number of different images output by the network
            differs from out_channels when using pseudo3d
        _model_num (ModelNum): internal param for ith of n models
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        *,
        network: nn.Module,
        n_epochs: builtins.int = 1,
        learning_rate: builtins.float = 1e-3,
        betas: typing.Tuple[builtins.float, builtins.float] = (0.9, 0.99),
        weight_decay: builtins.float = 1e-7,
        loss_function: builtins.str = "combo",
        pos_weight: typing.Optional[builtins.float] = None,
        focal_gamma: builtins.float = 0.0,
        combo_weight: builtins.float = 0.6,
        decay_after: builtins.int = 8,
        rmsprop: bool = False,
        soft_labels: builtins.bool = False,
        threshold: builtins.float = 0.5,
        min_lesion_size: builtins.int = 3,
        fill_holes: builtins.bool = True,
        predict_probability: builtins.bool = False,
        mixup: builtins.bool = False,
        mixup_alpha: builtins.float = 0.4,
        num_input: builtins.int = 1,
        num_classes: builtins.int = 1,
        _model_num: ModelNum = ModelNum(1, 1),
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self._model_num = _model_num
        self.save_hyperparameters(ignore=["network", "_model_num"])
        # noinspection PyPropertyAccess
        self.hparams: AttributeDict

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out: torch.Tensor = self.network(tensor)
        return out

    def setup(self, stage: typing.Optional[builtins.str] = None) -> None:
        if self.hparams.loss_function != "combo" and self.hparams.num_classes != 1:
            raise ValueError("Only combo loss supported for multi-class segmentation")
        self.criterion: typing.Callable
        num_classes = self.hparams.num_classes
        assert isinstance(num_classes, builtins.int)
        loss_func_str = self.hparams.loss_function
        assert isinstance(loss_func_str, builtins.str)
        loss_func = LossFunction.from_string(loss_func_str)
        if loss_func == LossFunction.COMBO:
            if self.hparams.num_classes == 1:
                self.criterion = functools.partial(
                    binary_combo_loss,
                    pos_weight=self.hparams.pos_weight,
                    focal_gamma=self.hparams.focal_gamma,
                    combo_weight=self.hparams.combo_weight,
                )
            elif num_classes > 1:
                self.criterion = functools.partial(
                    combo_loss,
                    num_classes=self.hparams.num_classes,
                    combo_weight=self.hparams.combo_weight,
                )
            else:
                msg = f"num_classes must be greater than zero. Got {self.num_classes}."
                raise ValueError(msg)
        elif loss_func == LossFunction.L1:
            self.criterion = l1_segmentation_loss
        elif loss_func == LossFunction.MSE:
            self.criterion = mse_segmentation_loss
        else:
            raise ValueError(f"{self.hparams.loss_function} not supported.")
        use_mixup = bool(self.hparams.mixup)
        if use_mixup:
            mixup_alpha = self.hparams.mixup_alpha
            assert isinstance(mixup_alpha, builtins.float)
            self._mix = Mixup(mixup_alpha)

    def training_step(  # type: ignore[override]
        self,
        batch: typing.Tuple[torch.Tensor, torch.Tensor],
        batch_idx: builtins.int,
    ) -> torch.Tensor:
        src, tgt = batch
        if self.hparams.mixup:
            src, tgt = self._mix(src, tgt)
        pred = self(src)
        loss: torch.Tensor = self.criterion(pred, tgt)
        self.log("loss", loss)
        return loss

    def validation_step(  # type: ignore[override]
        self,
        batch: typing.Tuple[torch.Tensor, torch.Tensor],
        batch_idx: builtins.int,
    ) -> typing.Dict[builtins.str, typing.Any]:
        src, tgt = batch
        pred = self(src)
        loss = self.criterion(pred, tgt)
        pred_seg = torch.sigmoid(pred) > self.hparams.threshold
        isbi15_score, dice, ppv = almost_isbi15_score(
            pred_seg, tgt, return_dice_ppv=True
        )
        num_input = self.hparams.num_input
        assert isinstance(num_input, builtins.int)
        logging.debug(
            f"ISBI15: {isbi15_score.item():0.3f}; "
            f"Dice: {dice.item():0.3f}; "
            f"PPV: {ppv.item():0.3f}; "
            f"Loss: {loss.item():0.3f}."
        )
        images: typing.Optional[
            typing.Dict[builtins.str, typing.Union[builtins.int, torch.Tensor]]
        ]
        if batch_idx == 0 and self._is_3d_image_batch(src):
            images = dict(truth=tgt, pred=pred, dim=3)
            for i in range(src.shape[1]):
                images[f"input_channel_{i}"] = src[:, i : i + 1, ...]
        elif batch_idx == 0 and self._is_2d_image_batch(src):
            images = dict(truth=tgt, pred=pred, dim=2)
            step = src.shape[1] // num_input
            start = step // 2
            end = src.shape[1]
            for i in range(start, end, step):
                images[f"input_channel_{i}"] = src[:, i : i + 1, ...]
        else:
            images = None
        return dict(
            loss=loss, isbi15_score=isbi15_score, dice=dice, ppv=ppv, images=images
        )

    def validation_epoch_end(self, outputs: typing.List[typing.Any]) -> None:
        images = outputs[0].pop("images")
        self._log_images(images)
        log_client = self.logger.experiment
        for k in outputs[0].keys():
            metric = torch.stack([output[k] for output in outputs]).mean()
            if hasattr(log_client, "log_metric"):
                log_client.log_metric(
                    run_id=self.logger.run_id,
                    key=f"val_{k}",
                    value=metric.item(),
                    step=self.current_epoch,
                )
                self.log(f"val_{k}", metric, logger=False)
            else:
                self.log(f"val_{k}", metric)

    def predict_step(
        self,
        batch: PredictBatch,
        batch_idx: builtins.int,
        dataloader_idx: typing.Optional[builtins.int] = None,
    ) -> torch.Tensor:
        if self._predict_with_patches(batch):
            assert isinstance(batch, PatchesImagePredictBatch)
            return self._predict_patch_image(batch)
        else:
            assert isinstance(batch, WholeImagePredictBatch)
            return self._predict_whole_image(batch)

    def on_predict_batch_end(  # type: ignore[override]
        self,
        pred_step_outputs: torch.Tensor,
        batch: PredictBatch,
        batch_idx: builtins.int,
        dataloader_idx: builtins.int,
    ) -> PredictBatch:
        if self._predict_with_patches(batch):
            assert isinstance(batch, PatchesImagePredictBatch)
            self._predict_accumulate_patches(pred_step_outputs, batch)
            if (batch_idx + 1) == batch.total_batches:
                self._predict_save_patch_image(batch)
        else:
            assert isinstance(batch, WholeImagePredictBatch)
            self._predict_save_whole_image(pred_step_outputs, batch)
        return batch

    def configure_optimizers(
        self,
    ) -> typing.Tuple[typing.List[Optimizer], typing.List[LambdaLR]]:
        optimizer: typing.Union[AdamW, RMSprop]
        betas = self.hparams.betas
        assert isinstance(betas, typing.Collection)
        assert len(betas) == 2
        beta1, beta2 = betas
        assert isinstance(beta1, builtins.float) and isinstance(beta2, builtins.float)
        lr = self.hparams.learning_rate
        assert isinstance(lr, builtins.float)
        weight_decay = self.hparams.weight_decay
        assert isinstance(weight_decay, builtins.float)
        if self.hparams.rmsprop:
            optimizer = RMSprop(
                self.parameters(),
                lr=lr,
                momentum=beta1,
                alpha=beta2,
                weight_decay=weight_decay,
            )
        else:
            optimizer = AdamW(
                self.parameters(),
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=weight_decay,
            )

        scheduler = LambdaLR(optimizer, lr_lambda=self.decay_rule)
        return [optimizer], [scheduler]

    def decay_rule(self, epoch: builtins.int) -> builtins.float:
        n_epochs = self.hparams.n_epochs
        assert isinstance(n_epochs, builtins.int)
        decay_after = self.hparams.decay_after
        assert isinstance(decay_after, builtins.int)
        numerator = max(0, epoch - decay_after)
        denominator = float(n_epochs + 1)
        lr: float = 1.0 - numerator / denominator
        return lr

    @staticmethod
    def _predict_with_patches(batch: PredictBatch) -> builtins.bool:
        return hasattr(batch, "grid_obj")

    def _predict_whole_image(self, batch: WholeImagePredictBatch) -> torch.Tensor:
        """for 3D networks, predict the whole image foreground at once"""
        src = batch.src
        bbox = BoundingBox3D.from_batch(src, pad=0)
        batch.src = bbox(src)
        pred_seg = self._predict_patch_image(batch)
        pred_seg = bbox.uncrop_batch(pred_seg)
        return pred_seg

    def _predict_patch_image(self, batch: PredictBatch) -> torch.Tensor:
        """for all 2D networks and 3D networks with a specified patch size"""
        src = batch.src
        pred = self(src)
        if self.hparams.num_classes == 1:
            pred_seg = torch.sigmoid(pred)
            if not self.hparams.predict_probability:
                pred_seg = pred_seg > self.hparams.threshold
        else:
            pred_seg = torch.softmax(pred, dim=1)
        pred_seg = pred_seg.float()
        return pred_seg

    def _clean_prediction(self, pred: np.ndarray) -> np.ndarray:
        assert pred.ndim == 3
        if not self.hparams.predict_probability:
            pred = clean_segmentation(pred)
        pred = pred.astype(np.float32)
        return pred

    def _predict_save_whole_image(
        self,
        pred_step_outputs: torch.Tensor,
        batch: WholeImagePredictBatch,
    ) -> None:
        assert len(pred_step_outputs) == len(batch.affine)
        nifti_attrs = zip(
            pred_step_outputs.detach().cpu(),
            batch.affine,
            batch.path,
            batch.out,
        )
        for pred, affine, path, fn in nifti_attrs:
            if self._model_num != ModelNum(num=1, out_of=1):
                fn = str(append_num_to_filename(fn, num=self._model_num.num))
            logging.info(f"Saving {fn}.")
            if batch.reorient:
                pred, affine = self._to_original_orientation(path, pred, affine)
            pred = pred.numpy().squeeze()
            pred = self._clean_prediction(pred)
            self._write_image(pred, affine, fn)

    def _predict_save_patch_image(self, batch: PatchesImagePredictBatch) -> None:
        pred_tensor = self.aggregator.get_output_tensor().detach().cpu()
        affine_tensor = batch.affine[0]
        if batch.reorient:
            pred_tensor, affine_tensor = self._to_original_orientation(
                batch.path, pred_tensor, affine_tensor
            )
        pred = pred_tensor.numpy().squeeze()
        affine = affine_tensor.numpy()
        pred = self._clean_prediction(pred)
        fn = batch.out[0]
        if self._model_num != ModelNum(num=1, out_of=1):
            fn = str(append_num_to_filename(fn, num=self._model_num.num))
        logging.info(f"Saving {fn}.")
        self._write_image(pred, affine, fn)
        del self.aggregator

    @staticmethod
    def _save_as_dicom(filename: builtins.str) -> builtins.bool:
        save_dicom = str(filename).endswith(".dcm")
        if save_dicom:
            warnings.warn(
                "DICOM Segmentation Objects only support uint8. "
                "Cannot save a probability image."
            )
        return save_dicom

    def _write_image(
        self,
        image: np.ndarray,
        affine: np.ndarray,
        filename: builtins.str,
    ) -> None:
        if image.ndim != 4:
            image = image[np.newaxis]
        assert image.ndim == 4
        if self._save_as_dicom(filename):
            image = (image > self.hparams.threshold).astype(np.uint8)
        output_image = tio.ScalarImage(tensor=image, affine=affine)
        output_image.save(filename)

    @staticmethod
    def _to_original_orientation(
        original_path: builtins.str,
        data: torch.Tensor,
        affine: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        original = tio.ScalarImage(original_path)
        image = tio.ScalarImage(tensor=data, affine=affine)
        if original.orientation != image.orientation:
            orientation = "".join(original.orientation)
            reoriented = sitk.DICOMOrient(image.as_sitk(), orientation)
            reoriented_data = sitk.GetArrayFromImage(reoriented).transpose()[np.newaxis]
            image = tio.ScalarImage(tensor=reoriented_data, affine=original.affine)
        new_affine = (
            torch.from_numpy(image.affine)
            if isinstance(image.affine, np.ndarray)
            else image.affine
        )
        return image.data, new_affine

    def _predict_accumulate_patches(
        self,
        pred_step_outputs: torch.Tensor,
        batch: PatchesImagePredictBatch,
    ) -> None:
        p3d = batch.pseudo3d_dim
        locations = batch.locations
        if not hasattr(self, "aggregator"):
            self.aggregator = tio.GridAggregator(
                batch.grid_obj,
                overlap_mode="average",
            )
        if p3d is not None:
            locations = self._fix_pseudo3d_locations(locations, p3d)
            pred_step_outputs.unsqueeze_(p3d + 2)  # +2 to offset batch/channel dims
        self.aggregator.add_batch(pred_step_outputs, locations)

    @staticmethod
    def _fix_pseudo3d_locations(
        locations: torch.Tensor, pseudo3d_dim: builtins.int
    ) -> torch.Tensor:
        """Fix locations for aggregator when using pseudo3d

        locations were determined by the pseudo3d input, not the 1 channel target.
        this fixes the locations to use 1 channel corresponding to the pseudo3d dim.
        """
        for n, location in enumerate(locations):
            i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
            if pseudo3d_dim == 0:
                i = torch.div(i_fin - i_ini, 2, rounding_mode="floor") + i_ini
                i_ini = i
                i_fin = i + 1
            elif pseudo3d_dim == 1:
                j = torch.div(j_fin - j_ini, 2, rounding_mode="floor") + j_ini
                j_ini = j
                j_fin = j + 1
            elif pseudo3d_dim == 2:
                k = torch.div(k_fin - k_ini, 2, rounding_mode="floor") + k_ini
                k_ini = k
                k_fin = k + 1
            else:
                raise ValueError(
                    f"pseudo3d_dim must be 0, 1, or 2. Got {pseudo3d_dim}."
                )
            locations[n, :] = torch.tensor(
                [i_ini, j_ini, k_ini, i_fin, j_fin, k_fin],
                dtype=locations.dtype,
                device=locations.device,
            )
        return locations

    def _log_images(
        self,
        images: typing.Dict[builtins.str, typing.Any],
        *,
        mlflow_image_limit: builtins.int = 5,
    ) -> None:
        n = self.current_epoch
        mid_slice = None
        dim: builtins.int = images.pop("dim")
        for i, (key, image) in enumerate(images.items()):
            if dim == 3:
                if mid_slice is None:
                    mid_slice = image.shape[-1] // 2
                image_slice = image[..., mid_slice]
            elif dim == 2:
                image_slice = image
            else:
                raise ValueError(f"Image dimension must be either 2 or 3. Got {dim}.")
            if self.hparams.soft_labels and key == "pred":
                image_slice = torch.sigmoid(image_slice)
            elif key == "pred":
                if self.hparams.num_classes == 1:
                    threshold = self.hparams.threshold
                    image_slice = torch.sigmoid(image_slice) > threshold
                else:
                    image_slice = torch.argmax(image_slice, 1, keepdim=True)
            elif key == "truth":
                image_slice = image_slice > 0.0
            else:
                image_slice = minmax_scale_batch(image_slice)
            log_client = self.logger.experiment
            if hasattr(log_client, "add_images"):
                log_client.add_images(key, image_slice, n, dataformats="NCHW")
            elif hasattr(log_client, "log_image"):
                _key = key.replace("channel_", "").replace("_", "-")
                _epoch = str(n).zfill(3)
                _image_slices = image_slice.detach().cpu().numpy().squeeze()
                if _image_slices.ndim == 2:
                    _image_slices = _image_slices[np.newaxis, ...]
                for j, _image_slice in enumerate(_image_slices):
                    _batch_idx = str(j).zfill(3)
                    log_client.log_image(
                        self.logger.run_id,
                        _image_slice,
                        f"epoch-{_epoch}_{_key}_batch-idx-{_batch_idx}.png",
                    )
                    if j >= mlflow_image_limit:
                        break
            else:
                raise RuntimeError("Image logging functionality not found in logger.")

    @staticmethod
    def _is_3d_image_batch(tensor: torch.Tensor) -> builtins.bool:
        ans: builtins.bool = tensor.ndim == 5
        return ans

    @staticmethod
    def _is_2d_image_batch(tensor: torch.Tensor) -> builtins.bool:
        ans: builtins.bool = tensor.ndim == 4
        return ans

    @staticmethod
    def add_io_arguments(parent_parser: ArgParser) -> ArgParser:
        parser = parent_parser.add_argument_group("I/O")
        parser.add_argument(
            "-ni",
            "--num-input",
            type=positive_int(),
            default=1,
            help="number of input images (should match the number "
            "of non-label/other fields in the input csv)",
        )
        parser.add_argument(
            "-nc",
            "--num-classes",
            type=positive_int(),
            default=1,
            help="number of classes to segment (1 for binary segmentation)",
        )
        return parent_parser

    @staticmethod
    def add_training_arguments(parent_parser: ArgParser) -> ArgParser:
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
            "-pw",
            "--pos-weight",
            type=positive_float_or_none(),
            default=None,
            help="weight of positive class in focal/bce loss component of "
            "combo loss function (None -> equal, which is equivalent to "
            "setting this to 1.0)",
        )
        parser.add_argument(
            "-fg",
            "--focal-gamma",
            type=nonnegative_float(),
            default=0.0,
            help="gamma parameter for focal loss component of combo loss (0.0 -> BCE)",
        )
        parser.add_argument(
            "-cw",
            "--combo-weight",
            type=probability_float(),
            default=0.6,
            help="weight of focal loss component in combo loss",
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
        parser.add_argument(
            "-tm",
            "--track-metric",
            type=str,
            default="isbi15_score",
            choices=("dice", "isbi15_score", "loss", "ppv"),
            help="pick the best network based on this metric; "
            "metric is the mean over a validation epoch.",
        )
        return parent_parser

    @staticmethod
    def add_other_arguments(parent_parser: ArgParser) -> ArgParser:
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
    def add_testing_arguments(parent_parser: ArgParser) -> ArgParser:
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


# flake8: noqa: E501
class LesionSegLightningTiramisu(LesionSegLightningBase):
    """3D Tiramisu-based PyTorch-Lightning module for lesion segmentation

    See Also:
        Jégou, Simon, et al. "The one hundred layers tiramisu: Fully
        convolutional densenets for semantic segmentation." CVPR. 2017.

        Zhang, Huahong, et al. "Multiple sclerosis lesion segmentation
        with Tiramisu and 2.5D stacked slices." International Conference
        on Medical Image Computing and Computer-Assisted Intervention.
        Springer, Cham, 2019.

    Args:
        network_dim (int): use a 2D or 3D convolutions
        in_channels (int): number of input channels
        num_classes (int): number of classes to segment with the network
        down_blocks (typing.Collection[int]): number of layers in each block in down path
        up_blocks (typing.Collection[int]): number of layers in each block in up path
        bottleneck_layers (int): number of layers in the bottleneck
        growth_rate (int): number of channels to grow by in each layer
        first_conv_out_channels (int): number of output channels in first conv
        dropout_rate (float): dropout rate/probability
        init_type (str): method to initialize the weights of network
        gain (float): gain parameter for initialization
        n_epochs (int): number of epochs to train the network
        learning_rate (float): learning rate for the optimizer
        betas (typing.Tuple[float, float]): momentum parameters for adam
        weight_decay (float): weight decay for optimizer
        loss_function (str): loss function to use in training
        pos_weight (typing.Optional[float]): weight for positive class
            in focal/bce loss if using combo loss function
        focal_gamma (float): gamma param for focal loss
            if using combo loss function (0. -> BCE)
        combo_weight (float): weight by which to balance focal and Dice
            losses in combo loss function
        decay_after (int): decay learning rate linearly after this many epochs
        rmsprop (bool): use rmsprop instead of adamw
        soft_labels (bool): use non-binary labels for training
        threshold (float): threshold by which to decide on positive class
        min_lesion_size (int): minimum lesion size in voxels in output prediction
        fill_holes (bool): use binary fill holes operation on label
        predict_probability (bool): save a probability image instead of a binary one
        mixup (bool): use mixup in training
        mixup_alpha (float): mixup parameter for beta distribution
        num_input (int): number of different images input to the network,
            differs from in_channels when using pseudo3d
        _model_num (ModelNum): internal param for ith of n models
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        *,
        network_dim: builtins.int = 3,
        in_channels: builtins.int = 1,
        num_classes: builtins.int = 1,
        down_blocks: typing.Collection[builtins.int] = (4, 4, 4, 4, 4),
        up_blocks: typing.Collection[builtins.int] = (4, 4, 4, 4, 4),
        bottleneck_layers: builtins.int = 4,
        growth_rate: builtins.int = 16,
        first_conv_out_channels: builtins.int = 48,
        dropout_rate: builtins.float = 0.2,
        init_type: builtins.str = "normal",
        gain: builtins.float = 0.02,
        n_epochs: builtins.int = 1,
        learning_rate: builtins.float = 1e-3,
        betas: typing.Tuple[builtins.float, builtins.float] = (0.9, 0.99),
        weight_decay: builtins.float = 1e-7,
        loss_function: builtins.str = "combo",
        pos_weight: typing.Optional[builtins.float] = None,
        focal_gamma: builtins.float = 0.0,
        combo_weight: builtins.float = 0.6,
        decay_after: builtins.int = 8,
        rmsprop: builtins.bool = False,
        soft_labels: builtins.bool = False,
        threshold: builtins.float = 0.5,
        min_lesion_size: builtins.int = 3,
        fill_holes: builtins.bool = True,
        predict_probability: builtins.bool = False,
        mixup: builtins.bool = False,
        mixup_alpha: builtins.float = 0.4,
        num_input: builtins.int = 1,
        resize_method: builtins.str = "crop",
        input_shape: typing.Optional[typing.Tuple[builtins.int, ...]] = None,
        static_upsample: builtins.bool = True,
        _model_num: ModelNum = ModelNum(1, 1),
        **kwargs,
    ):
        network_class: typing.Union[typing.Type[Tiramisu2d], typing.Type[Tiramisu3d]]
        if network_dim == 2:
            network_class = Tiramisu2d
        elif network_dim == 3:
            network_class = Tiramisu3d
        else:
            raise ValueError(f"Network dim. must be 2 or 3. Got {network_dim}.")
        network = network_class(
            in_channels=in_channels,
            out_channels=num_classes,
            down_blocks=down_blocks,
            up_blocks=up_blocks,
            bottleneck_layers=bottleneck_layers,
            growth_rate=growth_rate,
            first_conv_out_channels=first_conv_out_channels,
            dropout_rate=dropout_rate,
            resize_method=ResizeMethod.from_string(resize_method),
            input_shape=input_shape,
            static_upsample=static_upsample,
        )
        init_weights(network, init_type=InitType.from_string(init_type), gain=gain)
        super().__init__(
            network=network,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            loss_function=loss_function,
            pos_weight=pos_weight,
            focal_gamma=focal_gamma,
            combo_weight=combo_weight,
            decay_after=decay_after,
            rmsprop=rmsprop,
            soft_labels=soft_labels,
            threshold=threshold,
            min_lesion_size=min_lesion_size,
            fill_holes=fill_holes,
            predict_probability=predict_probability,
            mixup=mixup,
            mixup_alpha=mixup_alpha,
            num_input=num_input,
            num_classes=num_classes,
            _model_num=_model_num,
            **kwargs,
        )
        self.save_hyperparameters(ignore="_model_num")

    @staticmethod
    def add_model_arguments(parent_parser: ArgParser) -> ArgParser:
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument(
            "-ic",
            "--in-channels",
            type=positive_int(),
            default=1,
            help="number of input channels",
        )
        parser.add_argument(
            "-oc",
            "--out-channels",
            type=positive_int(),
            default=1,
            help="number of output channels",
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
        parser.add_argument(
            "-rm",
            "--resize-method",
            type=str,
            default="crop",
            choices=("crop", "interpolate"),
            help="use transpose conv and crop or normal conv "
            "and interpolate to correct size in upsample branch",
        )
        return parent_parser
