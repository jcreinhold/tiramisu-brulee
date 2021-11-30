=======
History
=======

0.1.35 (2021-11-30)
-------------------

* Add support for deterministic validation patches

0.1.34 (2021-11-19)
-------------------

* Add support for bandit
* Fix warning filter for dataloader

0.1.33 (2021-11-18)
-------------------

* Add support for a dataset that can prevent a known PyTorch/Python memory leak issue

0.1.32 (2021-11-16)
-------------------

* Fix bug in prediction where data not transferred to GPU

0.1.31 (2021-11-16)
-------------------

* Updates to support pytorch-lightning~=1.5.1

0.1.30 (2021-11-15)
-------------------

* Add option to change label sampling probabilities
* Bump pip version in requirements_dev.txt for security

0.1.29 (2021-11-02)
-------------------

* Support dicom images in lesion-predict
* Change logger.warnings to warnings.warn
* Remove deprecation warning for floor divide in torch in patch-based prediction

0.1.28 (2021-11-01)
-------------------

* Add commit hash logger function to tag MLFlow runs
* Save configuration files to MLFlow
* Add option to save top K checkpoints

0.1.27 (2021-10-12)
-------------------

* Add union and voting aggregation to prediction and other minor bug fixes

0.1.26 (2021-08-09)
-------------------

* Reformat with newer version of black (v21.7b0)
* Change to ``every_n_epochs`` in ``ModelCheckpoint`` since ``every_n_val_epochs`` will be deprecated

0.1.25 (2021-08-06)
-------------------

* Detect and use tensorboard directory (``/opt/ml/output/tensorboard``) for logging on SageMaker

0.1.24 (2021-08-06)
-------------------

* Add experiment and trial name as options to explicitly specify artifact locations

0.1.23 (2021-08-04)
-------------------

* Change AWS option to just MLFlow
* Compliant with mypy
* Other minor bug fixes and fix docs

0.1.22 (2021-07-30)
-------------------

* Add AWS extras (MLFlow and `train` and `serve` console scripts)
* Add option to resample images within a subject for consistent orientation
* Add optional check of DICOM images to determine if they are uniformly sampled
* Make package compatible with Python 3.6 and 3.9
* Split CLI functions into a subpackage for better organization

0.1.21 (2021-07-27)
-------------------

* Add MLFlow logging option
* Add support for reading DICOM images and writing DICOM (Segmentation Objects)
* Fix some type hints and make pos_weight a vector of length 1

0.1.20 (2021-07-25)
-------------------

* Make reorientation to canonical optional
* Add option to track best network on validation Dice, PPV, loss, or ISBI15 score
* Unify and simplify the positive weight in focal/bce component of combo loss
* Change flip in spatial augmentation to only do lateral flips
* Fix predict_probability flag in CLI

0.1.19 (2021-07-22)
-------------------

* Fix Dice score component of almost_isbi15_score metric

0.1.18 (2021-06-30)
-------------------

* Fix reorientation to original orientation from canonical in prediction.


0.1.17 (2021-06-11)
-------------------

* Migrate to Github actions for testing and deployment.

0.1.16 (2021-06-11)
-------------------

* Add support for training with all orientations. Convert all inputs to canonical
  orientation before input to network in training and prediction (and convert back
  to original orientation in prediction before saving).

0.1.15 (2021-06-05)
-------------------

* Add multi-class segmentation support, headers to predictions, and other bug fixes.

0.1.14 (2021-06-03)
-------------------

* Bug fixes for training multiple models, remove unintended restriction on column names

0.1.13 (2021-05-31)
-------------------

* Fix a bug when using pseudo3d_dim == 0.

0.1.12 (2021-05-31)
-------------------

* Fix bug with patch-based prediction and add support for training/predicting with networks
  with differing pseudo3d dimensions.

0.1.11 (2021-05-30)
-------------------

* Add better prediction support for pseudo3d networks.

0.1.10 (2021-05-29)
-------------------

* Add CLI usage documentation and fix some minor bugs/typos.

0.1.9 (2021-05-28)
------------------

* Add pseudo3d (2.5D) support and patch-based prediction

0.1.8 (2021-05-27)
------------------

* Fix ISBI 15 score metric

0.1.7 (2021-05-25)
------------------

* Add precision to arguments for prediction

0.1.6 (2021-05-25)
------------------

* Improve documentation

0.1.5 (2021-05-25)
------------------

* Add docs and split out CLIs from seg module

0.1.4 (2021-05-13)
------------------

* Add lesion segmentation CLI.

0.1.3 (2021-05-13)
------------------

* Fix deployment by fixing repo name in travis.

0.1.2 (2021-05-13)
------------------

* Fix supported versions and docs.

0.1.1 (2021-05-13)
------------------

* Fix tests and deployment.

0.1.0 (2021-05-13)
------------------

* First release on PyPI.
