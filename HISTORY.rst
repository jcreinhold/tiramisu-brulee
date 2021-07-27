=======
History
=======

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
