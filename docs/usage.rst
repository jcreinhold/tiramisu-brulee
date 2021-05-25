=====
Usage
=====

To use tiramisu-brulee in a project::

    import tiramisu_brulee

Import the 2D or 3D Tiramisu CNN version with:

.. code-block:: python

    from tiramisu_brulee.model import Tiramisu2d, Tiramisu3d

Command-line interfaces
=======================

lesion-train
------------

.. argparse::
   :module: tiramisu_brulee.experiment.lesion_seg.cli
   :func: train_parser
   :prog: lesion-train

lesion-predict
--------------

.. argparse::
   :module: tiramisu_brulee.experiment.lesion_seg.cli
   :func: predict_parser
   :prog: lesion-predict

lesion-predict-image
--------------------

.. argparse::
   :module: tiramisu_brulee.experiment.lesion_seg.cli
   :func: predict_image_parser
   :prog: lesion-predict-image
