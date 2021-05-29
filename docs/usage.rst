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

Tutorial
--------

To use the command-line interfaces (CLIs), install ``tiramisu-brulee`` with
the ``[lesionseg]`` extras.

Training
~~~~~~~~

To train a network, create one or more CSV files with headers like "t1",
"flair", "t2", "pd", etc. These columns will be interpreted as the input
images. Also include a column called "label" which will be the target image.
You should also create a separate CSV file or files for validation. The
validation CSVs should contain the same columns as the training but with paths
to images not included in the training CSVs.

Now you should create a configuration file for training with::

   lesion-train --print_config > train_config.yaml

Alternatively, you can create a configuration file for training
*with help comments* running, e.g.::

   lesion-train --print_config=comments > train_config.yaml

You'll need to add the training CSV files in the ``train_csv`` argument and
all of the the validation CSV files in the ``valid_csv`` argument. Make sure
they are in correspondence.

You'll also need to modify the ``num_input`` argument to match the number of
non-``label`` columns in the csv files.

Make sure to set the ``gpus`` argument to some number greater than or equal to
1 (assuming you want to train on one of more GPUs). If you use multiple GPUs,
you should also change ``accelerator`` to ``ddp``.

You should also consider using a 2.5D or pseudo-3d network. The
state-of-the-art in MS lesion segmentation uses such a methodology.
Basically, a 2.5D network is uses 2D operations instead of 3D and the
input to the network is a stack of adjacent slices, concatenated on the
channel row.

To use a 2.5D/pseudo-3D network, determine which axis you want to stack the
slices on. Set ``pseudo3d_dim`` to that axis (e.g., 2). Then change the patch
size to something like::

 - 128
 - 128
 - M

(where ``M`` is something relatively small and odd like 3). Note that the
``pseudo3d_dim`` corresponds to the (0-indexed) element of the patch size
list. If you have set ``num_input`` to ``N``, these settings will result in a
2D network with ``N * M`` input channels and trained/validated on ``128 x 128``
images.

Now you can use your config file to train a set of lesion segmentation Tiramisu
neural networks with::

    lesion-train --config train_config.yaml

This will create a directory called ``lesion_tiramisu_experiment`` in the
directory in which you run the above command (assuming you haven't changed
the ``default_root_dir`` field in the config file). The more times you run
the above command, the more ``lesion_tiramisu_experiment/version_*``
directories will be created. So the current run is usually in the last
``version_*`` directory. Note that if you provide ``T`` ``train_csv`` files,
``T`` ``version_*`` will be created for each run, e.g., if
``lesion_tiramisu_experiment`` contains ``version_12`` and you start training
with 3 CSV files, then lesion-train will create ``version_13``, ``version_14``
and ``version_15``.

You can montior your experiment with tensorboard by running, e.g.,::

    tensorboard --logdir=lesion_tiramisu_experiment/version_13

(where the version number is changed appropriately based on your experiment.)

Prediction
~~~~~~~~~~
Once training is completed, a prediction config file will automatically be
generated. A ``predict_config.yaml`` file will be generated in every
``lesion_tiramisu_experiment/version_*`` directory. The ``model_path`` will
contain the best model path already filled in according to the (approximate)
ISBI 15 score on the validation data.

Copy one of those config files and modify it to use the your prediction CSV
file (i.e., a CSV with the same columns as the training minus the
``label`` column and with the addition of an ``out`` column which contains
the paths to save the image).

You can either use patches for prediction (by setting ``patch_size``) or
predict the whole image volume at once. If you predict with patches,
you'll need to tune the batch size. If you predict the whole volume
at once, leave ``batch_size`` at 1 because, even though it crops the
image based on estimates foreground values and inputs the only the
foreground image into the network, it is still memory-intensive.

If you run out of memory, try it on a machine with more memory or use
patch-based prediction. And/or try setting the precision to 16.

Use ``lesion-predict --config predict_config.yaml`` to run prediction.

Alternatively, use the ``lesion-predict-image`` script for single time-point
prediction. Note that this interface doesn't accept a config file. Note that
you input the image using the same name used for the header in training,
e.g.::

    lesion-predict-image --t1 /path/to/t1.nii --flair /path/to/flair.nii \
         --out path/to/prediction.nii ...

where ``--out`` is the output prediction and ``--label`` is excluded.

lesion-train
------------

.. argparse::
   :module: tiramisu_brulee.experiment.cli
   :func: train_parser
   :prog: lesion-train

lesion-predict
--------------

.. argparse::
   :module: tiramisu_brulee.experiment.cli
   :func: predict_parser
   :prog: lesion-predict

lesion-predict-image
--------------------

.. argparse::
   :module: tiramisu_brulee.experiment.cli
   :func: predict_image_parser
   :prog: lesion-predict-image
