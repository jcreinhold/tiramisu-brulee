===============
tiramisù-brûlée
===============


.. image:: https://img.shields.io/pypi/v/tiramisu_brulee.svg
        :target: https://pypi.python.org/pypi/tiramisu-brulee

.. image:: https://api.travis-ci.com/jcreinhold/tiramisu-brulee.svg
        :target: https://travis-ci.com/jcreinhold/tiramisu-brulee

.. image:: https://readthedocs.org/projects/tiramisu-brulee/badge/?version=latest
        :target: https://tiramisu-brulee.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

A 2D and 3D PyTorch implementation of the Tiramisu CNN

This package is primarily used for multiple sclerosis (MS) lesion segmentation; specifically, T2 lesions in the brain.

* Free software: Apache Software License 2.0
* Documentation: https://tiramisu-brulee.readthedocs.io.

Install
-------

The easiest way to install the package is with::

    pip install tiramisu-brulee

Alternatively, you can download the source and run::

    python setup.py install

If you want a CLI to train a 3D lesion segmentation model, install with::

    pip install tiramisu-brulee[lesionseg]

Basic Usage
-----------

Import the 2D or 3D Tiramisu version with:

.. code-block:: python

    from tiramisu_brulee.model import Tiramisu2d, Tiramisu3d


References
----------

[1] Jégou, Simon, et al. "The one hundred layers tiramisu: Fully convolutional densenets for semantic segmentation."
CVPR. 2017.

[2] Zhang, Huahong, et al. "Multiple sclerosis lesion segmentation with Tiramisu and 2.5D stacked slices." International
Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2019.

Why the name?
-------------

Why is the name tiramisù-brûlée? Well, tiramisù is named after the neural network [1] whose name is inspired by
the dessert; however, tiramisu—by itself—was already taken as a package on PyPI. I added brûlée to get around the
existence of that package and because this package is written in PyTorch (torch -> burnt). Plus brûlée in English is
often associated with the dessert crème brûlée. Why combine an Italian word (tiramisù) with a French word (brûlée)?
Because I didn't think about it until after I already deployed the package to PyPI.
