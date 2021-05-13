===============
tiramisu-brûlée
===============


.. image:: https://img.shields.io/pypi/v/tiramisu_brulee.svg
        :target: https://pypi.python.org/pypi/tiramisu_brulee

.. image:: https://img.shields.io/travis/jcreinhold/tiramisu_brulee.svg
        :target: https://travis-ci.com/jcreinhold/tiramisu_brulee

.. image:: https://readthedocs.org/projects/tiramisu-brulee/badge/?version=latest
        :target: https://tiramisu-brulee.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


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

Basic Usage
-----------

    from tiramisu_brulee import Tiramisu2D, Tiramisu3D


References
---------------

[1] Jégou, Simon, et al. "The one hundred layers tiramisu: Fully convolutional densenets for semantic segmentation."
CVPR. 2017.

[2] Zhang, Huahong, et al. "Multiple sclerosis lesion segmentation with Tiramisu and 2.5D stacked slices." International
Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2019.
