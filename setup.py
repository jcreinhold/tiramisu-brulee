#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "torch",
]

setup_requirements = [
    "pytest-runner",
]

_other_reqs = [
    "jsonargparse",
    "lesion-metrics",
    "nibabel",
    "pandas",
    "pytorch-lightning",
    "PyYAML",
    "ruyaml",
    "scikit-image",
    "torchio",
    "torchmetrics",
]

test_requirements = _other_reqs + [
    "pytest>=3",
]

extras_requirements = {
    "lesionseg": _other_reqs,
}

setup(
    author="Jacob Reinhold",
    author_email="jcreinhold@gmail.com",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A 2D and 3D PyTorch implementation of the Tiramisu CNN",
    entry_points={
        "console_scripts": [
            "lesion-train=tiramisu_brulee.experiment.lesion_seg.seg:train",
            "lesion-predict=tiramisu_brulee.experiment.lesion_seg.seg:predict",
            "lesion-predict-image=tiramisu_brulee.experiment.lesion_seg.seg:predict_image",
        ],
    },
    install_requires=requirements,
    extras_require=extras_requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="tiramisu, segmentation, neural network, convolutional, pytorch",
    name="tiramisu_brulee",
    packages=find_packages(include=["tiramisu_brulee", "tiramisu_brulee.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/jcreinhold/tiramisu_brulee",
    version="0.1.4",
    zip_safe=False,
)
