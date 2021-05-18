#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'torch',
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'nibabel',
    'pandas',
    'pytest>=3',
    'pytorch-lightning',
    'torchio',
]

extras_requirements = {
    'medical': [
        'lesion-metrics',
        'nibabel',
        'pandas',
        'scikit-image',
        'torchio',
    ],
    'lightning': [
        'pytorch-lightning[all]',
        'torchmetrics',
    ]
}

setup(
    author="Jacob Reinhold",
    author_email='jcreinhold@gmail.com',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A 2D and 3D PyTorch implementation of the Tiramisu CNN",
    entry_points={
        'console_scripts': [
            'lesion-seg=tiramisu_brulee.experiment.lesion_seg.seg:main',
        ],
    },
    install_requires=requirements,
    extras_requires=extras_requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='tiramisu, segmentation, neural network, convolutional, pytorch',
    name='tiramisu_brulee',
    packages=find_packages(include=['tiramisu_brulee', 'tiramisu_brulee.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/jcreinhold/tiramisu_brulee',
    version='0.1.3',
    zip_safe=False,
)
