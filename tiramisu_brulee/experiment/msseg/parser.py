#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiramisu_brulee.experiment.msseg.parser

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 16, 2021
"""

__all__ = []

from argparse import ArgumentParser


def arg_parser():
    parser = ArgumentParser(description='train/test a 3D Tiramisu model')

    required = parser.add_argument_group('Required')
    required.add_argument('--train-csv', type=str, default=None,
                          help='path to csv with training images')
    required.add_argument('--valid-csv', type=str, default=None,
                          help='path to csv with validation images')
    required.add_argument('--test-csv', type=str, default=None,
                          help='path to csv with test images')
    required.add_argument('--trained-model-path', type=str, default=None,
                          help='path to output the trained model')
    required.add_argument('--out-path', type=str, default=None,
                          help='path to output the images in testing')

    options = parser.add_argument_group('Options')
    options.add_argument('-bs', '--batch-size', type=int, default=2,
                         help='training/test batch size [Default=%(default)s]')
    options.add_argument('-vbs', '--valid-batch-size', type=int, default=2,
                         help='validation batch size [Default=%(default)s]')
    options.add_argument('-hs', '--head-size', type=int, default=48,
                         help='size of head (for multi-task) [Default=%(default)s]')
    options.add_argument('-cw', '--combo-weight', type=float, default=0.6,
                         help='weight of positive class in combo loss [Default=%(default)s]')
    options.add_argument('-mg', '--multigpu', action='store_true', default=False,
                         help='use multiple gpus [Default=%(default)s]')
    options.add_argument('-nw', '--num-workers', type=int, default=16,
                         help='number of CPU processors to use [Default=%(default)s]')
    options.add_argument('-ps', '--patch-size', type=int, nargs=3, default=(96, 96, 96),
                         help='training/test patch size extracted from image [Default=%(default)s]')
    options.add_argument('-vps', '--valid-patch-size', type=int, nargs=3, default=(128, 128, 128),
                         help='validation patch size extracted from image [Default=%(default)s]')
    options.add_argument('-ql', '--queue-length', type=int, default=200,
                         help='queue length for torchio sampler [Default=%(default)s]')
    options.add_argument('-rs', '--resume', type=str, default=None,
                         help='resume from this path [Default=%(default)s]')
    options.add_argument('-sd', '--seed', type=int, default=0,
                         help='set seed for reproducibility [Default=%(default)s]')
    options.add_argument('-spv', '--samples-per-volume', type=int, default=10,
                         help='samples per volume for torchio sampler [Default=%(default)s]')
    options.add_argument('-th', '--threshold', type=float, default=0.5,
                         help='prob. threshold for seg [Default=%(default)s]')
    options.add_argument('-uls', '--use-label-sampler', action='store_true',
                         help="use label sampler instead of uniform [Default%(default)s]")
    options.add_argument('--use-multitask', action='store_true', default=False,
                         help='use multitask objective [Default=%(default)s]')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")

    train_options = parser.add_argument_group('Training Options')
    train_options.add_argument('-bt', '--betas', type=float, default=(0.9, 0.999), nargs=2,
                               help='adamw momentum parameters (or, for RMSprop, momentum and alpha params)'
                                    ' [Default=%(default)s]')
    train_options.add_argument('-da', '--decay-after', type=int, default=8,
                               help='decay learning rate after this number of epochs [Default=%(default)s]')
    train_options.add_argument('-iw', '--isbi15score-weight', type=float, default=1.,
                               help='weight for isbi15 score in isbi15_score_minus_loss'
                                    '(1. is equal weighting) [Default=%(default)s]')
    train_options.add_argument('-lr', '--learning-rate', type=float, default=3e-4,
                               help='learning rate for the optimizer [Default=%(default)s]')
    train_options.add_argument('-lf', '--loss-function', type=str, default='combo',
                               choices=['combo', 'l1', 'mse'],
                               help='loss function to train the network [Default=%(default)s]')
    train_options.add_argument('-ne', '--n-epochs', type=int, default=64,
                               help='number of epochs [Default=%(default)s]')
    train_options.add_argument('-ur', '--use-rmsprop', action='store_true',
                               help="use rmsprop instead of adam [Default=%(default)s]")
    train_options.add_argument('-uw', '--use-weight', action='store_true',
                               help="use the weight field in the csv to weight subjects [Default=%(default)s]")
    train_options.add_argument('-wd', '--weight-decay', type=float, default=1e-5,
                               help="weight decay parameter for adamw [Default=%(default)s]")
    train_options.add_argument('-sm', '--softmask', action='store_true',
                               help="use softmasks for training [Default=%(default)s]")
    train_options.add_argument('-sw', '--syn-weight', type=float, default=0.1,
                               help='weight of synthesis objective [Default=%(default)s]')

    nn_options = parser.add_argument_group('Neural Network Options')
    nn_options.add_argument('-ic', '--in-channels', type=int, default=1,
                            help='number of input channels [Default=%(default)s]')
    nn_options.add_argument('-oc', '--out-channels', type=int, default=1,
                            help='number of output channels [Default=%(default)s]')
    nn_options.add_argument('-dr', '--dropout-rate', type=float, default=0.1,
                            help='dropout rate/probability [Default=%(default)s]')
    nn_options.add_argument('-it', '--init-type', type=str, default='he_uniform',
                            choices=('normal', 'xavier_normal', 'he_normal', 'he_uniform', 'orthogonal'),
                            help='use this type of initialization for the network [Default=%(default)s]')
    nn_options.add_argument('-ig', '--init-gain', type=float, default=0.2,
                            help='use this initialization gain for initialization [Default=%(default)s]')
    nn_options.add_argument('-db', '--down-blocks', type=int, default=(4, 4, 4, 4, 4), nargs='+',
                            help='tiramisu down block specification [Default=%(default)s]')
    nn_options.add_argument('-ub', '--up-blocks', type=int, default=(4, 4, 4, 4, 4), nargs='+',
                            help='tiramisu up block specification [Default=%(default)s]')
    nn_options.add_argument('-bl', '--bottleneck-layers', type=int, default=4,
                            help='tiramisu bottleneck specification [Default=%(default)s]')
    nn_options.add_argument('-gr', '--growth-rate', type=int, default=12,
                            help='tiramisu growth rate specification [Default=%(default)s]')
    nn_options.add_argument('-fcoc', '--first-conv-out-channels', type=int, default=48,
                            help='number of output channels in first conv [Default=%(default)s]')

    aug_options = parser.add_argument_group('Data Augmentation Options')
    aug_options.add_argument('--use-aug', action='store_true', default=False,
                             help='use data augmentation [Default=%(default)s]')
    aug_options.add_argument('--use-mixup', action='store_true', default=False,
                             help='use mixup [Default=%(default)s]')
    aug_options.add_argument('-ma', '--mixup-alpha', type=float, default=0.4,
                             help='mixup alpha parameter for beta dist. [Default=%(default)s]')

    post_options = parser.add_argument_group('Post-processing Options')
    post_options.add_argument('-mls', '--min-lesion-size', type=int, default=3,
                              help='in testing, remove lesions smaller in voxels than this [Default=%(default)s]')
    post_options.add_argument('-fh', '--fill-holes', action='store_true', default=False,
                              help='in testing, preform binary hole filling [Default=%(default)s]')
    return parser
