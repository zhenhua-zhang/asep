#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for package asep"""

import os
import sys
import math
import json
import time
import functools

import yaml
import numpy
import scipy


def timmer(func):
    """Print the runtime of the decorated function

    Args:
        func (callable): function to be decoreated
    """
    @functools.wraps(func)
    def wrapper_timmer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        func_name = func.__name__ + " "
        used_time = time.perf_counter() - start_time
        print(
            '{:.<40} DONE, time: {:.5f} secs'.format(func_name, used_time),
            file=sys.stderr
        )
        return value

    return wrapper_timmer


def check_resources():
    """Chekc resources available"""
    cpus_per_task = os.environ['SLURM_CPUS_PER_TASK']
    math.sqrt(cpus_per_task)


def format_print(title, main_content, pipe=sys.stdout):
    """A method format the prints"""
    head = '-' * 3
    flag = ' '.join([head, title, ": "])
    print(flag, '\n   ', main_content, "\n\n", file=pipe)


def check_keys(pool_a, pool_b):
    """Check if all elements in pool_a are also in pool_b"""
    if not isinstance(pool_a, (list, tuple)):
        raise TypeError('Require iterable value for pool_a...')
    if not isinstance(pool_b, (list, tuple)):
        raise TypeError('Require iterable value for pool_b...')

    pool_a_size = len(pool_a)
    pool_b_size = len(pool_b)
    if pool_a_size >= pool_b_size:
        for key in pool_b:
            if key not in pool_a:
                raise KeyError('Invalid element {}'.format(key))
    else:
        for key in pool_a:
            if key not in pool_b:
                raise KeyError('Invalid element {}'.format(key))
    return True


def set_sed(sed=None):
    """Set the random seed of numpy"""
    if sed:
        numpy.random.seed(sed)
    else:
        numpy.random.seed(3142)


def feature_pre_selection_by_spearman(
        input_dataframe, drop_list, target=None, pvalue_threshhold=0.1):
    """Drop features with low correlation to target variables."""

    if not isinstance(drop_list, (list, tuple)):
        raise TypeError("drop_list should be list, tuple")

    candidates_pool = {}
    feature_pool = input_dataframe['columns']
    for _, candidate in enumerate(feature_pool):
        spearman_r = scipy.stats.spearmanr(input_dataframe[candidate], target)
        correlation = spearman_r.correlation
        pvalue = spearman_r.pvalue
        if pvalue <= pvalue_threshhold and candidate not in drop_list:
            candidates_pool[candidate] = dict(
                pvalue=pvalue, correlation=correlation
            )

    with open('pre_selected_features.json', 'w') as json_file:
        json.dump(candidates_pool, json_file, sort_keys=True, indent=4)

    pre_selected_features = candidates_pool.keys()

    return pre_selected_features


def config_parser(conf_file_path):
    """Parse config file in YAML format"""
    with open(conf_file_path) as cfh:
        configs = yaml.load(cfh, Loader=yaml.Loader)
    return configs
