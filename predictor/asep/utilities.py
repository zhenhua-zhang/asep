#!./env/bin/python
# -*- coding: utf-8 -*-
"""Utilities for package asep"""

import functools
import pickle
import json
import time
import sys
import os

import numpy
import scipy

# Global variable
# time stamp
TIME_STAMP = time.strftime("%Y_%b_%d_%H_%M_%S", time.gmtime())

def make_time_stamp():
    """Setup time stamp for the package"""
    global TIME_STAMP
    TIME_STAMP = time.strftime("%Y_%b_%d_%H_%M_%S", time.gmtime())


def make_file_name(file_name=None, prefix=None, suffix=None, _time_stamp=None):
    """Create file name based on timestamp

    Args:
        file_name (str or None): optional; defualt None
            The file name, if need.
        prefix (str or None): optional; default None
            The prefix of the dumped file
        suffix (str or None): optional; default None
            The suffix of the dumped file
        _time_stamp (str or None): optional; default None
            Time stamp used in file name
    Returns:
        file_name (str):
            The created filename.
    """
    if _time_stamp is None:
        global TIME_STAMP
        _time_stamp = TIME_STAMP

    if file_name is None:
        file_name = _time_stamp
    else:
        file_name += '_' + _time_stamp

    if prefix:
        file_name = prefix + '_' + file_name

    if suffix:
        file_name += '.' + suffix

    if not os.path.exists(_time_stamp):
        os.mkdir(_time_stamp)

    return os.path.join(".", _time_stamp, file_name)


def timmer(func):
    """Print the runtime of the decorated function

    Args:
        func (callable): function to be decoreated
    """
    @functools.wraps(func)
    def wrapper_timmer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        func_name = func.__name__
        used_time = time.perf_counter() - start_time
        sys.stderr.write(
            '{} is done; elapsed: {:.5f} secs\n'.format(func_name, used_time)
        )
        return value

    return wrapper_timmer


def format_print(title, main_content, pipe=sys.stdout):
    """A method format the prints"""
    head = '-' * 3
    flag = ' '.join([head, title, ": "])
    print(flag, '\n   ', main_content, "\n\n", file=pipe)


def save_obj_into_pickle(obj, file_name):
    """Save Object instance by pickle

    Args:
        obj (object): required;
        file_name (str): required;
    """
    pklf_name = make_file_name(file_name, prefix='training', suffix='pkl')
    with open(pklf_name, 'wb') as pklof:
        pickle.dump(obj, pklof)


def load_obj_from_pickle(file_name):
    """Load ASEPredictor instance by pickle"""
    with open(file_name, 'wb') as pklif:
        return pickle.load(pklif)


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


# TODO: finish it into a standalone function
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


# Expired fucntion from asep/predictor.py
# def train_test_slicer(self, **kwargs):
    # """Set up training and testing data set by train_test_split"""
    # (self.x_train_matrix, self.x_test_matrix,
    #  self.y_train_vector, self.y_test_vector
    # ) = train_test_split(self.x_matrix, self.y_vector, **kwargs)
