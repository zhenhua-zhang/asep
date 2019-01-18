#!./env/bin/python
# -*- coding: utf-8 -*-
"""Utilities for package asep"""

import functools
import pickle
import time
import sys
import os

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
    head = '-' * 10
    tail = '-' * max(60 - len(title), 0)
    flag = ' '.join([head, title, tail])

    print(' ', main_content, "\n", flag, "\n", file=pipe)


def save_obj_into_pickle(obj, file_name):
    """Save Object instance by pickle

    Args:
        obj (object): compulsory;
        file_name (str): compulsory;
    """
    pklf_name = make_file_name(file_name, prefix='training', suffix='pkl')
    with open(pklf_name, 'wb') as pklof:
        pickle.dump(obj, pklof)


def load_obj_from_pickle(file_name):
    """Load ASEPredictor instance by pickle"""
    with open(file_name, 'wb') as pklif:
        return pickle.load(pklif)
