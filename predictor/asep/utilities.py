#!./env/bin/python
# -*- coding: utf-8 -*-
"""Utilities for package asep"""

import functools
import pickle
import time
import json
import sys
import os

import numpy
import scipy

try:
    from matplotlib import pyplot
except ImportError as err:
    print(err, file=sys.stderr)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot

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


def draw_k_main_features_cv(feature_pool, first_k=20):
    """Draw feature importance for the model with cross-validation"""
    name_mean_std_pool = []
    for name, importances in feature_pool.items():
        mean = numpy.mean(importances)
        std = numpy.std(importances, ddof=1)
        name_mean_std_pool.append([name, mean, std])

    name_mean_std_pool = sorted(name_mean_std_pool, key=lambda x: -x[1])

    name_pool, mean_pool, std_pool = [], [], []
    for name, mean, std in name_mean_std_pool[:first_k]:
        name_pool.append(name)
        mean_pool.append(mean)
        std_pool.append(std)

    fig, ax_features = pyplot.subplots(figsize=(10, 10))
    ax_features.bar(name_pool, mean_pool, yerr=std_pool)
    ax_features.set_xticklabels(
        name_pool, rotation_mode='anchor', rotation=45,
        horizontalalignment='right'
    )
    ax_features.set(
        title="Feature importances(with stand deviation as error bar)",
        xlabel='Feature name', ylabel='Importance'
    )

    prefix = 'feature_importance_random_'
    fig.savefig(make_file_name(prefix=prefix, suffix='png'))


def draw_roc_curve_cv(auc_fpr_tpr_pool):
    """Draw ROC curve with cross-validation"""
    fig, ax_roc = pyplot.subplots(figsize=(10, 10))
    auc_pool, fpr_pool, tpr_pool = [], [], []
    space_len = 0
    for auc_area, fpr, tpr in auc_fpr_tpr_pool:
        auc_pool.append(auc_area)
        fpr_pool.append(fpr)
        tpr_pool.append(tpr)

        if len(fpr) > space_len:
            space_len = len(fpr)

    lspace = numpy.linspace(0, 1, space_len)
    interp_fpr_pool, interp_tpr_pool = [], []
    for fpr, tpr in zip(fpr_pool, tpr_pool):
        fpr_interped = scipy.interp(lspace, fpr, fpr)
        fpr_interped[0], fpr_interped[-1] = 0, 1
        interp_fpr_pool.append(fpr_interped)

        tpr_interped = scipy.interp(lspace, fpr, tpr)
        tpr_interped[0], tpr_interped[-1] = 0, 1
        interp_tpr_pool.append(tpr_interped)

    for fpr, tpr in zip(interp_fpr_pool, interp_tpr_pool):
        ax_roc.plot(fpr, tpr, lw=0.5)

    fpr_mean = numpy.mean(interp_fpr_pool, axis=0)
    tpr_mean = numpy.mean(interp_tpr_pool, axis=0)
    tpr_std = numpy.std(interp_tpr_pool, axis=0)

    # A 95% confidence interval for the mean of AUC by Bayesian mvs
    mean, *_ = scipy.stats.bayes_mvs(auc_pool)
    auc_mean, (auc_min, auc_max) = mean.statistic, mean.minmax

    ax_roc.plot(
        fpr_mean, tpr_mean, color="r", lw=2,
        label="Mean: AUC={:0.3}, [{:0.3}, {:0.3}]".format(
            auc_mean, auc_min, auc_max
        )
    )

    mean_upper = numpy.minimum(tpr_mean + tpr_std, 1)
    mean_lower = numpy.maximum(tpr_mean - tpr_std, 0)
    ax_roc.fill_between(
        fpr_mean, mean_upper, mean_lower, color='green', alpha=0.1,
        label="Standard deviation"
    )
    ax_roc.set(
        title="ROC curve",
        xlabel='False positive rate', ylabel='True positive rate'
    )
    ax_roc.plot([0, 1], color='grey', linestyle='--')
    ax_roc.legend(loc="best")

    prefix = 'roc_curve_cv'
    fig.savefig(make_file_name(prefix=prefix, suffix='png'))


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

def setup_xy(dataframe, x_cols=None, y_col=None):
    """Set up predictor variables and target variables.

    Args:
        x_cols(list, tuple, None):
        y_col(string, None):
    Returns: DataFrame
    Raises:
        ValueError:
    """
    cols = dataframe.columns
    if x_cols is None and y_col is None:
        x_cols, y_col = cols[:-1], cols[-1]
    elif x_cols is None:
        x_cols = cols.drop(y_col)
    elif y_col is None:
        y_col = cols[-1]
        if y_col in x_cols:
            raise ValueError('Target column is in predictor columns')

    x_matrix = copy.deepcopy(dataframe.loc[:, x_cols])
    y_vector = copy.deepcopy(dataframe.loc[:, y_col])
    return (x_matrix, y_vector)

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
