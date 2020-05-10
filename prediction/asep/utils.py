#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for package asep"""

import functools
import json
import time
from collections import OrderedDict
from sys import stderr as STDE

import numpy
import yaml

import scipy

DEBUG = 1


def timmer(func):
    """Print the runtime of the decorated function

    Arguments:
        func {callable} -- A function name that will be decorated

    Returns:
        Unknown -- Any object return by the decorated function
    """
    @functools.wraps(func)
    def wrapper_timmer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        func_name = func.__name__ + " "
        used_time = time.perf_counter() - start_time
        print('{:.<40} DONE, time: {:.5f} secs'.format(func_name, used_time), file=STDE)
        return value

    return wrapper_timmer


def my_debug(level=0):
    """A debug decorator function"""
    def decorator_debug(func):
        @functools.wraps(func)
        def wrapper_debug(*args, **kwargs):
            value = ""
            if level:
                value = func(*args, **kwargs)
            else:
                print("[debug] Skipping " + func.__name__)
            return value
        return wrapper_debug
    return decorator_debug


def format_print(title, main_content, pipe=STDE):
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
        numpy.random.RandomState(sed)
    else:
        numpy.random.RandomState(3142)


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


def config_parser(conf_file_path="./configs.yaml"):
    """Parse config file in YAML format"""
    with open(conf_file_path) as cfh:
        configs = yaml.safe_load(cfh)
    return configs


@my_debug(DEBUG)
def print_header(title=None, version=None, author=None, email=None,
                 institute=None, url=None):
    """A function to print a header including information of the package"""
    astr = "{: ^80}\n"
    bstr = "#{: ^48}#"
    head = astr.format("#" * 50)

    if title is None:
        title = "Allele-Specific Expression Predictor"
    head += astr.format(bstr.format(title))
    if version is None:
        version = 'Version 0.1.0'
    head += astr.format(bstr.format(version))
    if author is None:
        author = 'Zhen-hua Zhang'
    head += astr.format(bstr.format(author))
    if email is None:
        email = 'zhenhua.zhang217@gmail.com'
    head += astr.format(bstr.format(email))
    if institute is None:
        head += astr.format(bstr.format('Genomics Coordination Centre'))
        head += astr.format(bstr.format("University Medical Centre Groningen"))
    elif isinstance(institute, (tuple, list)):
        for i in institute:
            head += astr.format(bstr.format(i))
    if url is None:
        url = 'https://github.com/zhenhua-zhang/asep'
    head += astr.format(bstr.format(url))

    head += astr.format("#" * 50)
    print("\n", head, file=STDE, sep="")


@my_debug(DEBUG)
def print_flag(subc=None, flag=None):
    """A method to print running flags

    Keyword Arguments:
        subc {String} -- Name of subcommand (default: {None})
        flag {String} -- Any strings used as a flag of current run (default: {None})
    """

    if subc and flag:
        run_flag = "".join([" Subcommand: ", subc, ". Run flag: ", flag, " "])
    elif subc:
        run_flag = "".join([" Subcommand: ", subc, " "])
    elif flag:
        run_flag = "".join([" Run flag: ", flag, " "])
    else:
        run_flag = "-" * 80

    print("{:-^80}".format(run_flag), file=STDE)


@my_debug(DEBUG)
def print_args(args, fwd=-1):
    """Print arguments form command lines

    Arguments:
        args {NameSpace} -- A NameSpace containing commandline arguments

    Keyword Arguments:
        fwd {int} -- The number of space used to fill the argument and parameter, using an optimized fill-width if it's default (default: {-1})
    """

    print("Arguments for current run: ", file=STDE)
    args_pair = [(_d, _a) for _d, _a in vars(args).items()]
    args_pair = sorted(args_pair, key=lambda x: len(x[0]))

    if fwd == -1:
        fwd = len(args_pair[-1][0]) + 1

    for dst, arg in args_pair:
        print("  {d: <{w}}: {a: <{w}}".format(d=dst, a=str(arg), w=fwd), file=STDE)


def get_default_config():
    """Retrun the default configurations for the command line interface"""
    _configs = [
        "# Config for asep.py",
        "global:",
        "    run_flag: 'new_task_from_configs_file'  # No space is allowed",
        "    # config_file: configs.yaml  # The configuration file it self",
        "    # config_first: false  # Using configurations from config_file at high priority",
        "    # subcmd: train # the default subcommand [train, predict, validate, config]",
        "",
        "train: # Configs for `train` subcommand",
        "",
        "# Input",
        "    input_file: ''",
        "",
        "# Filter",
        "    first_k_rows: null",
        "    mask_as: null",
        "    mask_out: null",
        "    min_group_size: 2",
        "    max_group_size: 1.0E+5  # When using scientific notation, do NOT foget the decimal",
        "    max_na_ratio: 0.6",
        "",
        "## Which columns should be abundant manually: ('pLI_score', 'gnomAD_AF', 'EncExp', 'GerpN')",
        "    drop_cols: ['bb_p', 'bb_p_adj', 'bn_ASE', 'bn_p', 'bn_p_adj', 'group_size', 'log2FC', 'Chrom', 'Pos', 'Ref', 'Alt', 'CCDS', 'Exon', 'FeatureID', 'GeneID', 'GeneName', 'Intron', 'motifEName']",
        "    response_col: 'bb_ASE'",
        "",
        "# Configurations",
        "    random_sed: 1234",
        "    # test_size: null",
        "    classifier: 'gbc'  # Choices: abc (), gbc(gradient boosting classifier)",
        "    nested_cv: false",
        "    inner_cvs: 6",
        "    inner_n_jobs: 5",
        "    inner_n_iters: 50",
        "    outer_cvs: 6",
        "    with_learning_curve: false",
        "    learning_curve_cvs: 4",
        "    learning_curve_n_jobs: 5",
        "    learning_curve_space_size: 10",
        "",
        "# Output",
        "    output_dir: './'",
        "",
        "## How to save the model and other data to the disk. Options: [pickle, joblib]",
        "    save_method: 'pickle'  # Library used to save the model and other data set",
        "",
        "# Configs for `validate` subcommand",
        "validate:",
        "# Input",
        "    model_file: ''",
        "",
        "# Filter",
        "    first_k_rows: 0",
        "    drop_cols: null",
        "    response_col: 'bb_ASE'",
        "",
        "# Output",
        "    output_dir: './'",
        "",
        "predict:  # Configs for `predict`",
        "# Input",
        "    input_file: ''",
        "    model_file: ''",
        "",
        "# Filter",
        "    first_k_rows: 0",
        "    drop_cols: null",
        "",
        "# Output",
        "## Normally the same to the model file",
        "    output_dir: null",
        "",
        "# Configs to generate or check the sanity of given config file",
        "config:",
        "## The name of output config file",
        "    output_file: 'configs.yaml'",
        "",
        "## Only useful when you want to check the sanity of given config file",
        "    config_file: null",
        "    # overwrite: -1  # -1, 0, 1 ask, keep, overwrite",
    ]
    return "\n".join(_configs)

def dump_default_config_to_yaml(file_name="configs.yaml"):
    """Dump default configurations to configs.yaml"""
    default_configs = get_default_config()
    with open(file_name, 'w') as opt:
        opt.write(default_configs)


class NameSpace:
    """A local NameSpace class"""
    def __init__(self, name="NameSpace"):
        pass

    def add(self, key, val):
        """Add member for NameSpace"""
        setattr(self, key, val)

    def update(self, key, val):
        """Update member for NameSpace instance"""
        if hasattr(self, key):
            setattr(self, key, val)
        else:
            self.add(key, val)

    def remove(self, key):
        """Remove given memeber"""
        if hasattr(self, key):
            delattr(self, key)
        else:
            print("Unknown attr: {}".format(key), STDE)

    def to_dict(self, ordered=False):
        """Convert NameSpace instance into a dict object which could be ordered"""
        if ordered:
            return OrderedDict(vars(self))
        return vars(self)

    def from_dict(self, my_dict: dict):
        """Create an NameSpace instance from a dict object"""
        for key, val in my_dict.items():
            self.add(key, val)

# TODO: 目前还没找到好的解决方法,以后再试
    def clear(self):
        """Clear all attributes"""
        print("Not yet implemented", file=STDE)
#        keys = copy.deepcopy(self.to_dict().keys())
#        for key in keys:
#            self.remove(key)
