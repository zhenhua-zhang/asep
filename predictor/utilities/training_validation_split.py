#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A tiny script to split dataset into training and validation"""

import os
import sys
import copy
from optparse import OptionParser
from optparse import OptionGroup
import pandas

def option_parser():
    usage = "usage: %prog [options] arg1 arg2"
    parser = OptionParser(usage)
    group = OptionGroup(parser, "Input")
    group.add_option(
        "-i", "--input-file", dest="input_file", default=None,
        help="Path to the input file(including file name)"
    )
    parser.add_option_group(group)

    group = OptionGroup(parser, "Output")
    group.add_option(
        "-o", "--output-file", dest="output_file", default=None,
        help="Path to the output file(including file name)"
    )
    parser.add_option_group(group)

    return parser

def split_data_frame(ipdf, filter=None, cols=None, rows=None, keep=True,
                     random_seed=None):
    """Split data frame

    The input DataFrame will be deep-copy by copy.deepcopy() method.

    Args:
        filter (None or str): optional; default None
            A query string for the pandas.query() method
        cols (None, tuple, or list): optional; default None
            An iterable object including columns names.
        rows (None, tuple, or list): optional; default None
            An iterable object including row names.
        keep (bool): optional; default True
            Keep cells fulfill conditions or not.
        random_seed (None or int): optional; default None
            Do a random split based on the seed, seed will be set by 
            numpy.random.seed. If the it's given, a random will do split.
        sample (int, )
    """
    data_frame = copy.deepcopy(ipdf)

    if isinstance(filter, str):
        data_frame = data_frame.query(filter)

    if keep:
        data_frame = data_frame.loc[rows, cols]
        if cols is None and rows is None:  # Will drop all
            return 0
        elif rows is None:  # Keep cols
            data_frame = data_frame.loc[:, cols]
        elif cols is None:  # Keep rows
            data_frame = data_frame.loc[rows, :]
        else:
            data_frame = data_frame.loc[rows, cols]
    else:
        data_frame = data_frame.drop(index=rows, columns=cols)

    if random_seed is not None:
        numpy.random.sed(random_seed)
        data_frame = data_frame.sample()

    return 0


def main():
    parser = option_parser()
    opts, args = parser.parse_args()
    input_file = opts.input_file
    output_file = opts.output_file

    if  input_file is None:
        print("-i/--input-file CANNOT be empty", file=sys.stderr)
        parser.print_help()
        return 1
    else:
        input_dir_name, input_base_name = os.path.split(input_file)
        input_file_name, input_file_ext = os.path.splitext(input_base_name)

    if  output_file is None:
        output_file_ext = input_file_ext
        output_file_name =  input_file_name
    else:
        output_dir_name, output_base_name = os.path.split(output_file)
        output_file_name, output_file_ext = os.path.splitext(output_base_name)

    training_output = output_file_name + "_training" + output_file_ext
    validation_output = output_file_name + "_validation" + output_file_ext


if  __name__ == "__main__":
    main()
