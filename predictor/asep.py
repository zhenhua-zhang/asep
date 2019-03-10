#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main interface for asep"""

import os
import sys
from argparse import ArgumentParser
from asep.predictor import ASEPredictor


def get_args():
    """A method to get arguments from the command line"""

    __head = """
        ##############################################\n
        #    Allele-specific expression predictor    #\n
        #                                            #\n
        #  Zhenhua Zhang <zhenhua.zhang217@gmai.com  #\n
        ##############################################\n
    """

    __foot = """
        GitHub: https://github.com/zhenhua-zhang/asep
    """

    parser = ArgumentParser()

    group = parser.add_argument_group("Input")
    group.add_argument(
        "-i", "--input-file", dest="input_file", default=None, required=True,
        help="The path to file of training dataset"
    )
    group.add_argument(
        "-v", "--validation-file", dest="validation_file", default=None,
        help="The path to file of validation dataset"
    )

    group = parser.add_argument_group("Filter")
    group.add_argument(
        "-f", "--first-k-rows", dest="first_k_rows", default=None,
        help="Only read first k rows as input from input file"
    )
    group.add_argument(
        "-m", "--mask", dest="mask", default=None,
        help="Pattern will be masked or kept"
    )
    group.add_argument(
        "-s", "--group-size", dest="group_size", default=5,
        help="The least number of individuals bearing the same variant"
    )
    group.add_argument(
        "-S", "--skip-column", dest="skip_columns", default=None,
        help="""The columns will be skipped. Seperated by semi-colon and quote
        them by ','. if there are more than one columns."""
    )
    group.add_argument(
        "-t", "--target-col", dest="target_col", default=None,
        help="The column name of response variable or target variable"
    )

    group = parser.add_argument_group("Output")
    group.add_argument(
        "-o", "--output-dir", dest="output_dir", default='./',
        help="The directory including output files. Default: ./"
    )

    group = parser.add_argument_group("Configuration")
    group.add_argument(
        "-c", "--config-file", dest="config_file", default=None,
        help="""The path to configuration file, all configuration will be get
        from it, and overwrite values from command line except -i"""
    )
    group.add_argument(
        "-C", "--cross_validations", dest="cross_validations", default=8,
        help="How many folds of cross-validation will be done"
    )

    group = parser.add_argument_group("Misc")
    group.add_argument(
        "--test-size", dest="test_size", default=None,
        help="the proportion of dataset for testing"
    )
    group.add_argument(
        "--run-flag", dest="run_flag", default=None,
        help="Flags for current run"
    )

    return parser


def main():
    """Main function to run the module """
    parser = get_args()
    arguments = parser.parse_args()

    # config_file = arguments.config_file
    cross_validations = arguments.cross_validations
    # first_k_rows = arguments.first_k_rows
    group_size = arguments.group_size
    input_file = arguments.input_file
    # mask = arguments.mask
    output_dir = arguments.output_dir
    # run_flag = arguments.run_flag
    # skip_columns = arguments.skip_columns
    # target_col = arguments.target_col
    # test_size = arguments.test_size
    # validation_file = arguments.validation_file

#    /home/umcg-zzhang/Documents/projects/ASEPrediction/training/outputs/biosGavinOverlapCov10/biosGavinOlCv10AntUfltCstBin.tsv
    asep = ASEPredictor(input_file)

    mask = 'group_size < {}'.format(group_size)

    # Use Beta-Binomial
    response = 'bb_ASE' # target_col
    trim = [
        "log2FC", "bn_p", "bn_p_adj", "bb_p", "bb_p_adj", "group_size", "bn_ASE"
    ]
    asep.run(mask=mask, trim_cols=trim, response=response, cvs_=cross_validations)
    asep.save_to(output_dir)

    # Use Bionmial
    response = 'bn_ASE'
    trim = [
        "log2FC", "bn_p", "bn_p_adj", "bb_p", "bb_p_adj", "group_size", "bb_ASE"
    ]
    asep.run(mask=mask, trim_cols=trim, response=response, cvs_=cross_validations)
    asep.save_to(output_dir)

if __name__ == '__main__':
    main()
