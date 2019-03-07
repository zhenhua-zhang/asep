#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main interface for asep"""

import os
from argparse import ArgumentParser
from asep.predictor import ASEPredictor


def get_args():
    """A method to get arguments from the command line"""

    head = """
        ##############################################\n
        #    Allele-specific expression predictor    #\n
        #                                            #\n
        #  Zhenhua Zhang <zhenhua.zhang217@gmai.com  #\n
        ##############################################\n
    """

    foot = """
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
        "-s", "--group-size", dest="group_size", default=None,
        help="The least number of individuals bearing the same variant"
    )
    group.add_argument(
        "-S", "--skip-column", dest="skip_columns", default=None,
        help=" ".join(
            [
                "The columns will be skipped.",
                "Seperated by semi-colon and quote them by \",",
                "if there are more than one columns."
            ]
        )
    )
    group.add_argument(
        "-t", "--target-col", dest="target_col", default=None,
        help="The column name of response variable or target variable"
    )

    group = parser.add_argument_group("Output")
    group.add_argument(
        "-o", "--output-dir", dest="output_dir", default=None,
        help="The directory including output files"
    )

    group = parser.add_argument_group("Configuration")
    group.add_argument(
        "-c", "--config-file", dest="config_file", default=None,
        help="""The path to configuration file, all configuration will be get
        from it, and overwrite values from command line except -i"""
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
    arguments = vars(parser.parse_args())

    # validation_file = arguments.validation_file
    # skip_columns = arguments.skip_columns
    # config_file = arguments.config_file
    # group_size = arguments.group_size
    # input_file = arguments.input_file
    # output_dir = arguments.output_dir
    # target_col = arguments.target_col
    # test_size = arguments.test_size
    # run_flag = arguments.run_flag

    input_file = os.path.join(
        '/home', 'umcg-zzhang', 'Documents', 'projects', 'ASEPrediction',
        'training', 'outputs', 'biosGavinOverlapCov10',
        'biosGavinOlCv10AntUfltCstBin.tsv'
    )
    asep = ASEPredictor(input_file)

    MASK = 'group_size < 2'

    # Use Beta-Binomial
    RESPONSE = 'bb_ASE'
    TRIM = [
        "log2FC", "bn_p", "bn_p_adj", "bb_p", "bb_p_adj", "group_size", "bn_ASE"
    ]
    asep.run(limit=600, mask=MASK, trim_cols=TRIM, response=RESPONSE, cvs_=2)
    asep.save_to()

    # Use Bionmial
    RESPONSE = 'bn_ASE'
    TRIM = [
        "log2FC", "bn_p", "bn_p_adj", "bb_p", "bb_p_adj", "group_size", "bb_ASE"
    ]
    asep.run(limit=600, mask=MASK, trim_cols=TRIM, response=RESPONSE, cvs_=2)
    asep.save_to()

if __name__ == '__main__':
    main()
