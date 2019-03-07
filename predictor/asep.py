#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main interface for asep"""

import os
import copy

from argparse import ArgumentParser

from asep.utilities import save_obj_into_pickle
from asep.utilities import TIME_STAMP
from asep.utilities import setup_xy
from asep.utilities import set_sed

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
        help="The path to configuration file"
    )

    group = parser.add_argument_group("Misc")
    group.add_argument(
        "--test-size", dest="test_size", default=None,
        help="The proportion of dataset for testing"
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

    input_file = os.path.join(
        '/home', 'umcg-zzhang', 'Documents', 'projects', 'ASEPrediction',
        'training', 'outputs', 'biosGavinOverlapCov10',
        'biosGavinOlCv10AntUfltCstBin.tsv'
    )

    mask = 'group_size < 2'
    biclass = True
    response = 'bb_ASE'
    cols_discarded = [
        "log2FC", "bn_p", "bn_p_adj", "bb_p", "bb_p_adj", "group_size", "bn_ASE"
    ]

    asep = ASEPredictor(input_file)

    set_sed()
    asep.read_file_to_dataframe()
    asep.setup_work_dataframe()

    asep.slice_dataframe(mask=mask, cols=cols_discarded)

    asep.work_dataframe[response] = asep.work_dataframe[response].apply(abs)

    asep.simple_imputer()
    asep.label_encoder()

    asep.x_matrix, asep.y_vector = setup_xy(asep.work_dataframe, y_col=response)

    asep.setup_pipeline(estimator=asep.estimators_list, biclass=biclass)
    asep.k_fold_stratified_validation(cvs=2)
    asep.training_reporter()
    asep.draw_learning_curve(asep.model, strategy="pipe")

    save_obj_into_pickle(obj=asep, file_name="train_obj")

if __name__ == '__main__':
    main()
