#!./env/bin/python
# -*- coding: utf-8 -*-
"""Main interface for asep"""

import os

from optparse import OptionParser
from optparse import OptionGroup

from asep.utilities import save_obj_into_pickle
from asep.predictor import ASEPredictor


def parse_opts():
    parser = OptionParser()
    group = OptionGroup(parser, "Input")  # Options for input
    group.add_option(
        "-i", "--input-file", dest="input_file", help="The path of input file"
    )
    parser.add_option_group(group)

    group = OptionGroup(parser, "Filter")  # Options for filters
    group.add_option(
        "-s", "--group-size", dest="group_size", 
        help="The least number of individuals bearing the same variant"
    )
    group.add_option(
        "-S", "--skip-column", dest="skip_columns",
        help=" ".join(
            [
                "The columns will be skipped.",
                "Seperated by semi-colon and quote them by \",",
                "if there are more than one columns."
            ]
        )
    )
    group.add_option(
        "-t", "--target-col", dest="target_col",
        help="The column name of response variable or target variables"
    )
    parser.add_option_group(group)

    group = OptionGroup(parser, "Training")  # Options for  training
    group.add_option(
        "--tuning-method", dest="tunning_method", default=0,
        help=" ".join(
            [
                "The strategy used to optimize hyperparameters.",
                "0 is RandomizedSearchCV().",
                "1 is GridSearchCV().",
                "2 is both."
            ]
        )
    )

    group = OptionGroup(parser, "Output")
    group.add_option(
        "-o", "--output-dir", dest="output_dir",
        help="The directory including output files"
    )
    parser.add_option_group(group)

    group = OptionGroup(parser, "Configuration")
    group.add_option(
        "-c", "--config-file", dest="config_file",
        help="The path to configuration file"
    )
    parser.add_option_group(group)

    group = OptionGroup(parser, "Misc")
    group.add_option(
        "--test-size", dest="test_size",
        help="The proportation of dataset for testing"
    )
    parser.add_option_group(group)

    return parser.parse_args()


def main():
    """Main function to run the module """
    opts, args = parse_opts()

    input_file = opts.input_file
    group_size = opts.group_size
    config_file = opts.config_file
    test_size = opts.test_size
    tunning_method = opts.tunning_method
    target_col = opts.target_col

    input_file = os.path.join(
        '/home', 'umcg-zzhang', 'Documents', 'projects', 'ASEpredictor',
        'outputs', 'biosGavinOverlapCov10',
        'biosGavinOlCv10AntUfltCstLog2FCBin.tsv'
    )

    asep = ASEPredictor(input_file)
    asep.raw_df = asep.read_file_to_dataframe()
    asep.set_seed(1234)
    asep.check_df('raw_df')
    asep.setup_work_df()
    asep.update_work_dataframe_info()

    # change into binary classification.
    # need to change setup_pipeline multi_class into False
    multiclass = False
    if not multiclass:
        asep.work_df.ASE = asep.work_df.ASE.apply(abs)

    asep.simple_imputer()
    asep.label_encoder(remove=False)

    cols_discarded = [
        'var', 'mean', 'p_value', 'gp_size',
        'mirSVR.Score', 'mirSVR.E', 'mirSVR.Aln'
    ]

    flt = 'gp_size > 5'
    asep.train_test_df = asep.slice_data_frame(
        fltout=flt, cols=cols_discarded, rows=[], keep=False
    )

    flt = 'gp_size <= 5'
    asep.validating_df = asep.slice_data_frame(
        fltout=flt, cols=cols_discarded, rows=[], keep=False
    )

    asep.x_cols, asep.y_col = asep.setup_xy(asep.train_test_df, y_col='ASE')
    asep.train_test_slicer(test_size=0.1)

    asep.x_vals, asep.y_val = asep.setup_xy(asep.validating_df, y_col='ASE')

    estimators_list = asep.estimators_list
    asep.setup_pipeline(estimators=estimators_list, multi_class=multiclass)

    random_search_opt_params = asep.random_search_opt_params
    asep.random_search_opt(asep.pipeline, **random_search_opt_params)

    asep.training_reporter()
    asep.draw_figures()

    save_obj_into_pickle(obj=asep, file_name="train_obj")


if __name__ == '__main__':
    main()
