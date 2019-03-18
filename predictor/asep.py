#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main interface for asep"""

from argparse import ArgumentParser

from sklearn.model_selection import StratifiedKFold

from asep.predictor import ASEPredictor
from asep.configs import Config


def get_args():
    """A method to get arguments from the command line"""

    __head = """
        ##############################################
        #    Allele-specific expression predictor    #
        #                                            #
        # Zhenhua Zhang <zhenhua.zhang217@gmail.com> #
        ##############################################
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
        "-f", "--first-k-rows", dest="first_k_rows", default=None, type=int,
        help="Only read first k rows as input from input file"
    )
    group.add_argument(
        "-m", "--mask", dest="mask", default=None, type=str,
        help="Pattern will be masked or kept"
    )
    group.add_argument(
        "--min-group-size", dest="min_group_size", default=2,
        type=lambda x: int(x) > 1 and int(x) or parser.error(
            "--min-group-size must be >= 2"),
        help="The minimum number of individuals bearing the same variant (>= 2)"
    )
    group.add_argument(
        "--max-group-size", dest="max_group_size", default=None,
        type=lambda x: int(x) <= 1e4  and int(x) or parser.error(
            "--max-group-size must be <= 10,000"),
        help="""The maximum number of individuals bearing the same variant
        (<= 10,000)"""
    )
    group.add_argument(
        "--drop-cols", dest="drop_cols", default=None,
        help="""The columns will be dropped. Seperated by semi-colon and quote
        them by ','. if there are more than one columns."""
    )
    group.add_argument(
        "--response-col", dest="reponse_col", default='bb_ASE',
        # XXX: Default must be consistent with input dataset
        help="The column name of response variable or target variable"
    )

    group = parser.add_argument_group("Output")
    group.add_argument(
        "-o", "--output-dir", dest="output_dir", default='./', type=str,
        help="The directory including output files. Default: ./"
    )
    group.add_argument(
        "--with-learning-curve", dest="with_lc", default=False,
        action='store_true',
        help="Whether to draw a learning curve. Default: False"
    )

    group = parser.add_argument_group("Configuration")
    group.add_argument(
        "-c", "--config-file", dest="config_file", default=None, type=str,
        help="""The path to configuration file, all configuration will be get
        from it, and overwrite values from command line except -i"""
    )
    group.add_argument(
        "--resampling-method", dest="resampling_method", default=None,
        help="Use nested cross validation or not. Default: False"
    )
    group.add_argument(
        "--nested-cv", dest="nested_cv", default=False, action="store_true",
        help="Use nested cross validation or not. Default: False"
    )
    group.add_argument(
        "--inner-cvs", dest="inner_cvs", default=6, type=int,
        help="Fold of cross-validations for RandomizedSearchCV"
    )
    group.add_argument(
        "--inner-n-jobs", dest="inner_n_jobs", default=5, type=int,
        help="Number of jobs for RandomizedSearchCV"
    )
    group.add_argument(
        "--inner-n-iters", dest="inner_n_iters", default=20, type=int,
        help="Number of iters for RandomizedSearchCV"
    )
    group.add_argument(
        "--outer-cvs", dest="outer_cvs", default=6, type=int,
        help="Fold of cross-validation for outer_validation"
    )
    group.add_argument(
        "--outer-n-jobs", dest="outer_n_jobs", default=5, type=int,
        help="Number of jobs for outer_validation"
    )
    group.add_argument(
        "--learning-curve-cvs", dest="lc_cvs", default=4, type=int,
        help="Number of folds to draw learning curve"
    )
    group.add_argument(
        "--learning-curve-n-jobs", dest="lc_n_jobs", default=5, type=int,
        help="Number of jobs to draw learning curves"
    )
    group.add_argument(
        "--learning-curve-space-size", dest="lc_space_size", default=10, type=int,
        help="Number of splits will be create in learning curve"
    )

    group = parser.add_argument_group("Misc")
    group.add_argument(
        "--test-size", dest="test_size", default=None, type=int,
        help="the proportion of dataset for testing"
    )
    group.add_argument(
        "--verbose", dest="verbose_level", default=0, action="count",
        help="the proportion of dataset for testing"
    )
    group.add_argument(
        "--run-flag", dest="run_flag", default="New_task",
        help="Flags for current run"
    )

    return parser


def main():
    """Main function to run the module """
    parser = get_args()
    arguments = parser.parse_args()

    # config_file = arguments.config_file
    # test_size = arguments.test_size
    # validation_file = arguments.validation_file

    # /home/umcg-zzhang/Documents/projects/ASEPrediction/training/outputs/biosGavinOverlapCov10/biosGavinOlCv10AntUfltCstBin.tsv

    my_config = Config()
    inner_cvs = arguments.inner_cvs
    if inner_cvs != 6:
        my_config.optim_params['cv'] = StratifiedKFold(
            n_splits=inner_cvs, shuffle=True
        )
    else:
        pass

    inner_n_jobs = arguments.inner_n_jobs
    if inner_n_jobs != 5:
        my_config.optim_params['n_jobs'] = inner_n_jobs
    else:
        pass

    inner_n_iters = arguments.inner_n_iters
    if inner_n_iters != 25:
        my_config.optim_params['n_iter'] = inner_n_iters
    else:
        pass

    input_file = arguments.input_file
    asep = ASEPredictor(input_file, my_config)

    run_flag = arguments.run_flag
    print("{:-^80}".format(run_flag))

    drop_cols = arguments.drop_cols
    if drop_cols is None:
        drop_cols = [
            "log2FC", "bn_p", "bn_p_adj", "bb_p", "bb_p_adj", "group_size",
            "bn_ASE", "CaddChrom", "CaddPos", "CaddRef", "CaddAlt", "GeneName",
            "GeneID", "FeatureID", "chr", "pos"
        ]
    else:
        pass

    min_group_size = arguments.min_group_size
    max_group_size = arguments.max_group_size
    if min_group_size < 2:
        raise "--min-group-size must be at least 2"
    else:
        pass

    mask = arguments.mask
    first_k_rows = arguments.first_k_rows
    reponse_col = arguments.reponse_col

    nested_cv = arguments.nested_cv
    outer_cvs = arguments.outer_cvs
    outer_n_jobs = arguments.outer_n_jobs

    with_lc = arguments.with_lc
    lc_space_size = arguments.lc_space_size
    lc_n_jobs = arguments.lc_n_jobs
    lc_cvs = arguments.lc_cvs

    resampling_method = arguments.resampling_method

    asep.run(
        mask=mask, mings=min_group_size, maxgs=max_group_size,
        limit=first_k_rows, response=reponse_col, drop_cols=drop_cols,
        outer_cvs=outer_cvs, outer_n_jobs=outer_n_jobs, nested_cv=nested_cv,
        with_lc=with_lc, lc_space_size=lc_space_size, lc_n_jobs=lc_n_jobs,
        lc_cvs=lc_cvs, resampling_method=resampling_method
    )

    output_dir = arguments.output_dir
    asep.save_to(output_dir)

    print("{:-^80}".format(run_flag))

if __name__ == '__main__':
    main()
