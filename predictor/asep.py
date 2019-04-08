#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main interface for asep"""
import sys
import pickle

from argparse import ArgumentParser

from asep.predictor import ASEPredictor
from asep.utilities import timmer
from asep.configs import Config


def get_args():
    """A method to get arguments from the command line"""

    parser = ArgumentParser()
    _group = parser.add_argument_group("Global")
    _group.add_argument(
        "-V", dest="verbose_level", action="count", help="Verbose level"
    )
    _group.add_argument(
        "--run-flag", dest="run_flag", default="new_task",
        help="Flags for current run"
    )

    subparser = parser.add_subparsers(dest="subcommand")

    # Argument parser for subcommand `train`
    train_argparser = subparser.add_parser("train", help="Train a model")
    _group = train_argparser.add_argument_group("Input")
    _group.add_argument(
        "-i", "--train-input-file", dest="input_file", default=None, required=True,
        help="The path to file of training dataset. Default: None"
    )

    _group = train_argparser.add_argument_group("Filter")
    _group.add_argument(
        "-f", "--first-k-rows", dest="first_k_rows", default=None, type=int,
        help="Only read first k rows as input from input file. Default: None"
    )
    _group.add_argument(
        "-m", "--mask-as", dest="mask_as", default=None, type=str,
        help="Pattern will be kept. Default: None"
    )
    _group.add_argument(
        "-M", "--mask-out", dest="mask_out", default=None, type=str,
        help="Pattern will be masked. Default: None"
    )
    _group.add_argument(
        "--min-group-size", dest="min_group_size", default=2,
        type=lambda x: int(x) > 1 and int(x) or parser.error(
            "--min-group-size must be >= 2"),
        help="""The minimum of individuals bearing the same variant (>= 2).
        Default: 2"""
    )
    _group.add_argument(
        "--max-group-size", dest="max_group_size", default=None,
        type=lambda x: int(x) <= 1e4  and int(x) or parser.error(
            "--max-group-size must be <= 10,000"),
        help="""The maximum number of individuals bearing the same variant
        (<= 10,000). Default: None"""
    )
    _group.add_argument(
        "--drop-cols", dest="drop_cols", default=None, nargs='+',
        help="""The columns will be dropped. Seperated by semi-colon and quote
        them by ','. if there are more than one columns. Default: None"""
    )
    _group.add_argument(
        "--response-col", dest="reponse_col", default='bb_ASE',
        help="""The column name of response variable or target variable.
        Default: bb_ASE"""
    )

    _group = train_argparser.add_argument_group("Configuration")
    _group.add_argument(
        "--test-size", dest="test_size", default=None, type=int,
        help="The proportion of dataset for testing. Default: None"
    )
    _group.add_argument(
        "-c", "--config-file", dest="config_file", default=None, type=str,
        help="""The path to configuration file, all configuration will be get
        from it, and overwrite values from command line except -i. Default:
        None"""
    )
    _group.add_argument(
        "--classifier", dest="classifier", default='rfc', type=str,
        choices=["abc", "gbc", "rfc", "brfc"],
        help="Algorithm. Choices: [abc, gbc, rfc, brfc]. Default: rfc"
    )
    _group.add_argument(
        "--resampling", dest="resampling", action="store_true",
        help="Use resampling method or not. Default: False"
    )
    _group.add_argument(
        "--nested-cv", dest="nested_cv", default=False, action="store_true",
        help="Use nested cross validation or not. Default: False"
    )
    _group.add_argument(
        "--inner-cvs", dest="inner_cvs", default=6, type=int,
        help="Fold of cross-validations for RandomizedSearchCV. Default: 6"
    )
    _group.add_argument(
        "--inner-n-jobs", dest="inner_n_jobs", default=5, type=int,
        help="Number of jobs for RandomizedSearchCV, Default: 5"
    )
    _group.add_argument(
        "--inner-n-iters", dest="inner_n_iters", default=50, type=int,
        help="Number of iters for RandomizedSearchCV. Default: 50"
    )
    _group.add_argument(
        "--outer-cvs", dest="outer_cvs", default=6, type=int,
        help="Fold of cross-validation for outer_validation"
    )
    _group.add_argument(
        "--outer-n-jobs", dest="outer_n_jobs", default=5, type=int,
        help="Number of jobs for outer_validation"
    )
    _group.add_argument(
        "--with-learning-curve", dest="with_lc", default=False,
        action='store_true', help="Whether draw learning curve. Default: False"
    )
    _group.add_argument(
        "--learning-curve-cvs", dest="lc_cvs", default=4, type=int,
        help="Number of folds to draw learning curve"
    )
    _group.add_argument(
        "--learning-curve-n-jobs", dest="lc_n_jobs", default=5, type=int,
        help="Number of jobs to draw learning curves"
    )
    _group.add_argument(
        "--learning-curve-space-size", dest="lc_space_size", default=10,
        type=int, help="Number of splits will be create in learning curve"
    )

    _group = train_argparser.add_argument_group("Output")
    _group.add_argument(
        "-o", "--output-dir", dest="output_dir", default='./', type=str,
        help="The directory including output files. Default: ./"
    )

    # Argument parser for subcommand `validate`
    validate_argparser = subparser.add_parser(
        "validate", help="Validate the model."
    )
    _group = validate_argparser.add_argument_group("Input")
    _group.add_argument(
        "-v", "--validation-file", dest="validation_file", type=str,
        required=True, help="The path to file of validation dataset"
    )

    # Argument parser for subcommand `predict`
    predict_argparser = subparser.add_parser(
        "predict", help="Predict new dataset by the trained model"
    )
    _group = predict_argparser.add_argument_group("Input")
    _group.add_argument(
        "-i", "--predict-input-file", dest="input_file", type=str,
        required=True, help="New files including case to be predicted"
    )
    _group.add_argument(
        "-m", "--model-file", dest="model_file", type=str, required=True,
        help="Model to be used"
    )

    _group = predict_argparser.add_argument_group("Output")
    _group.add_argument(
        "-o", "--predict-output-dir", dest="output_dir", type=str,
        required=True, help="Output directory for predict input file"
    )

    return parser

@timmer
def train(arguments):
    """Train the model"""
    my_config = Config()

    inner_cvs = arguments.inner_cvs
    inner_n_jobs = arguments.inner_n_jobs
    inner_n_iters = arguments.inner_n_iters
    my_config.set_searcher_params(
        n_jobs=inner_n_jobs, n_iter=inner_n_iters, ncvs=inner_cvs
    )

    classifier = arguments.classifier
    my_config.set_classifier(classifier)

    my_config.assembly()

    input_file = arguments.input_file
    asep = ASEPredictor(input_file, my_config)

    drop_cols = arguments.drop_cols
    if drop_cols is None:
        drop_cols = [
            "log2FC", "bn_p", "bn_p_adj", "bb_p", "bb_p_adj", "group_size",
            "bn_ASE", "CaddChrom", "CaddPos", "CaddRef", "CaddAlt", "GeneName",
            "GeneID", "FeatureID", "chr", "pos", "gene", "Intron", "Exon",
            "CCDS", "motifEName"
        ]

    min_group_size = arguments.min_group_size
    max_group_size = arguments.max_group_size
    mask_as = arguments.mask_as
    mask_out = arguments.mask_out
    reponse_col = arguments.reponse_col
    first_k_rows = arguments.first_k_rows

    nested_cv = arguments.nested_cv
    outer_cvs = arguments.outer_cvs
    outer_n_jobs = arguments.outer_n_jobs

    with_lc = arguments.with_lc
    lc_space_size = arguments.lc_space_size
    lc_n_jobs = arguments.lc_n_jobs
    lc_cvs = arguments.lc_cvs

    resampling = arguments.resampling

    asep.trainer(
        mask=mask_out, mings=min_group_size, maxgs=max_group_size,
        limit=first_k_rows, response=reponse_col, drop_cols=drop_cols,
        outer_cvs=outer_cvs, outer_n_jobs=outer_n_jobs, nested_cv=nested_cv,
        with_lc=with_lc, lc_space_size=lc_space_size, lc_n_jobs=lc_n_jobs,
        lc_cvs=lc_cvs, resampling=resampling
    )

    run_flag = arguments.run_flag
    output_dir = arguments.output_dir
    asep.save_to(output_dir, run_flag=run_flag)


@timmer
def validate(arguments):
    """Validate the model using extra dataset"""


@timmer
def predict(arguments):
    """Predict new dataset based on constructed model"""
    model_file = arguments.model_file
    with open(model_file, 'rb') as model_file_handle:
        model_obj = pickle.load(model_file_handle)

    input_file = arguments.input_file
    output_dir = arguments.output_dir 
    model_obj.predictor(input_file, output_dir=output_dir)


def main():
    """Main function to run the module """
    # /home/umcg-zzhang/Documents/projects/ASEPrediction/training/outputs/biosGavinOverlapCov10/biosGavinOlCv10AntUfltCstBin.tsv

    __head = """
              ##############################################
              #    Allele-specific expression predictor    #
              #              Version 0.0.1                 #
              # Zhenhua Zhang <zhenhua.zhang217@gmail.com> #
              ##############################################
    """

    __foot = """
              GitHub: https://github.com/zhenhua-zhang/asep
    """
    print(__head)
    print(__foot)

    parser = get_args()
    arguments = parser.parse_args()

    run_flag = arguments.run_flag
    subcommand = arguments.subcommand
    verbose_level = arguments.verbose_level

    if subcommand:
        run_flag = "".join(
            [" Subcommand: ", subcommand, ". Run flag: ", run_flag, " "]
        )
        print("{:-^80}".format(run_flag))

    if subcommand == "train":
        model_pool = train(arguments)
        return 1
    elif subcommand == "validate":
        validate(arguments)
        return 1
    elif subcommand == "predict":
        predict(arguments)
        return 1
    else:
        parser.print_help()
        return 1

    if subcommand:
        print("{:-^80}".format(run_flag))

    return 0
if __name__ == '__main__':
    main()
