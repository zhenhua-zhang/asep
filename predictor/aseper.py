#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main interface for asep

TODO: 1. A module to parse configuration file, which could make life easier.
      2. The `mask` argument in predictor.trainer() func doesn't function at all.
"""
import pickle

from argparse import ArgumentParser

from asep.config import Config
from asep.model import ASEP
from asep.utils import config_parser, my_debug, print_args, print_flag, DEBUG, print_header, NameSpace, dump_default_config_to_yaml


def cli_parser():
    """A method to get arguments from the command line

    Returns:
        parser: ArgumentParser -- An intance of ArgumentParser()
    """

    parser = ArgumentParser()
    _group = parser.add_argument_group("Global") # Global-wide configs
    _group.add_argument("--run-flag", dest="run_flag", default="new_task", help="Flags for current run. The flag will be added to the name of the output dir. Default: new_task")
    _group.add_argument("--config-file", dest="config_file", default="./configs.yaml", type=str, help="The path to configuration file, and overwrite values from command line excepting -i. Default: cofig.yaml")
    _group.add_argument("--config-first", dest="config_first", action="store_true", help="Whether using configurations from config file first or not.")

    subparser = parser.add_subparsers(dest="subcmd") # Arguments parser for sub-command `train`
    train_argparser = subparser.add_parser("train", help="Train a model")

    _group = train_argparser.add_argument_group("Input") # Arguments for Input
    _group.add_argument("-i", "--input-file", dest="input_file", default=None, help="The path to file of training dataset. [Required]")

    _group = train_argparser.add_argument_group("Filter") # Arguments for Filter
    _group.add_argument("-f", "--first-k-rows", dest="first_k_rows", default=None, type=int, help="Only read first k rows as input from input file. Default: None")
    _group.add_argument("-m", "--mask-as", dest="mask_as", default=None, type=str, help="Pattern will be kept. Default: None")
    _group.add_argument("-M", "--mask-out", dest="mask_out", default=None, type=str, help="Pattern will be masked. Default: None")
    _group.add_argument("--min-group-size", dest="min_group_size", default=2, type=lambda x: int(x) > 1 and int(x) or parser.error("--min-group-size must be >= 2"), help="The minimum individuals bearing the same variant(>=2). Default: 2")
    _group.add_argument("--max-group-size", dest="max_group_size", default=1.0E5, type=lambda x: int(x) <= 1e4 and int(x) or parser.error("--max-group-size must be <= 10,000"), help="The maximum number of individuals bearing the same variant (<= 10,000). Default: None")
    _group.add_argument("--max-na-ratio", dest="max_na_ratio", default=0.6, type=float, help="The maximum ratio of NA in each feature, otherwise, the feature will be abundant")
    _group.add_argument("--drop-cols", dest="drop_cols", default=['bb_p', 'bb_p_adj', 'bn_ASE', 'bn_p', 'bn_p_adj', 'group_size', 'log2FC', 'Chrom', 'Pos', 'Ref', 'Alt', 'CCDS', 'Exon', 'FeatureID', 'GeneID', 'GeneName', 'Intron', 'motifEName'], nargs='+', help="The columns will be dropped. Seperated by semi-colon and quote them by ','. if there are more than one columns. Default: None")
    _group.add_argument("--response-col", dest="response_col", default='bb_ASE', help="The column name of response variable or target variable. Default: bb_ASE")

    _group = train_argparser.add_argument_group("Configuration") # Arguments for configuration
    _group.add_argument("--random-sed", dest="random_sed", default=1234, type=int, help="The random seed. Default: 1234")
    _group.add_argument("--classifier", dest="classifier", default='rfc', type=str, choices=["abc", "gbc", "rfc", "brfc"], help="Algorithm. Choices: [abc, gbc, rfc, brfc]. Default: rfc")
    _group.add_argument("--nested-cv", dest="nested_cv", default=False, action="store_true", help="Use nested cross validation or not. Default: False")
    _group.add_argument("--inner-cvs", dest="inner_cvs", default=6, type=int, help="Fold of cross-validations for RandomizedSearchCV. Default: 6")
    _group.add_argument("--inner-n-jobs", dest="inner_n_jobs", default=5, type=int, help="Number of jobs for RandomizedSearchCV. Default: 5")
    _group.add_argument("--inner-n-iters", dest="inner_n_iters", default=50, type=int, help="Number of iters for RandomizedSearchCV. Default: 50")
    _group.add_argument("--outer-cvs", dest="outer_cvs", default=6, type=int, help="Fold of cross-validation for outer_validation. Default: 6")
    _group.add_argument("--with-learning-curve", dest="with_learning_curve", default=False, action='store_true', help="Whether draw learning curve. Default: False")
    _group.add_argument("--learning-curve-cvs", dest="learning_curve_cvs", default=4, type=int, help="Number of folds to draw learning curve. Default: 4")
    _group.add_argument("--learning-curve-n-jobs", dest="learning_curve_n_jobs", default=5, type=int, help="Number of jobs to draw learning curves. Default: 5")
    _group.add_argument("--learning-curve-space-size", dest="learning_curve_space_size", default=10, type=int, help="Number of splits created in learning curve. Default: 10")
    # _group.add_argument("--test-size", dest="test_size", default=None, type=int, help="The proportion of dataset for testing. Default: None")

    _group = train_argparser.add_argument_group("Output") # Arguments for Output
    _group.add_argument("-o", "--output-dir", dest="output_dir", default='./', type=str, help="The directory including output files. Default: ./")
    _group.add_argument("--save-method", dest="save_method", default="pickle", choices=["pickle", "joblib"], help="The library used to save the model and other data set. Choices: pickle, joblib. Default: pickle")

    validate_argparser = subparser.add_parser("validate", help="Validate the model.") # Argument parser for subcommand `validate`
    _group = validate_argparser.add_argument_group("Input") # Arguments for Input
    _group.add_argument("-i", "--input-file", dest="input_file", type=str, help="Path to file of validation dataset. [Required]")
    _group.add_argument("-m", "--model-file", dest="model_file", type=str, help="Model to be validated. [Required]")

    _group = validate_argparser.add_argument_group("Filter") # Arguments for Filter
    _group.add_argument("-f", "--first-k-rows", dest="first_k_rows", default=-1, type=int, help="Only read first k rows as input from input file. Default: None")
    _group.add_argument("--response-col", dest="response_col", default='bb_ASE', help="The column name of response variable or target variable. Default: bb_ASE")

    _group = validate_argparser.add_argument_group("Output") # Arguments for Output
    _group.add_argument("-o", "--output-dir", dest="output_dir", default="./", type=str, help="The directory including output files. Default: ./")

    predict_argparser = subparser.add_parser("predict", help="Apply the model on new data set") # Argument parser for subcommand `predict`
    _group = predict_argparser.add_argument_group("Input") # Arguments for Input
    _group.add_argument("-i", "--input-file", dest="input_file", type=str, required=True, help="New dataset to be predicted. [Required]")
    _group.add_argument("-m", "--model-file", dest="model_file", type=str, required=True, help="Model to be used. [Required]")

    _group = predict_argparser.add_argument_group("Filter") # Arguments for Filter
    _group.add_argument("-f", "--first-k-rows", dest="first_k_rows", default=None, type=int, help="Only read first k rows as input from input file. Default: None")

    _group = predict_argparser.add_argument_group("Output") # Arguments for Output
    _group.add_argument("-o", "--output-dir", dest="output_dir", type=str, help="Output directory for input file. [Reqired]")

    config_argparser = subparser.add_parser("config", help="Generate a config template")
    config_argparser.add_argument("-o", "--output-file", dest="output_file", type=str, default="configs.yaml", help="File path to the default configs. Default: configs.yaml")
    config_argparser.add_argument("-c", "--check-config-file", dest="check_config_file", type=str, default=None, help="Check whether the config file in current directory is default. Default: configs.yaml")

    return parser


@my_debug(level=DEBUG)
def train(args):
    """Wrapper entry for `train` subcommand

    Arguments:
        args {ArgumentParser} -- An `ArgumentParser` instance caught by `parse_arg()`
    """

    my_config = Config()

    inner_cvs = args.inner_cvs
    inner_n_jobs = args.inner_n_jobs
    inner_n_iters = args.inner_n_iters
    my_config.set_searcher_params(n_jobs=inner_n_jobs, n_iter=inner_n_iters, ncvs=inner_cvs)

    classifier = args.classifier
    my_config.set_classifier(classifier)

    my_config.assembly()

    input_file = args.input_file
    random_sed = args.random_sed
    asep = ASEP(input_file, my_config, sed=random_sed)

    drop_cols = args.drop_cols
    min_group_size = args.min_group_size
    max_group_size = args.max_group_size
    max_na_ratio = args.max_na_ratio
    _mask_as = args.mask_as
    mask_out = args.mask_out
    response_col = args.response_col
    first_k_rows = args.first_k_rows

    nested_cv = args.nested_cv
    outer_cvs = args.outer_cvs

    with_lc = args.with_learning_curve
    learning_curve_space_size = args.learning_curve_space_size
    learning_curve_n_jobs = args.learning_curve_n_jobs
    learning_curve_cvs = args.learning_curve_cvs

    # FIXME: Nonused `mask` argument
    asep.trainer(
        mask=mask_out, mings=min_group_size, maxgs=max_group_size,
        limit=first_k_rows, response=response_col, drop_cols=drop_cols,
        outer_cvs=outer_cvs, nested_cv=nested_cv, with_lc=with_lc,
        lc_space_size=learning_curve_space_size, lc_n_jobs=learning_curve_n_jobs,
        lc_cvs=learning_curve_cvs, max_na_ratio=max_na_ratio
    )

    run_flag = args.run_flag
    output_dir = args.output_dir
    save_method = args.save_method
    asep.save_to(output_dir, run_flag=run_flag, save_method=save_method)


@my_debug(level=DEBUG)
def validate(args):
    """Validate the model using extra dataset"""
    model_file = args.model_file
    with open(model_file, 'rb') as model_file_handle:
        model_obj = pickle.load(model_file_handle)

    response_col = args.response_col
    first_k_rows = args.first_k_rows
    input_file = args.input_file
    output_dir = args.output_dir
    model_obj.validator(input_file, output_dir, first_k_rows, response_col)


@my_debug(level=DEBUG)
def predict(args):
    """Predict new dataset based on constructed model"""
    model_file = args.model_file
    with open(model_file, 'rb') as model_file_handle:
        model_obj = pickle.load(model_file_handle)

    input_file = args.input_file
    output_dir = args.output_dir
    first_k_rows = args.first_k_rows
    model_obj.predictor(input_file, output_dir, first_k_rows)


@my_debug(DEBUG)
def configurate(args):
    """Manipulations on config files"""
    output_file = args.output_file
    check_config_file = args.check_config_file
    if output_file is None:
        output_file = "configs.yaml"

    if check_config_file:  # TODO: A function to check the sanity of config file
        print("Not implemented yet")
    else:
        dump_default_config_to_yaml(output_file)



def main():
    """Main function to run the module """
    parser = cli_parser()
    cli_args = parser.parse_args()
    config_first = cli_args.config_first
    config_file = cli_args.config_file
    configs = config_parser(config_file)

    glbcfgs_ns = NameSpace()
    glbcfgs_ns.from_dict(configs.get("global"))

    if config_first:
        run_flag = glbcfgs_ns.run_flag
    else:
        run_flag = cli_args.run_flag
    cli_args.run_flag = run_flag
    subcmd = cli_args.subcmd  # Not configurable by config_file

    if subcmd not in ["train", "validate", "predict", "config"]:
        parser.print_help()
    else:
        print_header()
        print_flag(subcmd, run_flag)

        cli_args = vars(cli_args)
        subcmdcfg = configs.get(subcmd)
        nmsp = NameSpace()
        if config_first:
            cli_args.update(subcmdcfg)
            nmsp.from_dict(cli_args)
        else:
            subcmdcfg.update(cli_args)
            nmsp.from_dict(subcmdcfg)

        print_args(nmsp)

        if subcmd == "train":
            train(nmsp)
        elif subcmd == "validate":
            validate(nmsp)
        elif subcmd == "predict":
            predict(nmsp)
        elif subcmd == "config":
            configurate(nmsp)

        print_flag(subcmd, run_flag)


if __name__ == '__main__':
    main()
