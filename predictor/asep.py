#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main interface for asep"""
import pickle
import sys
from argparse import ArgumentParser

sys.path.append("/home/umcg-zzhang/Documents/git/asep/")
from asep.configs import Config
from asep.predictor import ASEPredictor
from asep.utilities import timmer


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

    # Argument parser for sub-command `train`
    train_argparser = subparser.add_parser("train", help="Train a model")
    _group = train_argparser.add_argument_group("Input")
    _group.add_argument(
        "-i", "--train-input-file", dest="input_file", default=None,
        required=True,
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
        help="The minimum of individuals bearing the same variant (>= 2). Default: 2"
    )
    _group.add_argument(
        "--max-group-size", dest="max_group_size", default=None,
        type=lambda x: int(x) <= 1e4 and int(x) or parser.error(
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
        from it, and overwrite values from command line except -i. Not
        implemented yet. Default: None"""
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
        help="Fold of cross-validation for outer_validation. Default: 6"
    )
    _group.add_argument(
        "--outer-n-jobs", dest="outer_n_jobs", default=5, type=int,
        help="Number of jobs for outer_validation. Default: 5"
    )
    _group.add_argument(
        "--with-learning-curve", dest="with_lc", default=False,
        action='store_true', help="Whether draw learning curve. Default: False"
    )
    _group.add_argument(
        "--learning-curve-cvs", dest="lc_cvs", default=4, type=int,
        help="Number of folds to draw learning curve. Default: 4"
    )
    _group.add_argument(
        "--learning-curve-n-jobs", dest="lc_n_jobs", default=5, type=int,
        help="Number of jobs to draw learning curves. Default: 5"
    )
    _group.add_argument(
        "--learning-curve-space-size", dest="lc_space_size", default=10,
        type=int, help="Number of splits will be create in learning curve. Default: 10"
    )
    _group.add_argument(
        "--with-rbm", dest="with_rbm", default=False, action="store_true",
        help="Whether using Reistricted Boltzmann Machine to create training components."
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

    with_rbm = arguments.with_rbm
    if with_rbm:
        my_config.set_constructor()

    my_config.assembly()

    input_file = arguments.input_file
    asep = ASEPredictor(input_file, my_config)

    drop_cols = arguments.drop_cols
    if drop_cols is None:
        drop_cols = [
            "log2FC", "bn_p", "bn_p_adj", "bb_p", "bb_p_adj", "group_size",
            "bn_ASE", "Chrom", "Pos", "Ref", "Alt", "GeneID", "FeatureID",
            "GeneName", "Intron", "Exon", "CCDS", "motifEName"
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
        version = 'Version 0.01'
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
    print(head, file=sys.stderr)


def print_flag(subc=None, flag=None):
    """A method to print runing flags"""
    if subc and flag:
        run_flag = "".join([" Subcommand: ", subc, ". Run flag: ", flag, " "])
    elif subc:
        run_flag = "".join([" Subcommand: ", subc, " "])
    elif flag:
        run_flag = "".join([" Run flag: ", flag, " "])
    else:
        run_flag = "-" * 80

    print("{:-^80}".format(run_flag), file=sys.stderr)


def print_arguments(arguments, fill_width=None):
    """Print artuments from command lines"""
    print("Arguments for current run: ", file=sys.stderr)
    arguments_dict = vars(arguments)
    arguments_pair = [(_dst, _arg) for _dst, _arg in arguments_dict.items()]
    arguments_pair = sorted(arguments_pair, key=lambda x: len(x[0]))

    if fill_width is None:
        fill_width = len(arguments_pair[-1][0]) + 1

    for _dst, _arg in arguments_pair:
        if _arg is None:
            _arg = "None"
        print(
            "{dst: <{wid}}: {arg: <{wid}}".format(dst=_dst, arg=_arg, wid=fill_width),
            file=sys.stderr
        )


def main():
    """Main function to run the module """
    parser = get_args()
    arguments = parser.parse_args()
    run_flag = arguments.run_flag
    subcommand = arguments.subcommand
    verbose_level = arguments.verbose_level

    print_header()
    print_flag(subcommand, run_flag)
    print_arguments(arguments)

    if subcommand == "train":
        train(arguments)
    elif subcommand == "validate":
        validate(arguments)
    elif subcommand == "predict":
        predict(arguments)
    else:
        parser.print_help()

    print_flag(subcommand, run_flag)


if __name__ == '__main__':
    main()


# Column names
#
# Chrom Pos Ref Alt Type Length AnnoType Consequence ConsScore ConsDetail GC
# CpG motifECount motifEName motifEHIPos motifEScoreChng oAA nAA GeneID
# FeatureID GeneName CCDS Intron Exon cDNApos relcDNApos CDSpos relCDSpos
# protPos relProtPos Domain Dst2Splice Dst2SplType minDistTSS minDistTSE
# SIFTcat SIFTval PolyPhenCat PolyPhenVal priPhCons mamPhCons verPhCons
# priPhyloP mamPhyloP verPhyloP bStatistic targetScan mirSVR-Score mirSVR-E
# mirSVR-Aln cHmmTssA cHmmTssAFlnk cHmmTxFlnk cHmmTx cHmmTxWk cHmmEnhG cHmmEnh
# cHmmZnfRpts cHmmHet cHmmTssBiv cHmmBivFlnk cHmmEnhBiv cHmmReprPC cHmmReprPCWk
# cHmmQuies GerpRS GerpRSpval GerpN GerpS TFBS TFBSPeaks TFBSPeaksMax
# tOverlapMotifs motifDist Segway EncH3K27Ac EncH3K4Me1 EncH3K4Me3 EncExp
# EncNucleo EncOCC EncOCCombPVal EncOCDNasePVal EncOCFairePVal EncOCpolIIPVal
# EncOCctcfPVal EncOCmycPVal EncOCDNaseSig EncOCFaireSig EncOCpolIISig
# EncOCctcfSig EncOCmycSig Grantham Dist2Mutation Freq100bp Rare100bp Sngl100bp
# Freq1000bp Rare1000bp Sngl1000bp Freq10000bp Rare10000bp Sngl10000bp
# dbscSNV-ada_score dbscSNV-rf_score RawScore PHRED log2FC bn_p bn_p_adj bb_p
# bb_p_adj group_size bn_ASE bb_ASE
