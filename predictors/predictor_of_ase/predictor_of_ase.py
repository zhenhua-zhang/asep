#!./env/bin/python
# -*- coding: utf-8 -*-

"""Predicting Allele-specific expression effect

Allele-specific expression predictor

Example:
    $ python3 predictor_of_ase.py

Attributes:
    input_file_name (str): data set used to train the model

Methods:
    __init__(self, file_name, verbose=False)

TODO:
    * Eliminate some module level variables
    * Add more input file type
"""

import os
import copy
import time
import pickle

from functools import wraps
from sys import stderr, stdout

# numpy
import numpy
from numpy import dtype

# pandas
import pandas
from pandas import DataFrame

# scipy
from scipy import stats, interp

# scikit-learn modules
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (precision_score, accuracy_score, make_scorer,
                             roc_curve, auc)
from sklearn.model_selection import (RandomizedSearchCV, train_test_split,
                                     StratifiedKFold, learning_curve)

# maplotlib as visualization modules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# global variables
TIME_STAMP = time.strftime("%Y_%b_%d_%H_%M_%S", time.gmtime())

def timmer(func):
    """Print the runtime of the decorated function

    Args:
        func (callable): function to be decoreated
    """
    @wraps(func)
    def wrapper_timmer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        function_name = func.__name__
        elapsed_time = time.perf_counter() - start_time
        stderr.write(
            '{} is done; elapsed: {:.5f} secs\n'.format(
                function_name, elapsed_time
            )
        )
        return value

    return wrapper_timmer


def make_time_stamp():
    """Setup time stamp for the package"""
    global TIME_STAMP
    TIME_STAMP = time.strftime("%Y_%b_%d_%H_%M_%S", time.gmtime())


def make_file_name(file_name=None, prefix=None, suffix=None, _time_stamp=None):
    """Create file name based on timestamp

    Args:
        file_name (str or None): optional; defualt None
            The file name, if need.
        prefix (str or None): optional; default None
            The prefix of the dumped file
        suffix (str or None): optional; default None
            The suffix of the dumped file
        _time_stamp (str or None): optional; default None
            Time stamp used in file name
    Returns:
        file_name (str):
            The created filename.
    """
    if _time_stamp is None:
        global TIME_STAMP
        _time_stamp = TIME_STAMP

    if file_name is None:
        file_name = _time_stamp
    else:
        file_name += '_' + _time_stamp

    if prefix:
        file_name = prefix + '_' + file_name

    if suffix:
        file_name += '.' + suffix

    if not os.path.exists(_time_stamp):
        os.mkdir(_time_stamp)

    return os.path.join(".", _time_stamp, file_name)


def format_print(title, main_content, pipe=stdout):
    """A method format the prints"""
    head_no = 10
    head = '-' * head_no
    tail_no = 60 - len(title)

    if tail_no < 0:
        tail_no = 0
    tail = '-' * tail_no

    flag = ' '.join([head, title, tail])

    print(flag, file=pipe)
    print(' ', main_content, file=pipe)
    print(file=pipe)


class Config:
    """Config module for the ASEPredictor

A class to configure the ASEPredictor class. You can use the default
configuration by using arrtibutes estimators_list, and
random_search_opt_params. You can also load your own configurations by
laod_config(YOUR-FILE-NAME), but please note it will covert the current
configurations(`set_default` will get you back to the default configs).

    Attributes:
        estimators_list (list): compulsory, no default
            A list of 2D-tuple, where tuple is (NAME, sklearn_estimator)
        random_search_opt_params (dict): options, default dict()
            A `dict` form built-in `collections` module

    Methods:
        set_default(self):
        get_configs(self):
        set_configs(self, **kwargs):
        dump_config(self, file_name=None): dump configurations into a pickle file
        load_config(self, file_name=None)

    Examples:
        >>> import Config
        >>> config = Config()
        >>> config.dump_config('configuration_file_name.pkl')
        >>> config = config.load
    """

    def __init__(self):
        """Initializing configuration metrics"""
        self.estimators_list = None
        self.random_search_opt_params = {}

        self.set_default()
        self.config_file_name = make_file_name(prefix='config', suffix='pkl')
        self.dump_configs(self.config_file_name)

    def set_default(self):
        """Set up default configuration

        A emperical hyperparameter matrix
        {
            'rfc__n_estimators': 290,
            'rfc__min_samples_split': 8,
            'rfc__min_samples_leaf': 4,
            'rfc__max_features': 'auto',
            'rfc__max_depth': 90,
            'rfc__bootstrap': False,
            'feature_selection__score_func': mutual_info_classif,
            'feature_selection__k': 51
        }
        """
        self.estimators_list = [
            # feature selelction function
            ('feature_selection', SelectKBest()),

            # Random forest classifier
            ('rfc', RandomForestClassifier()),
        ]

        scoring_dict = dict(
            precision=make_scorer(precision_score, average="micro"),
            accuracy=make_scorer(accuracy_score)
        )

        self.random_search_opt_params.update(
            dict(
                cv=5,
                n_jobs=8,  # Use all cores
                n_iter=15,
                iid=False,
                refit="precision",
                scoring=scoring_dict,
                param_distributions=dict(
                    feature_selection__score_func=[mutual_info_classif],
                    feature_selection__k=list(range(20, 81, 2)),
                    rfc__n_estimators=list(range(50, 501, 20)),
                    rfc__max_depth=list(range(50, 111, 10)),
                    rfc__min_samples_split=[4, 6, 8, 10, 12],
                    rfc__min_samples_leaf=[2, 4, 6, 8],
                    rfc__bootstrap=[True, False]
                ),
                return_train_score=True,  # to suppress a warning
            )
        )

    def get_configs(self):
        """Get current configs"""
        return {'estimators': self.estimators_list,
                'random_search_parameters': self.random_search_opt_params
                }

    def set_configs(self, **kwargs):
        """Set configs"""

    def dump_configs(self, file_name=None):
        """Write the config into a file to make life easier.

        Args:
            file_name (str or None): optional; default None
        """
        config_dict = {
            'estimators': self.estimators_list,
            'rs_params': self.random_search_opt_params
        }

        with open(file_name, 'wb') as file_handler:
            pickle.dump(config_dict, file_handler)

    def load_configs(self, file_name=None):
        """Load saved config into memory

        Args:
            file_name (str or None): compulsory; default None
        Raises:
            IOError: when argument file_name is None, raise IOError.
        """

        if file_name is None:
            raise IOError('Need input file name')

        with open(file_name, 'rb') as file_handler:
            config_dict = pickle.load(file_handler)

        self.estimators_list = config_dict['estimators']
        self.random_search_opt_params = config_dict['rs_params']


class ASEPredictor:
    """A class implementing prediction of ASE variance of a variant

    Example:
        >>> import ASEPredictor
        >>> ipf = 'input.tsv'
        >>> ap = ASEPredictor(ipf)
        >>> ap.run()
    """

    def __init__(self, file_name, verbose=False):
        """Set up basic variables

        Args:
            file_name (str): input data set
        """
        if verbose:
            print(verbose)

        self.input_file_name = file_name

        config = Config()
        self.estimators_list = config.estimators_list
        self.random_search_opt_params = config.random_search_opt_params

        self.raw_df = None
        self.work_df = None
        self.raw_df_info = {}
        self.work_df_info = {}

        self.train_test_df = None

        self.x_vars = None
        self.y_var = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.pipeline = None

        self.random_search = None
        self.random_search_fitted = None

    @timmer
    def run(self):
        """Execute a pre-designed construct pipeline"""
        limit = None
        self.raw_df = self.read_file_to_dataframe(nrows=limit)

        sed = 1234
        self.set_seed(sed)

        self.check_df('raw_df')
        self.setup_work_df()

        flt = None
        cols_discarded = [
            "meta_log2FC_mean", "log2FC_var", "log2FC_mean", "pval_tt_log2FC",
            "pval_tt_log2FC_adj", "pval_st_log2FC", "pval_st_log2FC_adj",
            "meta_FC_Mean", "FC_var", "FC_mean", "pval_tt_FC", "pval_tt_FC_adj",
            "pval_st_FC", "pval_st_FC_adj", "group_size", "mirSVR.Score",
            "mirSVR.E", "mirSVR.Aln"
        ]
        self.work_df = self.slice_data_frame(
            fltout=flt, cols=[], rows=[], keep=False
        )
        self.update_work_dataframe_info()

        # change into binary classification.
        # need to change setup_pipeline multi_class into False
        multiclass = False
        y_col = 'ASE'
        if not multiclass:
            self.work_df[y_col] = self.work_df[y_col].apply(abs)

        self.simple_imputer()
        self.label_encoder(remove=False)

        flt = 'group_size >= 5'
        self.train_test_df = self.slice_data_frame(
            fltout=flt, cols=cols_discarded, rows=[], keep=False
        )

        self.x_vars, self.y_var = self.setup_xy(self.train_test_df, y_col=y_col)

        # flt = 'group_size < 2'
        # self.validating_df = self.slice_data_frame(
            # fltout=flt, cols=cols_discarded, rows=[], keep=False
        # )
        # self.X_val, self.y_val = self.setup_xy(self.validating_df, y_col='ASE')

        self.setup_pipeline(
            estimators=self.estimators_list, multi_class=multiclass
        )
        self.k_fold_stratified_validation()
        self.training_reporter()
        self.draw_learning_curve(self.random_search_fitted, strategy="pipe")

    @staticmethod
    def set_seed(sed=None):
        """Set the random seed of numpy"""
        if sed:
            numpy.random.seed(sed)
        else:
            numpy.random.seed(1234)

    @staticmethod
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

    def get_input_file_name(self):
        """Get the name of input file."""
        return self.input_file_name

    def read_file_to_dataframe(self, nrows=None):
        """Read input file into pandas DataFrame."""
        file_name = self.input_file_name
        try:
            file_handler = open(file_name)
        except PermissionError as err:
            stderr.write('File IO error: ', err)
            return None
        else:
            with file_handler:
                return pandas.read_table(file_handler, nrows=nrows)

    def check_df(self, dataframe='work_df'):
        """Check the sanity of input DataFrame.

        Args:
            dataframe (str): the data frame to be checked
        Raises:
            TypeError:
            ValueError:
        """
        if dataframe == 'work_df':
            if not isinstance(self.work_df, DataFrame):
                raise TypeError('Input was not a DataFrame of Pandas')
        elif dataframe == 'raw_df':
            if not isinstance(self.raw_df, DataFrame):
                raise TypeError('Input was not a DataFrame of Pandas')
        else:
            raise ValueError('Unknown DataFrame {}...'.format(dataframe))

    def update_work_dataframe_info(self):
        """Update the information of working dataframe after modifying it"""
        self.work_df_info['shape'] = self.work_df.shape
        self.work_df_info['columns'] = self.work_df.columns
        self.work_df_info['index'] = self.work_df.index

    def setup_raw_dataframe_info(self):
        """Update the raw dataframe infromation"""
        self.raw_df_info['shape'] = self.raw_df.shape
        self.raw_df_info['columns'] = self.raw_df.columns
        self.raw_df_info['index'] = self.raw_df.index

    def setup_work_df(self):
        """Deep copy the raw DataFrame into work DataFrame"""
        try:
            self.work_df = copy.deepcopy(self.raw_df)
        except Exception('Failed to deepcopy raw_df to work_df') as exp:
            raise exp

        self.setup_raw_dataframe_info()
        self.update_work_dataframe_info()

    def slice_data_frame(self, rows=None, cols=None, keep=False, fltout=None):
        """Slice the DataFrame base on rows and cols.

        Args:
            rows (list, tuple, None): optional, default None
                Rows retained for the downstream. If it's None, all rows will
                be retained.
            cols (list, tuple, None): optional, default None
                Columns retained for the downstream. If it's None, all columns
                will be retained.
            keep (bool): optional, default `True`
                Whether the values of `rows` or `cols` will be kept or
                discarded. If True, cells coorderated by `rows` and `cols` will
                be keep and the exclusive will be discarded, otherwise the way
                around.
            fltout (callable, str, None): optional, default None
                A filter to screen dataframe. If the fltout is callable,
                `apply` method of DataFrame will be used; if it's a `str`
                object, `query` method will be called; otherwise, if it's
                `None`, no filter will be applied.
        """
        self.check_df()

        if not isinstance(keep, bool):
            raise TypeError('keep should be bool')

        if not keep and (rows is None or cols is None):
            raise TypeError(
                'if keep is False, neither rows nor cols can be None'
            )

        if isinstance(fltout, str):
            result_df = copy.deepcopy(self.work_df.query(fltout))
        elif callable(fltout):
            result_df = copy.deepcopy(
                self.work_df[self.work_df.apply(fltout, axis=1)]
            )
        else:
            result_df = copy.deepcopy(self.work_df)

        if rows is None and cols is None:
            rows = self.work_df.index
            cols = self.work_df.columns
        elif rows is None:
            rows = self.work_df.index
        elif cols is None:
            cols = self.work_df.columns

        if keep:
            result_df = result_df.loc[rows, cols]
        else:
            result_df = result_df.drop(index=rows, columns=cols)

        return result_df

    def label_encoder(self, target_cols=None, skip=None, remove=True):
        """Encode category columns.

        Args:
            target_cols(list or None): name of columns to be encoded
            skip(string, list, tuple, None): list of names of columns
                skipped encoded. string represents only the specific column
                will be skipped; list or tuple means all contained elements
                will be skipped; None means no columns will be skipped.
            remove (bool): remove columns need to be encoded.
        Raises:
            TypeError:
        """
        self.check_df()

        if target_cols is None:
            col_types = self.work_df.dtypes
            target_cols = [n for n, t in col_types.items() if t is dtype('O')]
        elif not isinstance(target_cols, list):
            raise TypeError('Need list type...')

        if isinstance(skip, str):
            if skip in target_cols:
                target_cols.remove(skip)
        elif isinstance(skip, (list, tuple)):
            for skipped in skip:
                if skip in target_cols:
                    target_cols.remove(skipped)
                else:
                    stderr.write('{} isn\'t in list...'.format(skip))
        elif skip is not None:
            raise TypeError('Need list, tuple or str type, or None...')

        target_cols_encoded = [n + '_encoded' for n in target_cols]

        encoder = LabelEncoder()
        for col_tag, col_tag_encoded in zip(target_cols, target_cols_encoded):
            if remove is True:
                del self.work_df[col_tag]
                continue

            try:
                self.work_df[col_tag_encoded] = encoder.fit_transform(
                    self.work_df[col_tag]
                )
                del self.work_df[col_tag]
            except ValueError as err:
                print(err, file=stderr)

        self.update_work_dataframe_info()

    def simple_imputer(self):
        """A simple imputater based on pandas DataFrame.replace method.

        The columns information are derived from Dannis

        For all columns. In fact all of the missing values are np.NaN
        to_replace_list = {
            'motifEName': '', 'GeneID': '', 'GeneName': '', 'CCDS': '',
            'Intron': '', 'Exon': '', 'ref': '', 'alt': '', 'Consequence': '',
            'GC': np.NaN, 'CpG': np.NaN, 'motifECount': np.NaN,
            'motifEScoreChng': np.NaN, 'motifEHIPos': np.NaN, 'oAA': np.NaN,
            'nAA': '', 'cDNApos': np.NaN, 'relcDNApos': np.NaN,
            'CDSpos': np.NaN, 'relCDSpos': np.NaN, 'protPos': np.NaN,
            'relProtPos': np.NaN, 'Domain': '', 'Dst2Splice': np.NaN,
            'Dst2SplType': '', 'minDistTSS': np.NaN, 'minDistTSE': np.NaN,
            'SIFTcat': '', 'SIFTval': np.NaN, 'PolyPhenCat': '',
            'PolyPhenVal': np.NaN, 'priPhCons': np.NaN, 'mamPhCons': np.NaN,
            'verPhCons': np.NaN, 'priPhyloP': np.NaN, 'mamPhyloP': np.NaN,
            'verPhyloP': np.NaN, 'bStatistic': np.NaN, 'targetScan': np.NaN,
            'mirSVR-Score': np.NaN, 'mirSVR-E': np.NaN, 'mirSVR-Aln': np.NaN,
            'cHmmTssA': np.NaN, 'cHmmTssAFlnk': np.NaN, 'cHmmTxFlnk': np.NaN,
            'cHmmTx': np.NaN, 'cHmmTxWk': np.NaN, 'cHmmEnhG': np.NaN,
            'cHmmEnh': np.NaN, 'cHmmZnfRpts': np.NaN, 'cHmmHet': np.NaN,
            'cHmmTssBiv': np.NaN, 'cHmmBivFlnk': np.NaN, 'cHmmEnhBiv': np.NaN,
            'cHmmReprPC': np.NaN, 'cHmmReprPCWk': np.NaN, 'cHmmQuies': np.NaN,
            'GerpRS': np.NaN, 'GerpRSpval': np.NaN, 'GerpN': np.NaN,
            'GerpS': np.NaN, 'TFBS': np.NaN, 'TFBSPeaks': np.NaN,
            'TFBSPeaksMax': np.NaN, 'tOverlapMotifs': np.NaN,
            'motifDist': np.NaN, 'Segway': '', 'EncH3K27Ac': np.NaN,
            'EncH3K4Me1': np.NaN, 'EncH3K4Me3': np.NaN, 'EncExp': np.NaN,
            'EncNucleo': np.NaN, 'EncOCC': np.NaN, 'EncOCCombPVal': np.NaN,
            'EncOCDNasePVal': np.NaN, 'EncOCFairePVal': np.NaN,
            'EncOCpolIIPVal': np.NaN, 'EncOCctcfPVal': np.NaN,
            'EncOCmycPVal': np.NaN, 'EncOCDNaseSig': np.NaN,
            'EncOCFaireSig': np.NaN, 'EncOCpolIISig': np.NaN,
            'EncOCctcfSig': np.NaN, 'EncOCmycSig': np.NaN, 'Grantham': np.NaN,
            'Dist2Mutation': np.NaN, 'Freq100bp': np.NaN, 'Rare100bp': np.NaN,
            'Sngl100bp': np.NaN, 'Freq1000bp': np.NaN, 'Rare1000bp': np.NaN,
            'Sngl1000bp': np.NaN, 'Freq10000bp': np.NaN, 'Rare10000bp': np.NaN,
            'Sngl10000bp': np.NaN, 'dbscSNV.ada_score': np.NaN,
            'dbscSNV.rf_score': np.NaN
        }"""
        impute_values_dict = {
            'motifEName': 'unknown', 'GeneID': 'unknown', 'GeneName': 'unknown',
            'CCDS': 'unknown', 'Intron': 'unknown',
            'Exon': 'unknown', 'ref': 'N', 'alt': 'N', 'Consequence': 'UNKNOWN',
            'GC': 0.42, 'CpG': 0.02, 'motifECount': 0, 'motifEScoreChng': 0,
            'motifEHIPos': 0, 'oAA': 'unknown', 'nAA': 'unknown', 'cDNApos': 0,
            'relcDNApos': 0, 'CDSpos': 0, 'relCDSpos': 0, 'protPos': 0,
            'relProtPos': 0, 'Domain': 'UD', 'Dst2Splice': 0,
            'Dst2SplType': 'unknown', 'minDistTSS': 5.5, 'minDistTSE': 5.5,
            'SIFTcat': 'UD', 'SIFTval': 0, 'PolyPhenCat': 'unknown',
            'PolyPhenVal': 0, 'priPhCons': 0.115, 'mamPhCons': 0.079,
            'verPhCons': 0.094, 'priPhyloP': -0.033, 'mamPhyloP': -0.038,
            'verPhyloP': 0.017, 'bStatistic': 800, 'targetScan': 0,
            'mirSVR-Score': 0, 'mirSVR-E': 0, 'mirSVR-Aln': 0,
            'cHmmTssA': 0.0667, 'cHmmTssAFlnk': 0.0667, 'cHmmTxFlnk': 0.0667,
            'cHmmTx': 0.0667, 'cHmmTxWk': 0.0667, 'cHmmEnhG': 0.0667,
            'cHmmEnh': 0.0667, 'cHmmZnfRpts': 0.0667, 'cHmmHet': 0.667,
            'cHmmTssBiv': 0.667, 'cHmmBivFlnk': 0.0667, 'cHmmEnhBiv': 0.0667,
            'cHmmReprPC': 0.0667, 'cHmmReprPCWk': 0.0667, 'cHmmQuies': 0.0667,
            'GerpRS': 0, 'GerpRSpval': 0, 'GerpN': 1.91, 'GerpS': -0.2,
            'TFBS': 0, 'TFBSPeaks': 0, 'TFBSPeaksMax': 0, 'tOverlapMotifs': 0,
            'motifDist': 0, 'Segway': 'unknown', 'EncH3K27Ac': 0,
            'EncH3K4Me1': 0, 'EncH3K4Me3': 0, 'EncExp': 0, 'EncNucleo': 0,
            'EncOCC': 5, 'EncOCCombPVal': 0, 'EncOCDNasePVal': 0,
            'EncOCFairePVal': 0, 'EncOCpolIIPVal': 0, 'EncOCctcfPVal': 0,
            'EncOCmycPVal': 0, 'EncOCDNaseSig': 0, 'EncOCFaireSig': 0,
            'EncOCpolIISig': 0, 'EncOCctcfSig': 0, 'EncOCmycSig': 0,
            'Grantham': 0, 'Dist2Mutation': 0, 'Freq100bp': 0, 'Rare100bp': 0,
            'Sngl100bp': 0, 'Freq1000bp': 0, 'Rare1000bp': 0, 'Sngl1000bp': 0,
            'Freq10000bp': 0, 'Rare10000bp': 0, 'Sngl10000bp': 0,
            'dbscSNV.ada_score': 0, 'dbscSNV.rf_score': 0
        }

        to_replace_list = numpy.NaN

        self.work_df = self.work_df.replace(
            to_replace_list, impute_values_dict
        )
        self.update_work_dataframe_info()

    @staticmethod
    def setup_xy(dataframe, x_cols=None, y_col=None):
        """Set up predictor variables and target variables.

        Args:
            x_cols(list, tuple, None):
            y_col(string, None):
        Returns: DataFrame
        Raises:
            ValueError:
        """
        cols = dataframe.columns
        if x_cols is None and y_col is None:
            x_cols, y_col = cols[:-1], cols[-1]
        elif x_cols is None:
            x_cols = cols.drop(y_col)
        elif y_col is None:
            y_col = cols[-1]
            if y_col in x_cols:
                raise ValueError('Target column is in predictor columns')

        x_cols_dataframe = copy.deepcopy(dataframe.loc[:, x_cols])
        y_col_series = copy.deepcopy(dataframe.loc[:, y_col])
        return (x_cols_dataframe, y_col_series)

    def train_test_slicer(self, **kwargs):
        """Set up training and testing data set by train_test_split"""
        (self.x_train, self.x_test,
         self.y_train, self.y_test
        ) = train_test_split(self.x_vars, self.y_var, **kwargs)

    def setup_pipeline(self, estimators=None, multi_class=False):
        """Setup a training pipeline

        Args:
            estimators (estimator): None or a list of dicts; optional
                A list with estimators and their parameters
        """
        if multi_class:
            self.pipeline = OneVsOneClassifier(Pipeline(estimators))
        else:
            self.pipeline = Pipeline(estimators)

    @timmer
    def random_search_opt(self, estimators=None, **kwargs):
        """Hyper-parameters optimization by RandomizedSearchCV

        # Strategy 2. Randomized parameter optimization

        Args:
            estimators (estimator): compulsory; scikit-learn estimator object
                A object
            **kwargs: keyword arguments
                Any keyword argument suitable
        """
        if estimators is None:
            estimators = self.pipeline

        self.random_search = RandomizedSearchCV(estimators, **kwargs)
        self.random_search_fitted = self.random_search.fit(self.x_train, self.y_train)

    @timmer
    def training_reporter(self):
        """Report the training information"""
        format_print('Work dataframe information', self.work_df_info)
        format_print('Params', self.random_search_fitted.get_params())
        format_print('Scorer', self.random_search_fitted.scorer_)
        format_print('Best estimators', self.random_search_fitted.best_estimator_)
        format_print('Best params', self.random_search_fitted.best_params_)
        format_print('Best score', self.random_search_fitted.best_score_)
        format_print('Best index', self.random_search_fitted.best_index_)

        prefix = 'cross_validation_random'
        cv_result_file_name = make_file_name(
            file_name='training', prefix=prefix, suffix='tvs'
        )

        cv_results = self.random_search_fitted.cv_results_
        with open(cv_result_file_name, 'w') as cvof:
            tmp_data_frame = pandas.DataFrame(cv_results)
            tmp_data_frame.to_csv(cvof, sep='\t')

        format_print('Cross-validation results', cv_result_file_name)

        model_score = self.random_search_fitted.score(self.x_test, self.y_test)
        format_print('Model score', model_score)

    @timmer
    def draw_learning_curve(self, estimator, strategy=None, **kwargs):
        """Draw the learning curve of specific estimator or pipeline

        Args:
            estimator (sklearn estimators): compulsary
            strategy (str or None): optional, default None
            **kwargs: optional; default empty
                Keyword arguments for learning_curve from scikit-learn
        """
        if strategy is None:
            estimator = copy.deepcopy(self.estimators_list[-1][-1])
            estimator.set_params(n_estimators=100)
        elif strategy == 'best':
            estimator = estimator.best_estimator_
        elif strategy == 'pipe':
            estimator.set_params(n_iter=3, cv=3, iid=False)
        else:
            raise Exception('Valid strategy, (None, \'best\', or \'pipe\')')

        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X=self.x_vars, y=self.y_var, cv=3, n_jobs=None,
            train_sizes=numpy.linspace(.1, 1., 20), **kwargs
        )

        train_scores_mean = numpy.mean(train_scores, axis=1)
        train_scores_std = numpy.std(train_scores, axis=1)
        test_scores_mean = numpy.mean(test_scores, axis=1)
        test_scores_std = numpy.std(test_scores, axis=1)

        fig, ax_learning = plt.subplots(figsize=(10, 10))

        ax_learning.fill_between(
            train_sizes,
            train_scores_mean + train_scores_std,
            train_scores_mean - train_scores_std,
            alpha=0.1
        )
        ax_learning.plot(
            train_sizes, train_scores_mean, color='r',
            label='Training score(precision)'
        )

        ax_learning.fill_between(
            train_sizes,
            test_scores_mean + test_scores_std,
            test_scores_mean - test_scores_std,
            alpha=0.1
        )
        ax_learning.plot(
            train_sizes, test_scores_mean, color='g',
            label='Cross-validation score(precision)'
        )

        ax_learning.set(
            title='Learning curve',
            xlabel='Training examples', ylabel='Score(precision)'
        )
        ax_learning.legend(loc='best')

        fig.savefig(
            make_file_name(prefix='learning_curve_random', suffix='png')
        )

    def k_fold_stratified_validation(self, cvs=3, **kwargs):
        """K-fold stratified validation by StratifiedKFold from scikit-learn"""
        skf = StratifiedKFold(n_splits=cvs, **kwargs)

        auc_fpr_tpr_pool = []
        feature_pool = {}
        for idx, (train_idx, test_idx) in enumerate(skf.split(self.x_vars, self.y_var)):
            self.x_train = self.x_vars.iloc[train_idx]
            self.x_test = self.x_vars.iloc[test_idx]
            self.y_train = self.y_var.iloc[train_idx]
            self.y_test = self.y_var.iloc[test_idx]

            self.random_search_opt(
                self.pipeline,
                **self.random_search_opt_params
            )

            y_test_pred_prob = self.random_search_fitted.predict_proba(self.x_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_test_pred_prob)
            roc_auc = auc(fpr, tpr)
            auc_fpr_tpr_pool.append([roc_auc, fpr, tpr])

            ftr_slc_est = self.random_search_fitted.best_estimator_.steps[0][-1]
            slc_ftr_idc = ftr_slc_est.get_support(True)
            rfc_ftr_ipt = self.random_search_fitted.best_estimator_.steps[-1][-1]
            rfc_ftr_ipt = rfc_ftr_ipt.feature_importances_
            ftr_nms = self.x_train.columns[slc_ftr_idc]
            for name, importance in zip(ftr_nms, rfc_ftr_ipt):
                if name in feature_pool:
                    feature_pool[name][idx] = importance
                else:
                    feature_pool[name] = [0] * cvs
                    feature_pool[name][0] = importance

        draw_roc_curve_cv(auc_fpr_tpr_pool)
        draw_k_main_features_cv(feature_pool)


@timmer
def draw_k_main_features_cv(feature_pool, first_k=20):
    """Draw feature importance for the model with cross-validation"""
    name_mean_std_pool = []
    for name, importances in feature_pool.items():
        mean = numpy.mean(importances)
        std = numpy.std(importances, ddof=1)
        name_mean_std_pool.append([name, mean, std])

    name_mean_std_pool = sorted(name_mean_std_pool, key=lambda x: -x[1])

    name_pool, mean_pool, std_pool = [], [], []
    for name, mean, std in name_mean_std_pool[:first_k]:
        name_pool.append(name)
        mean_pool.append(mean)
        std_pool.append(std)

    fig, ax_features = plt.subplots(figsize=(10, 10))
    ax_features.bar(name_pool, mean_pool, yerr=std_pool)
    ax_features.set_xticklabels(
        name_pool, rotation_mode='anchor', rotation=45,
        horizontalalignment='right'
    )
    ax_features.set(
        title="Feature importances(with stand deviation as error bar)",
        xlabel='Feature name', ylabel='Importance'
    )

    prefix = 'feature_importance_random_'
    fig.savefig(make_file_name(prefix=prefix, suffix='png'))


@timmer
def draw_roc_curve_cv(auc_fpr_tpr_pool):
    """Draw ROC curve with cross-validation"""
    fig, ax_roc = plt.subplots(figsize=(10, 10))
    auc_pool, fpr_pool, tpr_pool = [], [], []
    space_len = 0
    for auc_area, fpr, tpr in auc_fpr_tpr_pool:
        auc_pool.append(auc_area)
        fpr_pool.append(fpr)
        tpr_pool.append(tpr)

        if len(fpr) > space_len:
            space_len = len(fpr)

    lspace = numpy.linspace(0, 1, space_len)
    interp_fpr_pool, interp_tpr_pool = [], []
    for fpr, tpr in zip(fpr_pool, tpr_pool):
        fpr_interped = interp(lspace, fpr, fpr)
        fpr_interped[0], fpr_interped[-1] = 0, 1
        interp_fpr_pool.append(fpr_interped)

        tpr_interped = interp(lspace, fpr, tpr)
        tpr_interped[0], tpr_interped[-1] = 0, 1
        interp_tpr_pool.append(tpr_interped)

    for fpr, tpr in zip(interp_fpr_pool, interp_tpr_pool):
        ax_roc.plot(fpr, tpr, lw=0.5)

    fpr_mean = numpy.mean(interp_fpr_pool, axis=0)
    tpr_mean = numpy.mean(interp_tpr_pool, axis=0)
    tpr_std = numpy.std(interp_tpr_pool, axis=0)

    # A 95% confidence interval for the mean of AUC by Bayesian mvs
    mean, *_ = stats.bayes_mvs(auc_pool)
    auc_mean, (auc_min, auc_max) = mean.statistic, mean.minmax

    ax_roc.plot(
        fpr_mean, tpr_mean, color="r", lw=2,
        label="Mean: AUC={:0.3}, [{:0.3}, {:0.3}]".format(
            auc_mean, auc_min, auc_max
        )
    )

    mean_upper = numpy.minimum(tpr_mean + tpr_std, 1)
    mean_lower = numpy.maximum(tpr_mean - tpr_std, 0)
    ax_roc.fill_between(
        fpr_mean, mean_upper, mean_lower, color='green', alpha=0.1,
        label="Standard deviation"
    )

    ax_roc.legend(loc="best")

    prefix = 'roc_curve_cv_random'
    fig.savefig(make_file_name(prefix=prefix, suffix='png'))

def save_ap_obj(obj, file_name=None):
    """Save ASEPredictor instance by pickle"""
    if file_name is None:
        file_name = 'ASEPre'

    pklf_name = make_file_name(file_name, prefix='training', suffix='pkl')
    with open(pklf_name, 'wb') as pklof:
        pickle.dump(obj, pklof)


def load_asepredictor_obj(file_name):
    """Load ASEPredictor instance by pickle"""
    with open(file_name, 'wb') as pklif:
        return pickle.load(pklif)


def main():
    """Main function to run the module """
    make_time_stamp()
    input_file = os.path.join(
        '/home', 'umcg-zzhang', 'Documents', 'projects', 'ASEpredictor',
        'outputs', 'biosGavinOverlapCov10',
        'biosGavinOlCv10AntUfltCstLog2FCBin.tsv'
    )
    asepredictor = ASEPredictor(input_file)
    asepredictor.run()
    save_ap_obj(asepredictor)


if __name__ == '__main__':
    main()
