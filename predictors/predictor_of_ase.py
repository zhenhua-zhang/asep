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


import copy
import json
import pickle
import time

from collections import defaultdict
from functools import wraps
from os.path import join
from sys import stderr
from sys import stdout

# numpy
import numpy as np
from numpy import dtype

# pandas
import pandas as pd
from pandas import DataFrame

# scipy
from scipy.stats import spearmanr

# scikit-learn modules
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# from sklearn.feature_selection import SelectFromModel
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Normalizer
# from sklearn.pipeline import FeatureUnion
# from sklearn.impute import MissingIndicator
# from sklearn.impute import SimpleImputer

# maplotlib as visualization modules
try:
    import matplotlib.pyplot as plt
except ImportError('Failed to import matplotlib.pyplot...') as e:
    print(e)
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt


def timmer(func):
    """Print the runtime of the decorated function

    Args:
        func (callable): function to be decoreated
    Returns: None
    """

    @wraps(func)
    def wrapper_timmer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        fn = func.__name__
        rt = time.perf_counter() - start_time
        #  st = time.perf_counter()
        stderr.write('{} is done; elapsed: {:.5f} secs\n'.format(fn, rt))
        return value

    return wrapper_timmer


def make_time_stamp():
    """Setup time stamp for the package"""
    global time_stamp
    time_stamp = time.strftime("%Y_%b_%d_%H_%M_%S", time.gmtime())


def make_file_name(fn=None, pre=None, suf=None, ts=None):
    """Create file name based on timestamp

    Args:
        fn (str or None): optional; defualt None
            The file name, if need.
        pre (str or None): optional; default None
            The prefix of the dumped file
        suf (str or None): optional; default None
            The suffix of the dumped file
        ts (str or None): optional; default None
            Time stamp used in file name
    Returns:
        fn (str):
            The created filename.
    """
    if ts is None:
        global time_stamp
        ts = time_stamp

    if fn is None:
        fn = ts
    else:
        fn += '_' + ts

    if pre:
        fn = pre + '_' + fn

    if suf:
        fn += '.' + suf

    return fn


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
configuration by using arrtibutes estimators_list, grid_search_opt_params, and
random_search_opt_params. You can also load your own configurations by
laod_config(YOUR-FILE-NAME), but please note it will covert the current
configurations(`set_default` will get you back to the default configs).

    Attributes:
        estimators_list (list): compulsory, no default
            A list of 2D-tuple, where tuple is (NAME, sklearn_estimator)
        grid_search_opt_params (defaultdict): compulsory, default defaultdict()
            A `defaultdict` from built-in `collections` module
        random_search_opt_params (defaultdict): options, default defaultdict()
            A `defaultdict` form built-in `collections` module

    Methods:
        set_default(self):
        get_configs(self):
        set_configs(self, **kwargs):
        dump_config(self, fn=None): dump configurations into a pickle file
        load_config(self, fn=None)

    Examples:
        >>> import Config
        >>> config = Config()
        >>> config.dump_config('configuration_file_name.pkl')
        >>> config = config.load
    """

    def __init__(self):
        """Initializing configuration metrics"""
        self.estimator_list = None
        self.grid_search_opt_params = defaultdict(None)
        self.random_search_opt_params = defaultdict(None)

        self.set_default()
        self.config_file_name = make_file_name(pre='config', suf='pkl')
        self.dump_configs(self.config_file_name)

    def set_default(self):
        """Set up default configuration"""

        self.estimators_list = [
            # ('imputator', SimpleImputer()),
            # ('standard_scaler', StandardScaler()),  # TODO: useful ???
            # ('normalizer', Normalizer()),  # TODO: usefule ???
            ('feature_selection', SelectKBest()),

            # Random forest classifier
            ('rfc', RandomForestClassifier()),

            # Ada boost classifier
            # ('abc', AdaBoostClassifier())
        ]

        self.grid_search_opt_params.update(
            dict(
                cv=10,
                n_jobs=3,
                iid=False,
                refit=True,  # by default is True
                scoring=make_scorer(precision_score, average='micro'),
                param_grid=[
                    dict(
                        feature_selection__score_func=[mutual_info_classif],
                        feature_selection__k=list(range(3, 110, 2)),
                        rfc__n_estimators=list(range(100, 1000, 10)),
                        rfc__max_features=['auto', 'sqrt'],
                        rfc__max_depth=list(range(10, 110, 11)),
                        rfc__min_samples_split=[2, 5, 10],
                        rfc__min_samples_leaf=[1, 2, 4],
                        rfc__bootstrap=[True, False]
                    ),
                ],
                return_train_score=True,
            )
        )

        self.random_search_opt_params.update(
            dict(
                cv=5,
                n_jobs=3,
                refit=True,  # by default is true
                n_iter=10,
                iid=False,
                scoring=make_scorer(precision_score, average='micro'),
                param_distributions=dict(
                    feature_selection__score_func=[mutual_info_classif],
                    feature_selection__k=list(range(3, 110, 2)),
                    rfc__n_estimators=list(range(100, 1000, 10)),
                    rfc__max_features=['auto', 'sqrt'],
                    rfc__max_depth=list(range(10, 110, 11)),
                    rfc__min_samples_split=[2, 5, 10],
                    rfc__min_samples_leaf=[1, 2, 4],
                    rfc__bootstrap=[True, False]
                ),
                return_train_score=True,  # to supress a warning
            )
        )

    def get_configs(self):
        """Get current configs"""
        return {'estimators': self.estimators_list,
                'grid_search_parameters': self.grid_search_opt_params,
                'random_search_parameters': self.random_search_opt_params
                }

    def set_configs(self, **kwargs):  # TODO: to finish this method
        """Set configs"""

    def dump_configs(self, fn=None):
        """Write the config into a file to make life easier.

        Args:
            fn (str or None): optional; default None
        """

        config_dict = {
            'estimators': self.estimators_list,
            'gs_params': self.grid_search_opt_params,
            'rs_params': self.random_search_opt_params
        }

        with open(fn, 'wb') as fnh:
            pickle.dump(config_dict, fnh)

    def load_configs(self, fn=None):
        """Load saved config into memory

        Args:
            fn (str or None): compulsory; default None
        Raises:
            IOError: when argument fn is None, raise IOError.
        """

        if fn is None:
            raise IOError('Need input file name')

        with open(fn, 'rb') as fh:
            config_dict = pickle.load(fh)

        self.estimators_list = config_dict['estimators']
        self.grid_search_opt_params = config_dict['gs_params']
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

        self.input_file_name = file_name

        config = Config()
        self.estimators_list = config.estimators_list
        self.grid_search_opt_params = config.grid_search_opt_params
        self.random_search_opt_params = config.random_search_opt_params

        self.raw_df = None
        self.work_df = None
        self.raw_df_info = defaultdict(None)
        self.work_df_info = defaultdict(None)

        self.pre_selected_features = None

        self.X = None
        self.y = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.y_pred = None

        self.pipeline = None

        self.grid_search = None
        self.gsf = None

        self.random_search = None
        self.rsf = None

    def __str__(self):
        return "ASEPredictor"

    @timmer
    def run(self):
        """Execute a pre-designed construct pipeline

        Args: None
        Returns: None
        """
        self.debug()

    @timmer
    def debug(self):
        """Debug the whole module
        """
        limit = None
        self.raw_df = self.read_file_to_dataframe(nrows=limit)

        sed = 1234
        self.set_seed(sed)

        self.check_df('raw_df')
        self.setup_work_df()

        flt = None
        cols_discarded = [
            'var', 'mean', 'p_value', 'gp_size', 'mirSVR.Score', 'mirSVR.E', 'mirSVR.Aln'
        ]
        self.work_df = self.slice_data_frame(
            fltout=flt, cols=[], rows=[], keep=False
        )
        self.update_work_dataframe_info()

        # change into binary classification.
        # need to change setup_pipeline multi_class into False
        multiclass = False
        if not multiclass:
            self.work_df.ASE = self.work_df.ASE.apply(abs)

        self.simple_imputer()
        self.label_encoder(remove=False)

        flt = 'gp_size > 5'
        self.train_test_df = self.slice_data_frame(
            fltout=flt, cols=cols_discarded, rows=[], keep=False
        )

        flt = 'gp_size <= 5'
        self.validating_df = self.slice_data_frame(
            fltout=flt, cols=cols_discarded, rows=[], keep=False
        )

        self.X, self.y = self.setup_xy(self.train_test_df, y_col='ASE')
        self.train_test_slicer(test_size=0.05)

        self.X_val, self.y_val = self.setup_xy(self.validating_df, y_col='ASE')

        self.setup_pipeline(
            estimators=self.estimators_list, multi_class=multiclass
        )
        # self.grid_search_opt(self.pipeline, **self.grid_search_opt_params)
        self.random_search_opt(self.pipeline, **self.random_search_opt_params)

        self.training_reporter()
        self.draw_figures()

    @staticmethod
    def set_seed(sed=None):
        """Set the random seed of numpy
        """
        if sed:
            np.random.seed(sed)
        else:
            np.random.seed(1234)

    @staticmethod
    def check_keys(pool_a, pool_b):
        """Check if all elements in pool_a are also in pool_b
        """
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
        """Get the name of input file.
        """
        return self.input_file_name

    def read_file_to_dataframe(self, nrows=None):
        """Read input file into pandas DataFrame.
        """
        file_name = self.input_file_name
        try:
            file_handle = open(file_name)
        except PermissionError as e:
            stderr.write('File IO error: ', e)
            return None
        else:
            with file_handle:
                return pd.read_table(file_handle, nrows=nrows)

    def check_df(self, df='work_df'):
        """Check the sanity of input DataFrame.

        Args:
            df (str): the data frame to be checked
        Returns: none
        Raises:
            TypeError:
            ValueError:
        """
        if df == 'work_df':
            if not isinstance(self.work_df, DataFrame):
                raise TypeError('Input was not a DataFrame of Pandas')
        elif df == 'raw_df':
            if not isinstance(self.raw_df, DataFrame):
                raise TypeError('Input was not a DataFrame of Pandas')
        else:
            raise ValueError('Unknown DataFrame {}...'.format(df))

    def update_work_dataframe_info(self):
        """Update the working dataframe after modifying the working dataframe.
        """
        self.work_df_info['shape'] = self.work_df.shape
        self.work_df_info['columns'] = self.work_df.columns
        self.work_df_info['index'] = self.work_df.index

    def setup_raw_dataframe_info(self):
        """Update the raw dataframe infromation.
        """
        self.raw_df_info['shape'] = self.raw_df.shape
        self.raw_df_info['columns'] = self.raw_df.columns
        self.raw_df_info['index'] = self.raw_df.index

    def setup_work_df(self):
        """Deep copy the raw DataFrame into work DataFrame.

        Args: none
        Returns: none
        Raises:
            Exception:
        """
        try:
            self.work_df = copy.deepcopy(self.raw_df)
        except Exception('Failed to deepcopy raw_df to work_df') as exp:
            raise exp

        self.setup_raw_dataframe_info()
        self.update_work_dataframe_info()

    def slice_data_frame(self, rows=None, cols=None, keep=False,
                         fltout=None, ax=1):
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
            ax (0, 1): optional, default 1
                Axis along which the flt is applied. 0 for rows, 1 for column.
        Returns: none
        Raises: none
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
            reuslt_df = copy.deepcopy(
                self.work_df[self.work_df.apply(fltout, axis=ax)]
            )

        if rows is None and cols is None:
            rows = self.work_df.index
            cols = self.work_df.columns
        elif rows is None:
            rows = self.work_df.index
        elif cols is None:
            cols = self.work_df.columns

        if keep:
            result_df = copy.deepcopy(self.work_df.loc[rows, cols])
        else:
            result_df = copy.deepcopy(
                self.work_df.drop(index=rows, columns=cols)
            )

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
        Returns: none
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
            for s in skip:
                if skip in target_cols:
                    target_cols.remove(s)
                else:
                    stderr.write('{} isn\'t in list...'.format(skip))
        elif skip is not None:
            raise TypeError('Need list, tuple or str type, or None...')

        target_cols_encoded = [n + '_encoded' for n in target_cols]

        encoder = LabelEncoder()
        for cn, cne in zip(target_cols, target_cols_encoded):
            if remove is True:
                del self.work_df[cn]
                continue

            try:
                self.work_df[cne] = encoder.fit_transform(
                    self.work_df[cn])
                del self.work_df[cn]
            except Exception as e:
                print(e, file=stderr)
                print(cne)

        self.update_work_dataframe_info()

    def simple_imputer(self):
        """A simple imputater based on pandas DataFrame.replace method.

        The columns information are derived from Dannis

        For all columns. In fact all of the missing values are np.NaN
        # to_replace_list = {
        #     'motifEName': '', 'GeneID': '', 'GeneName': '', 'CCDS': '', 'Intron': '',
        #     'Exon': '', 'ref': '', 'alt': '', 'Consequence': '', 'GC': np.NaN,
        #     'CpG': np.NaN, 'motifECount': np.NaN, 'motifEScoreChng': np.NaN,
        #     'motifEHIPos': np.NaN, 'oAA': np.NaN, 'nAA': '', 'cDNApos': np.NaN,
        #     'relcDNApos': np.NaN, 'CDSpos': np.NaN, 'relCDSpos': np.NaN, 'protPos': np.NaN,
        #     'relProtPos': np.NaN, 'Domain': '', 'Dst2Splice': np.NaN,
        #     'Dst2SplType': '', 'minDistTSS': np.NaN, 'minDistTSE': np.NaN,
        #     'SIFTcat': '', 'SIFTval': np.NaN, 'PolyPhenCat': '',
        #     'PolyPhenVal': np.NaN, 'priPhCons': np.NaN, 'mamPhCons': np.NaN,
        #     'verPhCons': np.NaN, 'priPhyloP': np.NaN, 'mamPhyloP': np.NaN,
        #     'verPhyloP': np.NaN, 'bStatistic': np.NaN, 'targetScan': np.NaN,
        #     'mirSVR-Score': np.NaN, 'mirSVR-E': np.NaN, 'mirSVR-Aln': np.NaN,
        #     'cHmmTssA': np.NaN, 'cHmmTssAFlnk': np.NaN, 'cHmmTxFlnk': np.NaN,
        #     'cHmmTx': np.NaN, 'cHmmTxWk': np.NaN, 'cHmmEnhG': np.NaN,
        #     'cHmmEnh': np.NaN, 'cHmmZnfRpts': np.NaN, 'cHmmHet': np.NaN,
        #     'cHmmTssBiv': np.NaN, 'cHmmBivFlnk': np.NaN, 'cHmmEnhBiv': np.NaN,
        #     'cHmmReprPC': np.NaN, 'cHmmReprPCWk': np.NaN, 'cHmmQuies': np.NaN,
        #     'GerpRS': np.NaN, 'GerpRSpval': np.NaN, 'GerpN': np.NaN, 'GerpS': np.NaN,
        #     'TFBS': np.NaN, 'TFBSPeaks': np.NaN, 'TFBSPeaksMax': np.NaN, 'tOverlapMotifs': np.NaN,
        #     'motifDist': np.NaN, 'Segway': '', 'EncH3K27Ac': np.NaN,
        #     'EncH3K4Me1': np.NaN, 'EncH3K4Me3': np.NaN, 'EncExp': np.NaN, 'EncNucleo': np.NaN,
        #     'EncOCC': np.NaN, 'EncOCCombPVal': np.NaN, 'EncOCDNasePVal': np.NaN,
        #     'EncOCFairePVal': np.NaN, 'EncOCpolIIPVal': np.NaN, 'EncOCctcfPVal': np.NaN,
        #     'EncOCmycPVal': np.NaN, 'EncOCDNaseSig': np.NaN, 'EncOCFaireSig': np.NaN,
        #     'EncOCpolIISig': np.NaN, 'EncOCctcfSig': np.NaN, 'EncOCmycSig': np.NaN,
        #     'Grantham': np.NaN, 'Dist2Mutation': np.NaN, 'Freq100bp': np.NaN, 'Rare100bp': np.NaN,
        #     'Sngl100bp': np.NaN, 'Freq1000bp': np.NaN, 'Rare1000bp': np.NaN, 'Sngl1000bp': np.NaN,
        #     'Freq10000bp': np.NaN, 'Rare10000bp': np.NaN, 'Sngl10000bp': np.NaN,
        #     'dbscSNV.ada_score': np.NaN, 'dbscSNV.rf_score': np.NaN
        # }

        """
        impute_values_dict = {
            'motifEName': 'unknown', 'GeneID': 'unknown', 'GeneName': 'unknown',
            'CCDS': 'unknown', 'Intron': 'unknown',
            'Exon': 'unknown', 'ref': 'N', 'alt': 'N', 'Consequence': 'UNKNOWN', 'GC': 0.42,
            'CpG': 0.02, 'motifECount': 0, 'motifEScoreChng': 0,
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

        to_replace_list = np.NaN

        self.work_df = self.work_df.replace(
            to_replace_list, impute_values_dict
        )
        self.update_work_dataframe_info()

    def setup_xy(self, dataframe, x_cols=None, y_col=None):
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

        X_cols = copy.deepcopy(dataframe.loc[:, x_cols])
        y_col = copy.deepcopy(dataframe.loc[:, y_col])
        return (X_cols, y_col)

    def feature_pre_selection_by_spearman(self, drop_list=[], target=None,
                                          pvalue_threshhold=0.1):
        """Drop features with low correlation to target variables."""
        if target is None:
            target = self.y

        if not isinstance(drop_list, (list, tuple)):
            raise TypeError("drop_list should be list, tuple")

        candidates_pool = {}
        feature_pool = self.work_df_cols
        for _, candidate in enumerate(feature_pool):
            sm = spearmanr(self.work_df[candidate], target)
            c = sm.correlation
            p = sm.pvalue
            if p <= pvalue_threshhold and candidate not in drop_list:
                candidates_pool[candidate] = dict(pvalue=p, correlation=c)

        with open('candidates.json', 'w') as json_f:
            json.dump(candidates_pool, json_f, sort_keys=True, indent=4)

        self.pre_selected_features = candidates_pool.keys()
        self.slice_data_frame(cols=self.pre_selected_features)

    def train_test_slicer(self, **kwargs):
        """Set up training and testing data set by train_test_split
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, **kwargs)

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
    def grid_search_opt(self, estimator=None, **kwargs):
        """Hyper-parameters optimization by GridSearchCV

        Strategy 1. Exhaustive grid search

        Args:
            estimator (estimator): compulsory; scikit-learn estimator object
                Machine learning algorithm to be used
            **kwargs: optional, keyword argument
                Any keyword argument suitable
        """
        if estimator is None:
            estimator = self.pipeline

        self.grid_search = GridSearchCV(estimator=estimator, **kwargs)
        self.gsf = self.grid_search.fit(self.X_train, self.y_train)

    @timmer
    def random_search_opt(self, estimators=None, **kwargs):
        """Hyper-parameters optimization by RandomizedSearchCV

        # Strategy 2. Randomized parameter optimization

        Args:
            estimators (estimator): compulsory; scikit-learn estimator object
                A object
            **kwargs: keyword arguments
                Any keyword argument suitable

        Returns: none
        Raises: none
        Notes: none
        """
        if estimators is None:
            estimators = self.pipeline

        self.random_search = RandomizedSearchCV(estimators, **kwargs)
        self.rsf = self.random_search.fit(self.X_train, self.y_train)

    @timmer
    def training_reporter(self):
        """Report the training information"""

        model_index = {0: 'grid', 1: 'random'}

        for index, fitted_model in enumerate([self.gsf, self.rsf]):
            if fitted_model is None:
                continue

            # working dataframe information
            format_print('work dataframe information', self.work_df_info)

            model_params = fitted_model.get_params()
            best_estimators = fitted_model.best_estimator_
            best_index = fitted_model.best_index_
            best_params = fitted_model.best_params_
            best_score = fitted_model.best_score_
            cv_results = fitted_model.cv_results_
            scorer = fitted_model.scorer_
            format_print('Params', model_params)
            format_print('Scorer', scorer)
            format_print('Best estimators', best_estimators)
            format_print('Best params', best_params)
            format_print('Best score', best_score)
            format_print('Best index', best_index)

            prefix = 'cross_validation_' + model_index[index]
            cv_result_fn = make_file_name(
                fn='training', pre=prefix, suf='tvs')
            with open(cv_result_fn, 'w') as cvof:
                tmp_data_frame = pd.DataFrame(cv_results)
                tmp_data_frame.to_csv(cvof, sep='\t')

            format_print('Cross-validation results', cv_result_fn)

            self.y_pred = fitted_model.predict(self.X_test)
            fitted_model_score = fitted_model.score(self.X_test, self.y_test)
            format_print('Model score', fitted_model_score)

    @timmer
    def draw_learning_curve(self, estimator, model_index, strategy=None, **kwargs):
        """Draw the learning curve of specific estimator or pipeline

        Args:
            estimator (sklearn estimators): compulsary
            model_index (string): compulsary
            strategy (str or None): optional, default None
            **kwargs: optional; default empty
                Keyword arguments for learning_curve from scikit-learn

        Notes:
            https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
        """
        if strategy is None:
            estimator = copy.deepcopy(self.estimators_list[-1][-1])
            estimator.set_params(n_estimators=100)
        elif strategy == 'best':
            estimator = estimator.best_estimator_
        elif strategy == 'pipe':
            pass
        else:
            raise Exception(
                'For [strategy] argument, (None, \'best\', or \'pipe\')'
            )

        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X=self.X, y=self.y, cv=10, n_jobs=6,
            train_sizes=np.linspace(.1, 1., 10), **kwargs
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        fig, ax_learning = plt.subplots(figsize=(10, 10))

        upper = train_scores_mean + train_scores_std
        lower = train_scores_mean - train_scores_std
        ax_learning.fill_between(train_sizes, upper, lower, alpha=0.1)
        ax_learning.plot(
            train_sizes, train_scores_mean, color='r', label='Training score'
        )

        upper = test_scores_mean + test_scores_std
        lower = test_scores_mean - test_scores_std
        ax_learning.fill_between(train_sizes, upper, lower, alpha=0.1)
        ax_learning.plot(
            train_sizes, test_scores_mean, color='g', label='Cross-validation score'
        )

        ax_learning.set(
            title='Learning curve', xlabel='Training examples', ylabel='Score'
        )
        ax_learning.legend(loc='best')

        prefix = 'learning_curve_' + model_index
        fig.savefig(make_file_name(pre=predix, suf='png'))

    @timmer
    def draw_roc_curve(self, estimator, model_index, **kwargs):
        """Draw ROC curve for test and validate data set"""

        fig, ax_roc = plt.subplots(figsize=(10, 10))

        self.y_test_pred_prob = estimator.predict_proba(self.X_test)[:, 1]
        fp, tp, _ = roc_curve(self.y_test, self.y_test_pred_prob)
        ax_roc.plot(fp, tp, color='r', label='Testing')

        # if there is a validating data set
        if self.X_val is not None:
            self.y_val_pred_prob = estimator.predict_proba(self.X_val)[:, 1]
            fp, tp, _ = roc_curve(self.y_val, self.y_val_pred_prob)
            ax_roc.plot(fp, tp, color='g', label='Validating')

        ax_roc.set(
            title='ROC curve', xlabel='False positive rage', ylabel='True positive rate'
        )
        ax_roc.legend(loc='best')

        prefix = 'roc_curve_' + model_index
        fig.savefig(make_file_name(pre=prefix, suf='png'))

    @timmer
    def draw_K_main_features(self, estimator, index, k=20, **kwargs):
        """Draw feature importance for the model"""
        ftr_slc_est = estimator.best_estimator_.steps[0][-1]
        slc_ftr_idc = ftr_slc_est.get_support(True)

        rfc_ftr_ipt = estimator.best_estimator_.steps[-1][-1]
        rfc_ftr_ipt = rfc_ftr_ipt.feature_importances_

        ftr_nms = self.X_train.columns[slc_ftr_idc]
        ftr_ipt_mtx = list(zip(ftr_nms, rfc_ftr_ipt))

        ftr_ipt_mtx = sorted(ftr_ipt_mtx, key=lambda x: -x[-1])
        ftr_ipt_mtx = ftr_ipt_mtx[:k]

        rfc_ftr_ipt = [x[-1] for x in ftr_ipt_mtx]
        ftr_nms = [x[0] for x in ftr_ipt_mtx]

        fig, ax_features = plt.subplots(figsize=(10, 10))
        ax_features.bar(ftr_nms, rfc_ftr_ipt)

        ax_features.set_xticklabels(
            ftr_nms, rotation_mode='anchor', rotation=45, horizontalalignment='right'
        )
        ax_features.set(
            title='Feature importances', xlabel='Features', ylabel='Importance'
        )

        prefix = 'feature_importance_' + model_index[index]
        fig.savefig(make_file_name(pre=prefix, suf='png'))

    @timmer
    def draw_figures(self, **kwargs):
        """Draw learning curve, ROC curve, and bar graph for the importance of features"""
        model_indexs = {0: 'grid', 1: 'random'}
        for index, estimator in enumerate([self.rsf, self.gsf]):
            if estimator is None:
                continue

            model_index = model_indexs[index]
            self.draw_roc_curve(estimator, model_index)
            self.draw_learning_curve(estimator, model_index)
            self.draw_K_main_features(estimator, model_index)


def save_ap_obj(ob, file_name=None):
    """Save ASEPredictor instance by pickle"""
    if file_name is None:
        file_name = 'ASEPre'

    pklf_name = make_file_name(file_name, pre='training', suf='pkl')
    with open(pklf_name, 'wb') as pklof:
        pickle.dump(ob, pklof)


def load_asepredictor_obj(file_name):
    """Load ASEPredictor instance by pickle"""

    with open(file_name, 'wb') as pklif:
        return pickle.load(pklif)


def main():
    """Main function to run the module

    Args: none
    Returns: none
    """

    make_time_stamp()

    input_file = join(
        '/home', 'umcg-zzhang', 'Documents', 'projects', 'ASEpredictor',
        'outputs', 'biosGavinOverlapCov10',
        'biosGavinOlCv10AntUfltCstLog2FCBin.tsv'
    )
    ap = ASEPredictor(input_file)
    ap.debug()
    save_ap_obj(ap)


make_time_stamp()

input_file = join(
    '/home', 'umcg-zzhang', 'Documents', 'projects', 'ASEpredictor',
    'outputs', 'biosGavinOverlapCov10',
    'biosGavinOlCv10AntUfltCstLog2FCBin.tsv'
)
ap = ASEPredictor(input_file)
ap.debug()
save_ap_obj(ap)


if __name__ == '__main__':
    main()
