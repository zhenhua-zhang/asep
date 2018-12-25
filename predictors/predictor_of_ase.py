#!./env/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
"""Predictor of the variance of log2fc of variant with ASE

Allele-specific expression predictor

Example:
    $ python3 predictor_of_ase.py

Attributes:
    input_file_name (str): data set used to train the model

TODO:
    * Eliminate some module level variables
    * Add more input file type

"""


import copy
import json
import time

from collections import defaultdict
from functools import wraps
from os.path import join
from sys import stderr

# Third party modules
import joblib
import numpy as np
import pandas as pd

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
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
# from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr
from pandas import DataFrame
from numpy import dtype

# from sklearn.feature_selection import SelectFromModel
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Normalizer
# from sklearn.pipeline import FeatureUnion
# from sklearn.impute import MissingIndicator

# maplotlib as visualization modules
import matplotlib as mpl
mpl.use('agg')


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


class Config:

    """Config module for the ASEPredictor

    Args:
        extimators_list (list): compulsory, no default
            A list of 2D-tuple, where tuple is (NAME, sklearn_estimator)
        grid_search_opt_params (defaultdict): compulsory, default defaultdict()
            A defaultdict from built-in
    """

    def __init__(self):
        """
        """

        self.estimators_list = [
            # ('imputator', SimpleImputer()),
            # ('standard_scaler', StandardScaler()),  # TODO: useful ???
            # ('normalizer', Normalizer()),  # TODO: usefule ???
            ('feature_selection', SelectKBest()),

            # Random forest classifier
            ('rfc', RandomForestClassifier(n_estimators=20))
        ]

        precision_scorer = make_scorer(precision_score, average='micro')
        accuracy_scorer = make_scorer(accuracy_score)

        self.grid_search_opt_params = defaultdict(None)
        self.grid_search_opt_params.update(
            dict(
                cv=5,
                n_jobs=3,
                refit='accu',
                iid=False,
                scoring=dict(  # model evaluation metrics
                    accu=accuracy_scorer,
                    preci=precision_scorer
                ),
                param_grid=[
                    dict(
                        # imputator__strategy=['mean'],
                        estimator__feature_selection__score_func=[
                            mutual_info_classif],
                        estimator__feature_selection__k=list(range(2, 20, 2)),
                        estimator__rfc__min_samples_split=list(range(2, 10))
                    ),
                ],
                return_train_score=True,
            )
        )

        self.random_search_opt_params = defaultdict(None)
        self.random_search_opt_params.update(
            dict(
                cv=5,
                n_jobs=3,
                refit='preci',
                n_iters=10,
                iid=False,  # To supress warnings
                scoring=dict(  # model evaluation metrics
                    preci='precision',
                    accu='accuracy'
                ),
                param_distribution=[
                    dict(
                        # imputator__strategy=['mean'],
                        estimator__feature_selection__score_func=[
                            mutual_info_classif],
                        estimator__feature_selection__k=list(range(2, 20, 2)),
                        estimator__rfc__min_samples_split=list(range(2, 10))
                    ),
                ],
                return_train_score=True,  # to supress a warning
            )
        )

    def set_estimators_list(self, **kwargs):
        """Set estimators
        """


class ASEPredictor:

    """A class implementing prediction of ASE variance of a variant

    Example:
        >>> import ASEPredictor
        >>> ipf = 'input.tsv'
        >>> ap = ASEPredictor(ipf)
        >>> ap.run()
    """

    def __init__(self, file_name, verbose=False, save_model=False):
        """Set up basic variables

        Args:
            file_name (str): input data set
        """

        self.input_file_name = file_name

        config = Config()
        self.estimators_list = config.estimators_list
        self.grid_search_opt_params = config.grid_search_opt_params

        self.raw_df = None
        self.raw_df_info = dict(shape=None, rows=[], cols=[])
        self.raw_df_shape = None
        self.raw_df_rows = None
        self.raw_df_cols = None

        self.work_df = None
        self.work_df_info = dict(shape=None, rows=[], cols=[])
        self.work_df_shape = None
        self.work_df_cols = None
        self.work_df_rows = None

        self.X = None
        self.y = None

        self.pre_selected_features = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.pipeline = None

        self.grid_search = None
        self.gsf = None
        self.gsf_results_matrix = defaultdict(list)

        self.gsf_results = None
        self.gsf_cv_df = None
        self.gsf_best_estimator = None
        self.gsf_best_score = None
        self.gsf_best_params = None
        self.gsf_best_index = None
        self.gsf_y_pred = None

        self.random_search = None
        self.rsf = None
        self.rsf_results = None
        self.rsf_results_matrix = defaultdict(list)
        self.rsf_cv_df = None
        self.rsf_best_estimator = None
        self.rsf_best_score = None
        self.rsf_best_params = None
        self.rsf_best_index = None
        self.rsf_y_pred = None

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

        # flt = 'log2FCVar>0'
        flt = None
        cols_discarded = ['var', 'mean', 'p_value',
                          'gp_size', 'mirSVR.Score', 'mirSVR.E', 'mirSVR.Aln']
        self.slice_data_frame(fltout=flt, cols=cols_discarded,
                              rows=[], keep=False)
        self.simple_imputer()
        self.label_encoder(remove=False)
        self.setup_xy(y_col='ASE')
        self.train_test_slicer(test_size=0.15)

        self.setup_pipeline(estimators=self.estimators_list, multi_class=True)
        self.grid_search_opt(self.pipeline, **self.grid_search_opt_params)
        self.draw_learning_curve(scoring='accuracy')

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
        """Get the name of input file
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
        """Check the sanity of input DataFrame

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
        """Update the working dataframe after modifying the working dataframe
        """
        self.work_df_cols = self.work_df.columns
        self.work_df_rows = self.work_df.index
        self.work_df_shape = self.work_df.shape

    def setup_raw_dataframe_info(self):
        """Restore the raw dataframe infromation
        """
        self.raw_df_cols = self.raw_df.columns
        self.raw_df_rows = self.raw_df.index
        self.raw_df_shape = self.raw_df.shape

    def setup_work_df(self):
        """Deep copy the raw DataFrame into work DataFrame

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
        """Slice the DataFrame base on rows and cols

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
            self.work_df = self.work_df.query(fltout)
        elif callable(fltout):
            self.work_df = self.work_df[self.work_df.apply(fltout, axis=1)]

        if rows is None and cols is None:
            rows = self.work_df.index
            cols = self.work_df.columns
        elif rows is None:
            rows = self.work_df.index
        elif cols is None:
            cols = self.work_df.columns

        if keep:
            self.work_df = self.work_df.loc[rows, cols]
        else:
            self.work_df = self.work_df.drop(index=rows, columns=cols)

        self.update_work_dataframe_info()

    def label_encoder(self, target_cols=None, skip=None, remove=True):
        """Encode category columns

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
                self.work_df[cne] = encoder.fit_transform(self.work_df[cn])
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

    def setup_xy(self, x_cols=None, y_col=None):
        """Set up predictor variables and target variables

        Args:
            x_cols(list, tuple, None):
            y_col(string, None):
        Returns: none
        Raises:
            ValueError:
        """
        cols = self.work_df.columns
        if x_cols is None and y_col is None:
            x_cols, y_col = cols[:-1], cols[-1]
        elif x_cols is None:
            x_cols = cols.drop(y_col)
        elif y_col is None:
            y_col = cols[-1]
            if y_col in x_cols:
                raise ValueError('Target column is in predictor columns')

        self.X = self.work_df.loc[:, x_cols]
        self.y = self.work_df.loc[:, y_col]

    def feature_pre_selection_by_spearman(self, drop_list=[], target=None,
                                          pvalue_threshhold=0.1):
        """Drop features with low correlation to target variables
        """
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

        Returns: none
        Raises: none
        """
        if estimator is None:
            estimator = self.pipeline

        self.grid_search = GridSearchCV(estimator=estimator, **kwargs)

        gsf = self.grid_search.fit(self.X_train, self.y_train)

        self.gsf = gsf
        self.gsf_results = gsf.cv_results_
        self.gsf_cv_df = pd.DataFrame(self.gsf_results)

        if 'refit' in kwargs:
            print('refit method is: {}'.format(kwargs['refit']))
            self.gsf_best_estimator = gsf.best_estimator_
            self.gsf_best_score = gsf.best_score_
            self.gsf_best_params = gsf.best_params_
            self.gsf_best_index = gsf.best_index_
            print('Grid search cv, Best score: {}'.format(gsf.best_score_))
            print('Grid search cv, Best params: {}'.format(gsf.best_params_))
            print('Grid search cv, Best index: {}'.format(gsf.best_index_))

        self.gsf_y_pred = gsf.predict(self.X_test)

        gsf_score = gsf.score(self.X_test, self.y_test)
        print('model refit score: {}'.format(gsf_score))

    def collect_cv_results(self, method='grid', param_dict=None):
        pass

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

        random_search = RandomizedSearchCV(estimators, **kwargs)

        rsf = random_search.fit(self.X_train, self.y_train)
        self.rsf_results = rsf
        self.rsf_cv_df = pd.DataFrame(rsf.cv_results_)

    @timmer
    def draw_learning_curve(self, estimator=None, file_name=None, title=None,
                            x_label=None, y_label=None, **kwargs):
        """Draw the learning curve of specific estimator or pipeline

        Args:
            estimator (estimator): compulsary, defualt None
            file_name (str): optional, default None
            title (str): optional, default None
            x_label (str): optional; default None
            y_label (str): optional; default None
            **kwargs: optional; default empty

        Returns: none
        Raises: none
        Notes:
            https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
        """
        if estimator is None:
            estimator = self.gsf_best_estimator
        elif estimator == 'rscv':
            estimator = self.rsf_best_estimator
        else:
            raise ValueError(
                'Current only support GridSearchCV and RandomSearchCV'
            )

        if file_name is None:
            file_name = 'learning_curve'

        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X=self.X, y=self.y, cv=10, n_jobs=6,
            train_sizes=np.linspace(.1, 1., 10), **kwargs)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        fig, ax = plt.subplots()

        if title is None:
            title = 'Learning_curve'
        ax.set_title(title)

        if x_label is None:
            x_label = 'Training examples'
        ax.set_xlabel(x_label)

        if y_label is None:
            y_label = 'Score'
        ax.set_ylabel(y_label)

        upper_border = train_scores_mean + train_scores_std
        lower_border = train_scores_mean - train_scores_std
        ax.fill_between(train_sizes, upper_border, lower_border, alpha=0.1)

        upper_border = test_scores_mean + test_scores_std
        lower_border = test_scores_mean - test_scores_std
        ax.fill_between(train_sizes, upper_border, lower_border, alpha=0.1)

        ax.plot(train_sizes, train_scores_mean, 'o-', color='r',
                label='Training score')

        ax.plot(train_sizes, test_scores_mean, 'o-', color='g',
                label='Cross-validation score')

        ax.legend(loc='best')
        fig.savefig(file_name)

    def save_model(self, pickle_file_name=None):
        """Save the mode in to pickle format

        Args:
            pickle_file_name (str): optional, default None
        Returns: none
        Raises: none
        """

        if pickle_file_name is None:
            pickle_file_name = 'slkearn_model.pl'

        joblib.dump(self.gsf, pickle_file_name)
        # joblib.dump(self.rsf, pickle_file_name)

    # TODO: save the training data
    def save_training_data(self):
        pass

    # TODO: save the signature(e.g version and dependencies) of scikit-learn
    def save_sklearn_sig(self):
        pass

    # TODO: save the cross-validation data set obtained from training data
    def save_cv_data(self):
        pass


def main():
    """Main function to run the module

    Args: none
    Returns: none
    """

    input_file = join(
        '/home', 'umcg-zzhang', 'Documents', 'projects', 'ASEpredictor',
        'outputs', 'biosGavinOverlapCov10',
        'biosGavinOlCv10AntUfltCstLog2FCBin.tsv'
    )
    ap = ASEPredictor(input_file)
    ap.debug()


input_file = join(
    '/home', 'umcg-zzhang', 'Documents', 'projects', 'ASEpredictor',
    'outputs', 'biosGavinOverlapCov10',
    'biosGavinOlCv10AntUfltCstLog2FCBin.tsv'
)
ap = ASEPredictor(input_file)
ap.debug()


if __name__ == '__main__':
    main()
