#!./env/bin/python
# -*- coding: utf-8 -*-
"""Predictor of the variance of log2fc of variant with ASE
"""

import copy
import json
import random
import time
from collections import defaultdict
from functools import wraps
from os.path import join
from sys import stderr

import joblib
import numpy as np
import pandas as pd
import scipy as sp
from numpy import dtype
from pandas import DataFrame
from scipy.stats import spearmanr
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesClassifier,
                              RandomForestRegressor)
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     learning_curve, train_test_split)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.utils.estimator_checks import check_estimator

from config import estimators_list, grid_search_opt_params

# Visualization
try:
    import matplotlib.pyplot as plt
except ImportError:
    stderr.write("failed to import maplotlib.pyplot directly\n")
    import matplotlib as mpl
    mpl.use('agg')
    import matplotlib.pyplot as plt


def timmer(func):
    """Print the runtime of the decorated function
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


class AseVariancePredictor:
    """A class implementing prediction of ASE variance of a variant
    """

    def __init__(self, file_name, verbose=False, save_model=False):
        """Set up basic variables
        """

        self.input_file_name = file_name

        self.raw_df = None
        self.raw_df_shape = None
        self.raw_df_rows = None
        self.raw_df_cols = None

        self.work_df = None
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

    @timmer
    def run(self):
        """Execute a pre-designed construct pipeline
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
        self.slice_data_frame(flt=flt)

        self.label_encoder(encode=False)
        self.setup_xy()
        self.feature_pre_selection_by_spearman(
            drop_list=['CaddChrom', 'CaddPos', 'RawScore', 'PHRED'])
        self.train_test_slicer()

        self.setup_pipeline(estimators=estimators_list)
        self.grid_search_opt(self.pipeline, **grid_search_opt_params)
        self.draw_learning_curve()

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
        Parameters
        ----------
        df: str; optional; default work_df
            The DataFrame to be checked
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

    def setup_work_df(self):
        """Deep copy the raw DataFrame into work DataFrame
        """

        try:
            self.work_df = copy.deepcopy(self.raw_df)
        except Exception('Failed to deepcopy raw_df to work_df') as exp:
            raise exp

        self.raw_df_shape = self.raw_df.shape
        self.raw_df_cols = self.raw_df.columns
        self.raw_df_rows = self.raw_df.index
        self.update_work_dataframe_info()

    def slice_data_frame(self, rows=None, cols=None, flt=None):
        """Slice the DataFrame base on rows and cols
        """

        self.check_df()

        if isinstance(flt, str):
            self.work_df = self.work_df.query(flt)
        elif callable(flt):
            self.work_df = self.work_df[self.work_df.apply(flt, axis=1)]

        if rows is None and cols is None:
            rows = self.work_df.index
            cols = self.work_df.columns
        elif rows is None:
            rows = self.work_df.index
        elif cols is None:
            cols = self.work_df.columns

        self.work_df = self.work_df.loc[rows, cols]
        self.update_work_dataframe_info()

    def label_encoder(self, target_cols=None, skip=None, encode=True):
        """Encode category columns

        Parameters
        ----------
        target_cols: list or None; optional; default None
        skip: list, str or None; optional; default None

        Returns
        -------
        None

        Notes
        -----
        None
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
        for cn, _ in zip(target_cols, target_cols_encoded):
            if encode is False:
                del self.work_df[cn]
                continue

            self.work_df[cn] = encoder.fit_transform(
                self.work_df[cn].fillna('NA')
            )

        self.update_work_dataframe_info()

    def setup_xy(self, x_cols=None, y_col=None):
        """Set up predictor variables and target variables
        """

        cols = self.work_df.columns
        if x_cols is None and y_col is None:
            x_cols, y_col = cols[:-1], cols[-1]
        elif x_cols is None:
            cols.remove(y_col)
            x_cols = cols
        elif y_col is None:
            y_col = cols[-1]
            if y_col in x_cols:
                raise ValueError('Target column is in predictor columns')

        self.X = self.work_df.loc[:, x_cols]
        self.y = self.work_df.loc[:, y_col]

    def feature_pre_selection_by_spearman(self, drop_list=[], target=None,
                                          pvalue_threshhold=0.1):
        """Drop features with low corrlation to target variables
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

        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(self.X, self.y, **kwargs)

    def setup_pipeline(self, estimators=None):
        """Setup a training pipeline

        Parameters
        ----------
        estimators: None or a list of dicts; optional
            A list with estimators and their parameters
        """

        self.pipeline = Pipeline(estimators)

    @timmer
    def grid_search_opt(self, estimator=None, **kwargs):
        """Hyper-parameters optimization by GridSearchCV

        Strategy 1. Exhaustive grid search

        Parameters
        ----------
        estimator: Callable; compulsory
            Machine learning algorithm to be used

        param_grid: list of dict; compulsory
            Initial hyper-parameters
            e.g for SVC model, the initialized hyper-parameters could be:
            [
                {'kernel':['rbf'], 'gamma': [1e-3, 1e-4], 'C': [10, 1000]},
                {'kernel':['linear'], 'C': [10, 100, 1000]},
            ]

        scoring: None or list; optional
            Scoring metric

        cv: int or Callable; optional
            Cross-validation strategy
            e.g cv_fold = 10

        jobs: int; optional; default 1
            number of jobs at backend

        **kwargs: keyword options
            Any keyword argument suitable
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

        mae = mean_absolute_error(self.y_test, self.gsf_y_pred)
        print('mean absolute error: {}'.format(mae))

        evs = explained_variance_score(self.y_test, self.gsf_y_pred)
        print('explained_variance_score: {}'.format(evs))

        mes = mean_squared_error(self.y_test, self.gsf_y_pred)
        print('mean_square_error: {}'.format(mes))

        r2s = r2_score(self.y_test, self.gsf_y_pred)
        print('r2_score: {}'.format(r2s))

    def collect_cv_results(self, method='grid', param_dict=None):
        pass

    @timmer
    def random_search_opt(self, estimator=None, **kwargs):
        """Hyper-parameters optimization by RandomizedSearchCV

        # Strategy 2. Randomized parameter optimization

        Parameters
        ----------
        estimator: estimator object; compulsory
            A object

        param_dist: list of dict; compulsory
            Dictionary with parameters names as keys and distributions or
            lists of parameters to try.
            e.g for SVC model, the initialized hyper-parameters could be:
            tunned_parameters = [
                {'kernel':['rbf'], 'gamma': [1e-3, 1e-4], 'C': [10, 100]},
                {'kernel':['linear'], 'C': [10, 100, 1000]},
            ]

        iters: int; optional; default 20
            number of parameter settings that are sampled.

        cv: int or Callable; optional; default 10
            Cross-validation strategy
            e.g cv_fold = 10

        scoring: None or list; optional
            Evaluate the predictions on the test set

        jobs: int; optional; default 10
            Number of jobs to run in parallel.

        **kwargs: keyword options
            Any keyword argument suitable


        Returns
        -------
        None
        """

        if estimator is None:
            estimator = self.pipeline

        random_search = RandomizedSearchCV(estimator, **kwargs)

        rsf = random_search.fit(self.X_train, self.y_train)
        self.rsf_results = rsf
        self.rsf_cv_df = pd.DataFrame(rsf.cv_results_)

    @timmer
    def draw_learning_curve(self, estimator=None, file_name=None, title=None,
                            x_label=None, y_label=None, **kwargs):
        """Draw the learning curve of specific estimator or pipeline

        Parameters
        ----------
        estimator:
        title:
        xlabel: string; optional; default None
        ylabel: string; optional; default None

        Returns
        -------
        None

        Notes
        -----
        :param estimator: sklearn estimator; optional; default None
        :param file_name: string; optional; default None
        :param title: string; optional; default None
        :param x_label: string; optional; default None
        :param y_label: string; optional; default None
        :param kwargs:

        :returns None

        :Reference:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

        """

        if estimator is None:
            estimator = self.gsf_best_estimator
        elif estimator == 'rscv':
            estimator = self.rsf_best_estimator
        else:
            raise ValueError('Current only support GridSearchCV and '
                             + 'RandomSearchCV')

        if file_name is None:
            file_name = 'learning_curve'

        self.grid_search.set_params(cv=3)  # Use a lower vc times to accelerate

        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X=self.X, y=self.y, cv=5, n_jobs=6,
            train_sizes=np.linspace(.1, 1., 10))

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

    @timmer
    def draw_evaluation_fig(self, fig_name=None, fig_type='png'):
        """Visualization of the performance of the model

        :param fig_name:
        :param fig_type:
        :param scoring_dict:
        :return: None
        """

        if fig_name is None:
            fig_name = 'evaluation_fig'

        fig_name += '.' + fig_type

        plt.figure(figsize=(13, 13))
        plt.title('GridSearchCV evaluation by R^2 score')
        plt.xlabel('Minimum samples split')

        parameters = ['min_samples_split']  # , 'n_estimators']
        fig_num = len(parameters)
        fig, axs = plt.subplots(ncols=fig_num)

        if fig_num == 1:
            axs = [axs]

        for i, x in enumerate(parameters):
            _key = 'param_rfr__{}'.format(x)
            x_axis = np.array(self.gsf_results[_key].data, dtype=float)

            train_mean = self.gsf_results['mean_train_r2']
            train_std = self.gsf_results['std_train_r2']
            test_mean = self.gsf_results['mean_test_r2']
            test_std = self.gsf_results['std_test_r2']

            axs[i].fill_between(
                x_axis, color='r', alpha=0.1, linestyle='-',
                y1=train_mean - train_std, y2=train_mean + train_std
            )

            axs[i].fill_between(
                x_axis, color='g', alpha=0.1, linestyle='--',
                y1=test_mean - test_std, y2=test_mean + test_std
            )

            axs[i].plot(x_axis, train_mean, color='r', linestyle='-.')
            axs[i].plot(x_axis, test_mean, color='g', linestyle='-.')

        fig.savefig(fig_name)

    def save_model(self, pickle_file_name=None):
        """Save the mode in to pickle format

        :type pickle_file_name: string or None; default None
        """

        if pickle_file_name is None:
            pickle_file_name = 'slkearn_model.pl'

        joblib.dump(self.gsf, pickle_file_name)

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
    """

    input_path = join('/home', 'umcg-zzhang', 'Documents', 'projects',
                      'ASEpredictor', 'outputs', 'biosGavinOverlapCov10')
    input_file = join(input_path, 'biosGavinOlCv10AntUfltCstLog2FCVar.tsv')
    avp = AseVariancePredictor(input_file)
    avp.debug()


if __name__ == '__main__':
    main()
