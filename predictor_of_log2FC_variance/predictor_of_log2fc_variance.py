#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Predictor of the variance of log2fc of variant with ASE
'''

import time
import copy

import numpy as np
from numpy import dtype

import pandas as pd
from pandas import DataFrame

from collections import Iterable
from functools import wraps
from os.path import join
from sys import stderr

# Data pre-processing and model optimization
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer

# Models
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

# Visualization
try:
    import matplotlib.pyplot as plt
except Exception as exc:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except:
    raise ImportError('Failed to import matplotlib...')


def timmer(func):
    """Print the runtime of the decorated function
    """
    @wraps(func)
    def wrapper_timmer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        fn = func.__name__
        rt = time.perf_counter() - start_time
        stderr.write('{} is done; elapsed: {:.5f} secs\n'.format(fn, rt))
        start_time = time.perf_counter()
        return value
    return wrapper_timmer


class AseVariancePredictor():
    """A class implementing prediction of ASE variance of a variant
    """
    def __init__(self, file_name, verbose=False):
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

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.pipeline = None
        
        self.grid_search = None
        self.gsf = None
        self.gsf_results = None
        self.gsf_cv_df = None
        self.gsf_best_estimator = None
        self.gsf_best_score = None
        self.gsf_best_params = None
        self.gsf_best_index = None
        self.gsf_y_pred = None

        self.rsf_results = None
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

        LIMIT=5000
        self.raw_df = self.read_file_to_dataframe()

        sed = 1234
        self.set_seed(sed)

        self.check_df('raw_df')  
        self.setup_work_df()

        filter = 'log2FCVar>0'
        # filter = None
        self.slice_data_frame(filter=filter)
        self.label_encoder()
        self.train_test_slicer()

        transformer = FeatureUnion(
            transformer_list=[
                ('imputator', SimpleImputer()),
                ('indicators', MissingIndicator())
            ]
        )

        estimators = [
            ('transformer', transformer),
            ('robustScaler', RobustScaler()),
            ('normalizer', Normalizer()),
            ('feature_selection', SelectFromModel(
                    ExtraTreesClassifier(n_estimators=50)
                )
            ),
            # Neural network, multi-layer perceptron regressor
            ('mlp', MLPRegressor()),
            
            # Ada boost regressor
            # ('adb', AdaBoostRegressor()),

            # Random forest regressor
            # ('rfr', RandomForestRegressor(n_estimators=10))
        ]
        self.setup_pipeline(estimators=estimators)

        scoring = {
            'r2': 'r2',
            'ev': 'explained_variance',
            'nmae': 'neg_mean_absolute_error',
            'nmse': 'neg_mean_squared_error',
            'nmdae': 'neg_median_absolute_error'
        }

        param_grid = [
            {
                'transformer__imputator__strategy': ['mean'],
                'transformer__indicators__features': ['missing-only'],
                'transformer__indicators__error_on_new': [False],
                
                # Neural network, multi-layer perceptron regresssor
                'mlp__hidden_layer_sizes': [75, 100, 125],
                'mlp__alpha': [0.0001, 0.001, 0.01],
                'mlp__max_iter': [300],
                

                # Ada boost regressor
                # 'adb__n_estimators': [50],
                # 'adb__learning_rate': [0.2, 0.5, 1.0, 1.5, 2.0],

                # Random forest regressor
            #    'rfr__n_estimators': [50],
            #    'rfr__min_samples_split': range(2, 23, 10),
            },
        ]

        self.grid_search_opt(self.pipeline, param_grid=param_grid, jobs=3,
                             cv=5, scoring=scoring, refit='r2', iid=False,
                             return_train_score=True)

        self.draw_learning_curve()
        

    def set_seed(self, sed=None):
        """Set the random seed of numpy
        """
        if sed:
            np.random.seed(sed)
        else:
            np.random.seed(1234)


    def get_input_file_name(self):
        """Get the name of input file 
        """
        return self.input_file_name


    def check_keys(self, pool_a, pool_b):
        """Check if all elements in pool_a are also in pool_b
        """
        if not isinstance(pool_a, Iterable):
            raise TypeError('Require iterable value for pool_a...')
        if not isinstance(pool_b, Iterable):
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


    def read_file_to_dataframe(self, nrows=None):
        """Read input file into pandas DataFrame.
        """
        #  TODO: function the limit argument
        file_name = self.input_file_name
        try:
            file_handle = open(file_name)
        except PermissionError as err:
            stderr.write('File IO error: ', err)
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


    def descripe_raw_data(self, target_col):
        """Descriptive statistics of raw data
        """
        # Simple statistics of raw data for both X and y
        self.work_df.shape
        self.work_df[target_col]
        pass


    def setup_work_df(self):
        '''Deep copy the raw DataFrame into work DataFrame
        '''
        try:
            self.work_df = copy.deepcopy(self.raw_df)
        except Exception('Failed to deepcopy raw_df to work_df') as exp:
            raise exp

        self.raw_df_shape = self.raw_df.shape
        self.raw_df_cols = self.raw_df.columns
        self.raw_df_rows = self.raw_df.index
        self.work_df_shape = self.work_df.shape
        self.work_df_cols = self.work_df.columns
        self.work_df_rows = self.work_df.index


    def slice_data_frame(self, rows=None, cols=None, filter=None):
        """Slice the DataFrame base on rows and cols
        """

        self.check_df()

        if isinstance(filter, str):
            self.work_df = self.work_df.query(filter)
        elif callable(filter):
            self.work_df = self.work_df[self.work_df.apply(filter, axis=1)]

        if rows is None and cols is None:
            rows = self.work_df.index
            cols = self.work_df.columns
        elif rows is None:
            rows = self.work_df.index
        elif cols is None:
            cols = self.work_df.columns
        
        self.work_df = self.work_df.loc[rows, cols]
    

    def label_encoder(self, target_cols=None, skip=None):
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
        for cn, cne in zip(target_cols, target_cols_encoded):
            self.work_df[cne] = encoder.fit_transform(
                self.work_df[cn].fillna('NA')
            )
            del self.work_df[cn]

        self.work_df_shape = self.work_df.shape
        self.work_df_cols = self.work_df.columns
        self.work_df_rows = self.work_df.index


    def train_test_slicer(self, X_cols=None, y_col=None, **kwargs):
        """Set up training and testing data set by train_test_split
        """

        cols = self.work_df.columns
        if X_cols is None and y_col is None:
            X_cols, y_col = cols[:-1], cols[-1]
        elif X_cols is None:
            cols.remove(y_col)
            X_cols = cols
        elif y_col is None:
            y_col = cols[-1]
            if y_col in X_cols:
                raise ValueError('Target column is in predictor columns')
        
        self.X, self.y = self.work_df.loc[:, X_cols], self.work_df.loc[:, y_col]
        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(self.X, self.y)


    def setup_pipeline(self, estimators=None):
        #  TODO: complete the parse of estimators 
        """Setup a trainig pipeline
        
        Parameters
        ----------
        estimators: None or a list of dicts; optional
            A list with estimators and their parameters
        """

        if estimators == None:
            transformer = FeatureUnion(
                transformer_list=[
                    ('imputator', SimpleImputer(strategy='mean')),

                    #
                    # Use error_on_new to suppress the error when there are 
                    # features with missing values in transform that have no 
                    # missing values in fit
                    #
                    ('indicators', MissingIndicator(
                        features='missing-only', error_on_new=False))
                ]
            )
            
            estimators = [
                ('transformer', transformer),
                ('robustScaler', RobustScaler()),
                ('normalizer', Normalizer()),
                ('feature_selection', SelectFromModel(
                        ExtraTreesClassifier(n_estimators=50)
                    )
                ),
                ('svr', SVR())
            ]
        
        self.pipeline = Pipeline(estimators)


    @timmer
    def grid_search_opt(self, estimator=None, param_grid=[], scoring=None, 
                        cv=10, jobs=5, **kwargs): 
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
                {'kernel':['rbf'], 'gamma': [1e-3, 1e-4], 'C': [10, 100, 1000]}, 
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

        if not param_grid:
            param_grid = [
                {
                    'robustScaler__quantile_range': [(25, 75)],
                    'normalizer__norm': ['l2', 'l1'],
                    'svr__C': [10, 100, 1000],
                    'svr__degree': [2, 3, 4],
                    'svr__kernel': ['linear', 'rbf'], #, 'poly', 'sigmoid'],
                    'svr__gamma': [1e-3, 1e-4],
                },
            ]

        if scoring is None:
            scoring = 'explained_variance'

        self.grid_search = GridSearchCV(
            estimator=estimator, param_grid=param_grid, cv=cv, 
            scoring=scoring, n_jobs=jobs, **kwargs
        )

        gsf = self.grid_search.fit(self.X_train, self.y_train)
    
        self.gsf = gsf
        self.gsf_results = gsf.cv_results_
        self.gsf_cv_df = pd.DataFrame(self.gsf_results)

        if 'refit' in kwargs:
            self.gsf_best_estimator = gsf.best_estimator_
            self.gsf_best_score = gsf.best_score_
            self.gsf_best_params = gsf.best_params_
            self.gsf_best_index = gsf.best_index_

        self.gsf_y_pred = gsf.predict(self.X_test)


    @timmer
    def random_search_opt(self, estimator=None, param_dist=[], iters=10, 
                          scoring=None, cv=10, jobs=5, **kwargs):
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
                {'kernel':['rbf'], 'gamma': [1e-3, 1e-4], 'C': [10, 100, 1000]}, 
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
        
        if not param_dist:
            param_dist = [
                {
                    'robustScaler__quantile_range': [(25, 75)],
                    'normalizer__norm': ['l2', 'l1'],
                    # 'pca__n_components': [5, 20, 30, 40, 50],
                    'svr__C': [0.5, 1., 2.],
                    'svr__degree': [2, 3, 4],
                    'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'svr__gamma': ['auto', 'scale'],
                },
            ]

        if scoring is None:
            scoring = ['explained_variance', 'neg_mean_absolute_error']
        
        random_search = RandomizedSearchCV(
            estimator, param_distributions=param_dist, n_iter=iters, 
            cv=cv, scoring=scoring, n_jobs=jobs, **kwargs
        )
    
        rsf = random_search.fit( self.X_train, self.y_train) 
        self.rsf_results = rsf
        self.rsf_cv_df = pd.DataFrame(rsf.cv_results_)

    
    @timmer
    def draw_learning_curve(self, estimator=None, file_name=None, title=None, 
                           xlabel=None, ylabel=None, **kwargs):
        """Draw the learning curve of specific estimator or pipeline
        
        Parameters
        ----------
        estimator: estimator; optional; default None
        title: string; optional; default None
        xlabel: string; optional; default None
        ylabel: string; optional; default None
        
        Returns
        -------
        None
        
        Notes
        -----
        Reference: 
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
        """

        if estimator is None:
            estimator = self.grid_search
        elif estimator == 'rscv':
            estimator = self.random_search
        # else:
        #     self.check_estimators(estimator)
        
        if file_name is None:
            file_name = 'learning_curve'
        
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X=self.X, y=self.y, cv=5, n_jobs=6,
            train_sizes=np.linspace(.1, 1., 10))
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std= np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        fig, ax = plt.subplots()

        if title is None: title = 'Learning_curve'
        ax.set_title(title)
        
        if xlabel is None: xlabel = 'Training examples'
        ax.set_xlabel(xlabel)
        if ylabel is None: ylabel = 'Score'
        ax.set_ylabel(ylabel)
        
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
    def draw_evaluation_fig(self, scoring_dict=None):
        """Visualization of the performance of the model
        """
        
        plt.figure(figsize=(13,13))
        plt.title('GridSearchCV evaluation by R^2 score')
        plt.xlabel('Minimum samples split')

        parameters = ['min_samples_split'] #, 'n_estimators']
        fig_num = len(parameters)
        fig, axs = plt.subplots(ncols=fig_num)

        if fig_num == 1:
            axs = [axs]

        for i, x in enumerate(parameters):
            _key = 'param_rfr__{}'.format(x)
            X_axis = np.array(self.gsf_results[_key].data, dtype=float)

            sample_score_train_mean = self.gsf_results['mean_train_r2']
            sample_score_train_std = self.gsf_results['std_train_r2']
            sample_score_test_mean = self.gsf_results['mean_test_r2']
            sample_score_test_std = self.gsf_results['std_test_r2']

            axs[i].fill_between(X_axis, color='r', alpha=0.1, linestyle='-',
                y1=sample_score_train_mean - sample_score_train_std,
                y2=sample_score_train_mean + sample_score_train_std)

            axs[i].fill_between(X_axis, color='g', alpha=0.1, linestyle='--', 
                y1=sample_score_test_mean - sample_score_test_std,
                y2=sample_score_test_mean + sample_score_test_std)
        
            axs[i].plot(X_axis, sample_score_train_mean, 
                        color='r', linestyle='-.'
                    )
            axs[i].plot(X_axis, sample_score_test_mean, 
                        color='g', linestyle='-.'
                    )
        
        fig.savefig('test.png')
        
        # print('Best score', self.gsf_best_score)
        
    
    def model_save(self, pickle_file_name):
        """Save the mode in to pickle format
        """
        pass


def main():
    '''Main function to run the module
    '''

    input_path = join('/home', 'umcg-zzhang', 'Documents', 'projects', 
                      'ASEpredictor', 'outputs', 'biosGavinOverlapCov10')
    input_file = join(input_path, 'biosGavinOlCv10AntUfltCstLog2FCVar.tsv')
    avp = AseVariancePredictor(input_file)
    avp.debug()


if __name__ == '__main__':
    main()
