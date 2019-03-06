#!./env/bin/python
# -*- coding: utf-8 -*-

"""Predicting Allele-specific expression effect

Allele-specific expression predictor

Attributes:
    input_file_name (str): data set used to train the model

Methods:
    __init__(self, file_name, verbose=False)

TODO:
    * Eliminate some module level variables
    * Add more input file type
"""

import copy
import sys

import pandas
import numpy

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

try:
    from matplotlib import pyplot
except ImportError as err:
    print(err, file=sys.stderr)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot

from .utilities import make_file_name
from .utilities import format_print
from .utilities import timmer
from .configs import Config


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
            file_name (str): required
        """
        self.input_file_name = file_name

        config = Config()
        self.estimators_list = config.estimators_list
        self.optim_params = config.optim_params

        self.raw_dataframe = None
        self.work_dataframe = None
        self.raw_dataframe_info = dict()
        self.work_dataframe_info = dict()

        self.train_test_df = None

        self.x_cols = None
        self.y_col = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None

        self.pipeline = None

        self.random_search = None
        self.estimator = None

    @timmer
    def run(self, sed=None, limit=None, filter="group_size > 4",
            mult_class=False, cols_discarded=None):
        """Execute a pre-designed construct pipeline"""

        self.set_seed(sed)
        self.raw_dataframe = self.read_file_to_dataframe(nrows=limit)
        self.setup_work_dataframe()
        self.update_work_dataframe_info()

        if not mult_class:
            self.work_dataframe.ASE = self.work_dataframe.ASE.apply(abs)

        self.simple_imputer()
        self.label_encoder(remove=False)

        if cols_discarded is None:
            cols_discarded = ['var', 'mean', 'p_value', 'group_size']

        self.train_test_df = self.slice_data_frame(
            fltout=filter, cols=cols_discarded, rows=[], keep=False
        )

        self.x_cols, self.y_col = self.setup_xy(self.train_test_df, y_col='ASE')
        self.train_test_slicer(test_size=0.1)

        self.setup_pipeline(
            estimators=self.estimators_list, multi_class=mult_class
        )
        self.random_search_opt(self.pipeline, **self.optim_params)

        self.training_reporter()
        self.draw_figures()

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
            file_handle = open(file_name)
        except PermissionError as err:
            print(err, file=sys.stderr)
            return None
        else:
            with file_handle:
                return pandas.read_table(file_handle, nrows=nrows)

    def update_work_dataframe_info(self):
        """Update the information of working dataframe after modifying it"""
        self.work_dataframe_info['shape'] = self.work_dataframe.shape
        self.work_dataframe_info['columns'] = self.work_dataframe.columns
        self.work_dataframe_info['index'] = self.work_dataframe.index

    def update_raw_dataframe_info(self):
        """Update the raw dataframe infromation"""
        self.raw_dataframe_info['shape'] = self.raw_dataframe.shape
        self.raw_dataframe_info['columns'] = self.raw_dataframe.columns
        self.raw_dataframe_info['index'] = self.raw_dataframe.index

    def setup_work_dataframe(self):
        """Deep copy the raw DataFrame into work DataFrame"""
        try:
            self.work_dataframe = copy.deepcopy(self.raw_dataframe)
        except RuntimeError as err:
            print(err, file=sys.stderr)

        self.update_raw_dataframe_info()
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
            axis (0, 1): optional, default 1
                Axis along which the flt is applied. 0 for rows, 1 for column.
        """

        if not isinstance(keep, bool):
            raise TypeError('keep should be bool')

        if not keep and (rows is None or cols is None):
            raise TypeError(
                'if keep is False, neither rows nor cols can be None'
            )

        if isinstance(fltout, str):
            tmp_dataframe = copy.deepcopy(self.work_dataframe.query(fltout))
        elif callable(fltout):
            tmp_dataframe = copy.deepcopy(
                self.work_dataframe[self.work_dataframe.apply(fltout, axis=1)]
            )
        else:
            tmp_dataframe = copy.deepcopy(self.work_dataframe)

        if rows is None and cols is None:
            rows = self.work_dataframe.index
            cols = self.work_dataframe.columns
        elif rows is None:
            rows = self.work_dataframe.index
        elif cols is None:
            cols = self.work_dataframe.columns

        if keep:
            tmp_dataframe = tmp_dataframe.loc[rows, cols]
        else:
            tmp_dataframe = tmp_dataframe.drop(index=rows, columns=cols)

        return tmp_dataframe

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

        if target_cols is None:
            col_types = self.work_dataframe.dtypes
            target_cols = [
                n for n, t in col_types.items() if t is numpy.dtype('O')
            ]
        elif not isinstance(target_cols, list):
            raise TypeError('Need list type...')

        if isinstance(skip, str):
            if skip in target_cols:
                target_cols.remove(skip)
        elif isinstance(skip, (list, tuple)):
            for _ in skip:
                if skip in target_cols:
                    target_cols.remove(_)
                else:
                    print('{} isn\'t in list...'.format(_), file=sys.stderr)
        elif skip is not None:
            raise TypeError('Need list, tuple or str type, or None...')

        target_cols_encoded = [n + '_encoded' for n in target_cols]

        encoder = LabelEncoder()
        for col_name, col_name_encode in zip(target_cols, target_cols_encoded):
            if remove is True:
                del self.work_dataframe[col_name]
                continue

            try:
                self.work_dataframe[col_name_encode] = \
                        encoder.fit_transform(self.work_dataframe[col_name])
                del self.work_dataframe[col_name]
            except ValueError as err:
                print(err, file=sys.stderr)

        self.update_work_dataframe_info()

    def simple_imputer(self):
        """A simple imputater based on pandas DataFrame.replace method.

        The columns information are derived from Dannis
        """
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
            'Freq10000bp': 0.0, 'Rare10000bp': 0.0, 'Sngl10000bp': 0.0,
            'dbscSNV.ada_score': 0.0, 'dbscSNV.rf_score': 0.0,
            'mirSVR.Score': 0.0, 'mirSVR.E': 0.0, 'mirSVR.Aln': 0.0
        }

        to_replace_list = numpy.NaN

        self.work_dataframe = self.work_dataframe.replace(
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

        return (
            copy.deepcopy(dataframe.loc[:, x_cols]),
            copy.deepcopy(dataframe.loc[:, y_col])
        )

    def train_test_slicer(self, **kwargs):
        """Set up training and testing data set by train_test_split"""
        (self.x_train, self.x_test,
         self.y_train, self.y_test
        ) = train_test_split(self.x_cols, self.y_col, **kwargs)

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

        Args:
            estimators (estimator): required; scikit-learn estimator object
                A object
            **kwargs: keyword arguments
                Any keyword argument suitable
        """
        if estimators is None:
            estimators = self.pipeline

        self.random_search = RandomizedSearchCV(estimators, **kwargs)
        self.estimator = self.random_search.fit(self.x_train, self.y_train)

    @timmer
    def training_reporter(self):
        """Report the training information"""

        fitted_model = self.estimator
        # working dataframe information
        format_print('Work dataframe information', self.work_dataframe_info)
        format_print('Params', fitted_model.get_params())
        format_print('Scorer', fitted_model.scorer_)
        format_print('Best estimators', fitted_model.best_estimator_)
        format_print('Best params', fitted_model.best_params_)
        format_print('Best score', fitted_model.best_score_)
        format_print('Best index', fitted_model.best_index_)

        prefix = 'cross_validation_'
        cv_result_fn = make_file_name(
            file_name='training', prefix=prefix, suffix='tvs')

        cv_results = fitted_model.cv_results_
        with open(cv_result_fn, 'w') as cvof:
            tmp_data_frame = pandas.DataFrame(cv_results)
            tmp_data_frame.to_csv(cvof, sep='\t')

        format_print('Cross-validation results', cv_result_fn)

        self.y_pred = fitted_model.predict(self.x_test)
        fitted_model_score = fitted_model.score(self.x_test, self.y_test)
        format_print('Model score', fitted_model_score)

    @timmer
    def draw_learning_curve(self, estimator, strategy=None, **kwargs):
        """Draw the learning curve of specific estimator or pipeline

        Args:
            estimator (sklearn estimators): compulsary
            strategy (str or None): optional, default None
        """
        if strategy is None:
            estimator = copy.deepcopy(self.estimators_list[-1][-1])
            estimator.set_params(n_estimators=100)
        elif strategy == 'best':
            estimator = estimator.best_estimator_
        elif strategy == 'pipe':
            pass
        else:
            raise Exception('Valid strategy, (None, \'best\', or \'pipe\')')

        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X=self.x_cols, y=self.y_col, cv=10, n_jobs=6,
            train_sizes=numpy.linspace(.1, 1., 10), **kwargs
        )

        train_scores_mean = numpy.mean(train_scores, axis=1)
        train_scores_std = numpy.std(train_scores, axis=1)
        test_scores_mean = numpy.mean(test_scores, axis=1)
        test_scores_std = numpy.std(test_scores, axis=1)

        fig, ax_learning = pyplot.subplots(figsize=(10, 10))

        ax_learning.fill_between(
            train_sizes,
            train_scores_mean + train_scores_std,
            train_scores_mean - train_scores_std,
            alpha=0.1
        )
        ax_learning.plot(
            train_sizes, train_scores_mean, color='r', label='Training score'
        )

        ax_learning.fill_between(
            train_sizes,
            test_scores_mean + test_scores_std,
            test_scores_mean - test_scores_std,
            alpha=0.1
        )
        ax_learning.plot(
            train_sizes, test_scores_mean, color='g',
            label='Cross-validation score'
        )

        ax_learning.set(
            title='Learning curve', xlabel='Training examples', ylabel='Score'
        )
        ax_learning.legend(loc='best')

        fig.savefig(
            make_file_name(
                prefix='learning_curve_', suffix='png'
            )
        )

    @timmer
    def draw_roc_curve(self, estimator):
        """Draw ROC curve for test and validate data set"""
        fig, ax_roc = pyplot.subplots(figsize=(10, 10))

        y_test_pred_prob = estimator.predict_proba(self.x_test)[:, 1]
        false_pos, true_pos, _ = roc_curve(self.y_test, y_test_pred_prob)
        roc_area = auc(false_pos, true_pos)
        sample_size = len(false_pos)
        ax_roc.plot(
            false_pos, true_pos, color='r',
            label=' '.join([
                'Testing:',
                'area={:2f}, sample_size={}'.format(roc_area, sample_size)
            ])
        )

        ax_roc.set(
            title='ROC curve', xlabel='False positive rage',
            ylabel='True positive rate'
        )
        ax_roc.legend(loc='best')

        prefix = 'roc_curve_'
        fig.savefig(make_file_name(prefix=prefix, suffix='png'))

    @timmer
    def draw_k_main_features(self, estimator, k=20):
        """Draw feature importance for the model"""
        ftr_slc_est = estimator.best_estimator_.steps[0][-1]
        slc_ftr_idc = ftr_slc_est.get_support(True)

        rfc_ftr_ipt = estimator.best_estimator_.steps[-1][-1]
        rfc_ftr_ipt = rfc_ftr_ipt.feature_importances_

        ftr_nms = self.x_train.columns[slc_ftr_idc]
        ftr_ipt_mtx = list(zip(ftr_nms, rfc_ftr_ipt))

        ftr_ipt_mtx = sorted(ftr_ipt_mtx, key=lambda x: -x[-1])
        ftr_ipt_mtx = ftr_ipt_mtx[:k]

        rfc_ftr_ipt = [x[-1] for x in ftr_ipt_mtx]
        ftr_nms = [x[0] for x in ftr_ipt_mtx]

        fig, ax_features = pyplot.subplots(figsize=(10, 10))
        ax_features.bar(ftr_nms, rfc_ftr_ipt)

        ax_features.set_xticklabels(
            ftr_nms, rotation_mode='anchor', rotation=45,
            horizontalalignment='right'
        )
        ax_features.set(
            title='Feature importances', xlabel='Features', ylabel='Importance'
        )

        prefix = 'feature_importance_'
        fig.savefig(make_file_name(prefix=prefix, suffix='png'))

    @timmer
    def draw_figures(self):
        """Draw learning curve, ROC curve, and feature importance bar graph"""
        estimator = self.estimator
        self.draw_roc_curve(estimator)
        self.draw_k_main_features(estimator)
        self.draw_learning_curve(estimator, strategy='pipe')


if __name__ == '__main__':
    print("Please use this module by import...", file=sys.stderr)
