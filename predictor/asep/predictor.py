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
from sklearn.model_selection import StratifiedKFold
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

from .utilities import draw_k_main_features_cv
from .utilities import draw_roc_curve_cv
from .utilities import make_file_name
from .utilities import format_print
from .utilities import setup_xy
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

    def __init__(self, file_name):
        """Set up basic variables

        Args:
            file_name (str): input data set
        """
        self.input_file_name = file_name

        config = Config()
        self.estimators_list = config.estimators_list
        self.optim_params = config.optim_params

        self.raw_dataframe = None
        self.work_dataframe = None

        self.x_matrix = None
        self.y_vector = None
        self.x_train_matrix = None
        self.y_train_vector = None
        self.x_test_matrix = None
        self.y_test_vector = None

        self.pipeline = None
        self.model = None

    @timmer
    def run(self, sed=3142, limit=100, mask=None, response='ASE'):
        """Execute a pre-designed construct pipeline"""

        self.set_sed(sed)
        self.read_file_to_dataframe(nrows=limit)
        self.setup_work_dataframe()
        self.slice_dataframe(mask=mask)

        self.work_dataframe[response] = self.work_dataframe[response].apply(abs)

        self.simple_imputer()
        self.label_encoder(remove=False)

        cols_discarded = ["log2FC", "binom_p", "binom_p_adj", "group_size"]
        self.slice_dataframe(cols=cols_discarded)

        self.x_matrix, self.y_vector = setup_xy(
            self.work_dataframe, y_col=response
        )
        self.setup_pipeline(estimator=self.estimators_list)
        self.k_fold_stratified_validation(cvs=2)
        self.training_reporter()
        self.draw_learning_curve(self.model, strategy="pipe")

    @staticmethod
    def set_sed(sed=None):
        """Set the random seed of numpy"""
        if sed:
            numpy.random.seed(sed)
        else:
            numpy.random.seed(3142)

    def read_file_to_dataframe(self, nrows=None):
        """Read input file into pandas DataFrame."""
        file_name = self.input_file_name
        try:
            file_hand = open(file_name)
        except PermissionError as err:
            sys.stderr.write('File IO error: ', err)
        else:
            self.raw_dataframe = pandas.read_table(file_hand, nrows=nrows)

    def setup_work_dataframe(self):
        """Deep copy the raw DataFrame into work DataFrame"""
        try:
            self.work_dataframe = copy.deepcopy(self.raw_dataframe)
        except Exception('Failed to deepcopy raw_df to work_df') as exp:
            raise exp

    def slice_dataframe(self, rows=None, cols=None, mask=None, remove=True):
        """Slice the DataFrame base on rows and cols.

        This method will remove or keep rows, columns or any cells match the
        `mask` in place meaning change the dataframe directly, which is time
        and memory sufficient.

        Args:
            rows (list, tuple, None): optional, default None
                Rows retained for the downstream. If it's None, all rows will
                be retained.
            cols (list, tuple, None): optional, default None
                Columns retained for the downstream. If it's None, all columns
                will be retained.
            mask (str, None): optional, default None
                A filter to screen dataframe. If it's a `str` object, `query`
                method will be called; otherwise, if it's `None`, no filter
                will be applied.
            remove (bool): optional, default `True`
                Whether the values of `rows` or `cols` will be kept or
                discarded. If True, cells coorderated by `rows` and `cols` will
                be keep and the exclusive will be discarded, otherwise the way
                around.
        """
        if not isinstance(remove, bool):
            raise TypeError('remove should be bool')

        if remove:
            if rows is not None or cols is not None:
                self.work_dataframe.drop(
                    index=rows, columns=cols, inplace=True
                )

            if mask is not None:
                self.work_dataframe.query(mask, inplace=True)

        else:
            if rows is None:
                rows = self.work_dataframe.index
            if cols is None:
                cols = self.work_dataframe.columns
            self.work_dataframe = self.work_dataframe.loc[rows, cols]

            if mask is not None:
                reverse_mask = "~({})".format(mask)  # reverse of mask
                self.work_dataframe.query(reverse_mask, inplace=True)

    def label_encoder(self, target_cols=None, skip=None, remove=False):
        """Encode category columns

        Args:
            target_cols (list or None):
                name of columns to be encoded
            skip (str, list, tuple, None):
                list of names of columns
                skipped encoded. string represents only the specific column
                will be skipped; list or tuple means all contained elements
                will be skipped; None means no columns will be skipped.
            remove (bool):
                remove columns need to be encoded.
        Raises:
            TypeError:
        """
        if target_cols is None:
            col_types = self.work_dataframe.dtypes
            target_cols = [
                n for n, t in col_types.items() if t is numpy.dtype('O')
            ]

        if isinstance(skip, str):
            if skip in target_cols:
                target_cols.remove(skip)
        elif isinstance(skip, (list, tuple)):
            for skipped in skip:
                if skip in target_cols:
                    target_cols.remove(skipped)
                else:
                    sys.stderr.write('{} isn\'t in list...'.format(skip))

        if remove:
            format_print("Deleted columns (require encode)", target_cols)
            self.work_dataframe.drop(columns=target_cols, inplace=True)
        else:
            format_print("Encoded columns", target_cols)
            target_cols_encod = [n + '_encoded' for n in target_cols]

            encoder = LabelEncoder()
            for col_tag, col_tag_encod in zip(target_cols, target_cols_encod):
                try:
                    self.work_dataframe[col_tag_encod] = encoder.fit_transform(
                        self.work_dataframe[col_tag]
                    )
                    del self.work_dataframe[col_tag]
                except ValueError as err:
                    print(err, file=sys.stderr)

    def simple_imputer(self):
        """A simple imputater based on pandas DataFrame.replace method.

        The columns information are derived from Dannis
        """
        defaults = {
            'motifEName': 'unknown', 'GeneID': 'unknown', 'GeneName': 'unknown',
            'CCDS': 'unknown', 'Intron': 'unknown', 'Exon': 'unknown',
            'ref': 'N', 'alt': 'N', 'Consequence': 'UNKNOWN', 'GC': 0.42,
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
            'dbscSNV.ada_score': 0, 'dbscSNV.rf_score': 0, "mirSVR.Score": 0,
            "mirSVR.E": 0, "mirSVR.Aln": 0
        }

        targets = numpy.NaN
        self.work_dataframe = self.work_dataframe.replace(targets, defaults)

    def train_test_slicer(self, **kwargs):
        """Set up training and testing data set by train_test_split"""
        (self.x_train_matrix, self.x_test_matrix,
         self.y_train_vector, self.y_test_vector
        ) = train_test_split(self.x_matrix, self.y_vector, **kwargs)

    def setup_pipeline(self, estimator=None, biclass=True):
        """Setup a training pipeline

        Args:
            estimator (estimator): None or a list of dicts; optional
                A list with estimator and their parameters
            biclass (bool): optional; defaul True
                Binary classes problem(True) or multiple classes problem(False)
        """
        if biclass:
            self.pipeline = Pipeline(estimator)
        else:
            self.pipeline = OneVsOneClassifier(Pipeline(estimator))

    @timmer
    def random_search(self, estimator=None, **kwargs):
        """Hyper-parameters optimization by RandomizedSearchCV

        Args:
            estimator (estimator): compulsory; scikit-learn estimator object
                An object
            **kwargs: keyword arguments
                Any keyword argument suitable
        """
        if estimator is None:
            estimator = self.pipeline

        random_search = RandomizedSearchCV(estimator, **kwargs)
        self.model = random_search.fit(
            self.x_train_matrix, self.y_train_vector
        )

    @timmer
    def training_reporter(self):
        """Report the training information"""
        format_print('Params', self.model.get_params())
        format_print('Scorer', self.model.scorer_)
        format_print('Best estimator', self.model.best_estimator_)
        format_print('Best params', self.model.best_params_)
        format_print('Best score', self.model.best_score_)
        format_print('Best index', self.model.best_index_)

        prefix = 'cross_validation_random'
        cv_result_file_name = make_file_name(
            file_name='training', prefix=prefix, suffix='tvs'
        )

        cv_results = self.model.cv_results_
        with open(cv_result_file_name, 'w') as cvof:
            pandas.DataFrame(cv_results).to_csv(cvof, sep='\t')

        format_print('Cross-validation results', cv_result_file_name)

        model_score = self.model.score(self.x_test_matrix, self.y_test_vector)
        format_print('Model score', model_score)

    @timmer
    def draw_learning_curve(self, estimator, strategy=None, **kwargs):
        """Draw the learning curve of specific estimator or pipeline

        Args:
            estimator (sklearn estimator): compulsary
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
            estimator.set_params(n_iter=10, cv=6, iid=False)
        else:
            raise Exception('Valid strategy, (None, \'best\', or \'pipe\')')

        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X=self.x_matrix, y=self.y_vector, cv=2, n_jobs=8,
            train_sizes=numpy.linspace(.1, 1., 20), **kwargs
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

    def k_fold_stratified_validation(self, cvs=10, **kwargs):
        """K-fold stratified validation by StratifiedKFold from scikit-learn"""
        skf = StratifiedKFold(n_splits=cvs, **kwargs)

        auc_fpr_tpr_pool = []
        feature_pool = {}
        for idx, (train_idx, test_idx) in enumerate(
                skf.split(self.x_matrix, self.y_vector)):
            self.x_train_matrix = self.x_matrix.iloc[train_idx]
            self.x_test_matrix = self.x_matrix.iloc[test_idx]
            self.y_train_vector = self.y_vector.iloc[train_idx]
            self.y_test_vector = self.y_vector.iloc[test_idx]

            self.random_search(self.pipeline, **self.optim_params)

            fpr, tpr, _ = roc_curve(
                self.y_test_vector,
                self.model.predict_proba(self.x_test_matrix)[:, 1]
            )
            auc_fpr_tpr_pool.append([auc(fpr, tpr), fpr, tpr])

            name_importance = zip(
                self.x_train_matrix.columns[
                    self.model.best_estimator_.steps[0][-1].get_support(True)
                ],
                self.model.best_estimator_.steps[-1][-1].feature_importances_
            )

            for name, importance in name_importance:
                if name in feature_pool:
                    feature_pool[name][idx] = importance
                else:
                    feature_pool[name] = [0] * cvs
                    feature_pool[name][0] = importance

        draw_roc_curve_cv(auc_fpr_tpr_pool)
        draw_k_main_features_cv(feature_pool)
