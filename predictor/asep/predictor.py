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

import pickle
import copy
import time
import sys
import os

import pandas
import numpy
import scipy

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
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

from .utilities import format_print
from .utilities import set_sed
from .utilities import timmer
from .configs import Config

def save_file(filename, target):
    """Save your file smartly"""

    with open(filename, "wb") as opfh:
        if hasattr(target, "savefig"):
            target.savefig(opfh)
        else:
            pickle.dump(target, opfh)


class ASEPredictor:
    """A class implementing prediction of ASE variance of a variant

    Example:
        >>> import ASEPredictor
        >>> ipf = 'input.tsv'
        >>> ap = ASEPredictor(ipf)
        >>> ap.run()
    """

    def __init__(self, file_name, sed=3142):
        """Set up basic variables

        Args:
            file_name (str): input data set
        """
        set_sed(sed)
        self.TIME_STAMP = None

        self.input_file_name = file_name

        self.configuration = Config()
        self.estimators_list = self.configuration.estimators_list
        self.optim_params = self.configuration.optim_params

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

        self.feature_importance_pool = None
        self.feature_importance_hist = None

        self.auc_false_true_pool = None
        self.auc_false_true_curve = None

        self.learning_line = None
        self.learning_report = None

    @timmer
    def run(self, limit=None, mask=None, response="bb_ASE", trim_cols=None,
            biclass_=True, cvs_=2, learning_curve_strategy="pipe",
            output_dir=None):
        """Execute a pre-designed construct pipeline"""

        self.TIME_STAMP = time.strftime("%Y_%b_%d_%H_%M_%S", time.gmtime())

        self.read_file_to_dataframe(nrows=limit)
        self.setup_work_dataframe()
        self.slice_dataframe(mask=mask)

        self.work_dataframe[response] = self.work_dataframe[response].apply(abs)

        self.simple_imputer()
        self.label_encoder()
        self.slice_dataframe(cols=trim_cols)
        self.setup_xy(y_col=response)
        self.setup_pipeline(estimator=self.estimators_list, biclass=biclass_)
        self.k_fold_stratified_validation(cvs=cvs_)
        self.training_reporter()
        self.draw_roc_curve_cv()
        self.draw_k_main_features_cv()
        self.draw_learning_curve(self.model, strategy=learning_curve_strategy)

    def read_file_to_dataframe(self, nrows=None):
        """Read input file into pandas DataFrame."""
        file_name = self.input_file_name
        try:
            file_hand = open(file_name)
        except PermissionError as err:
            print('File IO error: ', err, file=sys.stderr)
        else:
            self.raw_dataframe = pandas.read_table(file_hand, nrows=nrows)

    def setup_work_dataframe(self):
        """Deep copy the raw DataFrame into work DataFrame"""
        try:
            self.work_dataframe = copy.deepcopy(self.raw_dataframe)
        except Exception('Failed to deepcopy raw_df to work_df') as exp:
            raise exp

    def slice_dataframe(self, rows=None, cols=None, mask=None, remove=True,
                        mask_first=True):
        """Slice the DataFrame base on rows, columns, and mask.

        This method will remove or keep rows, columns or any fields match the
        `mask` in place meaning change the dataframe directly, which is time
        and memory sufficient. NOTE: rows or columns will be removed first,
        then the dataframe will be masked.

        Args:
            rows (`list`, `tuple`, `None`): optional, default `None`
                Rows retained for the downstream. If it's `None`, all rows will
                be retained.
            cols (list, tuple, None): optional, default `None`
                Columns retained for the downstream. If it's `None`, all
                columns will be retained.
            mask (str, None): optional, default `None`
                A filter to screen dataframe. If it's a `str` object, `query`
                method will be called; otherwise, if it's `None`, no filter
                will be applied.
            remove (bool): optional, default `True`
                Whether the values of `rows` or `cols` will be kept or
                discarded. If True, cells coorderated by `rows` and `cols` will
                be keep and the exclusive will be discarded, otherwise the way
                around.
            mask_first (bool): optional, defautl `True`
                Do mask first or not.

        TODO:
            Remove and mask can be conflict with each other. For instance, if
            you want to do mask first then do remove second, after one or more
            rows were masked by `mask`, the method won't check whether the
            masked rows in those to be removed.
        """
        if not isinstance(remove, bool):
            raise TypeError('remove should be bool')

        def do_mask(mask, remove):
            if mask is not None:
                if remove:
                    reverse_mask = "~({})".format(mask)  # reverse of mask
                    self.work_dataframe.query(reverse_mask, inplace=True)
                else:
                    self.work_dataframe.query(mask, inplace=True)
            else:
                print("\nMask is empty, skip mask\n", file=sys.stderr)

        def do_trim(cols, rows, remove):
            if remove:
                if cols is not None or rows is not None:
                    self.work_dataframe.drop(
                        index=rows, columns=cols, inplace=True
                    )
            else:
                if rows is None:
                    rows = self.work_dataframe.index
                if cols is None:
                    cols = self.work_dataframe.columns
                self.work_dataframe = self.work_dataframe.loc[rows, cols]

        if mask_first:
            do_mask(mask=mask, remove=remove)
            do_trim(cols=cols, rows=rows, remove=remove)
        else:
            do_trim(cols=cols, rows=rows, remove=remove)
            do_mask(mask=mask, remove=remove)

    def setup_xy(self, x_cols=None, y_col=None):
        """Set up predictor variables and target variables.

        Args:
            x_cols(list, tuple, None):
            y_col(string, None):
        Raises:
            ValueError:
        """
        cols = self.work_dataframe.columns
        if x_cols is None and y_col is None:
            x_cols, y_col = cols[:-1], cols[-1]
        elif x_cols is None:
            x_cols = cols.drop(y_col)
        elif y_col is None:
            y_col = cols[-1]
            if y_col in x_cols:
                raise ValueError('Target column is in predictor columns')

        x_matrix = copy.deepcopy(self.work_dataframe.loc[:, x_cols])
        y_vector = copy.deepcopy(self.work_dataframe.loc[:, y_col])

        self.x_matrix, self.y_vector = x_matrix, y_vector

    def label_encoder(self, target_cols=None, skip=None, remove=False):
        """Encode category columns

        Args:
            target_cols (list or None):
                name of columns to be encoded
            skip (str, list, tuple, None):
                list of names of columns skipped encoded. string represents
                only the specific column will be skipped; list or tuple means
                all contained elements will be skipped; None means no columns
                will be skipped.
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
                    print('{} isn\'t in list...'.format(skip), file=sys.stderr)

        if remove:
            format_print("Deleted columns (require encode)", "\n".join(target_cols))
            self.work_dataframe.drop(columns=target_cols, inplace=True)
        else:
            format_print("Encoded columns", ", ".join(target_cols))
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
            'motifEName': 'unknown', 'GeneID': 'unknown', 'GeneName':
            'unknown', 'CCDS': 'unknown', 'Intron': 'unknown', 'Exon':
            'unknown', 'ref': 'N', 'alt': 'N', 'Consequence': 'UNKNOWN', 'GC':
            0.42, 'CpG': 0.02, 'motifECount': 0, 'motifEScoreChng': 0,
            'motifEHIPos': 0, 'oAA': 'unknown', 'nAA': 'unknown', 'cDNApos': 0,
            'relcDNApos': 0, 'CDSpos': 0, 'relCDSpos': 0, 'protPos': 0,
            'relProtPos': 0, 'Domain': 'UD', 'Dst2Splice': 0, 'Dst2SplType':
            'unknown', 'minDistTSS': 5.5, 'minDistTSE': 5.5, 'SIFTcat': 'UD',
            'SIFTval': 0, 'PolyPhenCat': 'unknown', 'PolyPhenVal': 0,
            'priPhCons': 0.115, 'mamPhCons': 0.079, 'verPhCons': 0.094,
            'priPhyloP': -0.033, 'mamPhyloP': -0.038, 'verPhyloP': 0.017,
            'bStatistic': 800, 'targetScan': 0, 'mirSVR-Score': 0, 'mirSVR-E':
            0, 'mirSVR-Aln': 0, 'cHmmTssA': 0.0667, 'cHmmTssAFlnk': 0.0667,
            'cHmmTxFlnk': 0.0667, 'cHmmTx': 0.0667, 'cHmmTxWk': 0.0667,
            'cHmmEnhG': 0.0667, 'cHmmEnh': 0.0667, 'cHmmZnfRpts': 0.0667,
            'cHmmHet': 0.667, 'cHmmTssBiv': 0.667, 'cHmmBivFlnk': 0.0667,
            'cHmmEnhBiv': 0.0667, 'cHmmReprPC': 0.0667, 'cHmmReprPCWk': 0.0667,
            'cHmmQuies': 0.0667, 'GerpRS': 0, 'GerpRSpval': 0, 'GerpN': 1.91,
            'GerpS': -0.2, 'TFBS': 0, 'TFBSPeaks': 0, 'TFBSPeaksMax': 0,
            'tOverlapMotifs': 0, 'motifDist': 0, 'Segway': 'unknown',
            'EncH3K27Ac': 0, 'EncH3K4Me1': 0, 'EncH3K4Me3': 0, 'EncExp': 0,
            'EncNucleo': 0, 'EncOCC': 5, 'EncOCCombPVal': 0, 'EncOCDNasePVal':
            0, 'EncOCFairePVal': 0, 'EncOCpolIIPVal': 0, 'EncOCctcfPVal': 0,
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
        self.model = random_search.fit(self.x_train_matrix, self.y_train_vector)

    @timmer
    def training_reporter(self):
        """Report the training information"""
        self.learning_report = dict(
            Scorer=self.model.scorer_,
            Params=self.model.get_params(),
            Best_params=self.model.best_params_,
            Best_score=self.model.best_score_,
            Best_index=self.model.best_index_,
            Cross_validations=self.model.cv_results_,
            Best_estimator=self.model.best_estimator_,
            Model_score=self.model.score(self.x_test_matrix,
                                         self.y_test_vector)
        )

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

        self.learning_line = (fig, ax_learning)

    def k_fold_stratified_validation(self, cvs=8, **kwargs):
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

        self.auc_false_true_pool = auc_fpr_tpr_pool
        self.feature_importance_pool = feature_pool

    def draw_roc_curve_cv(self):
        """Draw ROC curve with cross-validation"""
        fig, ax_roc = pyplot.subplots(figsize=(10, 10))
        auc_pool, fpr_pool, tpr_pool = [], [], []
        space_len = 0
        for auc_area, fpr, tpr in self.auc_false_true_pool:
            auc_pool.append(auc_area)
            fpr_pool.append(fpr)
            tpr_pool.append(tpr)

            if len(fpr) > space_len:
                space_len = len(fpr)

        lspace = numpy.linspace(0, 1, space_len)
        interp_fpr_pool, interp_tpr_pool = [], []
        for fpr, tpr in zip(fpr_pool, tpr_pool):
            fpr_interped = scipy.interp(lspace, fpr, fpr)
            fpr_interped[0], fpr_interped[-1] = 0, 1
            interp_fpr_pool.append(fpr_interped)

            tpr_interped = scipy.interp(lspace, fpr, tpr)
            tpr_interped[0], tpr_interped[-1] = 0, 1
            interp_tpr_pool.append(tpr_interped)

        for fpr, tpr in zip(interp_fpr_pool, interp_tpr_pool):
            ax_roc.plot(fpr, tpr, lw=0.5)

        fpr_mean = numpy.mean(interp_fpr_pool, axis=0)
        tpr_mean = numpy.mean(interp_tpr_pool, axis=0)
        tpr_std = numpy.std(interp_tpr_pool, axis=0)

        # A 95% confidence interval for the mean of AUC by Bayesian mvs
        mean, *_ = scipy.stats.bayes_mvs(auc_pool)
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
        ax_roc.set(
            title="ROC curve",
            xlabel='False positive rate', ylabel='True positive rate'
        )
        ax_roc.plot([0, 1], color='grey', linestyle='--')
        ax_roc.legend(loc="best")

        self.auc_false_true_curve = (fig, ax_roc)

    def draw_k_main_features_cv(self, first_k=20):
        """Draw feature importance for the model with cross-validation"""
        name_mean_std_pool = []
        for name, importances in self.feature_importance_pool.items():
            mean = numpy.mean(importances)
            std = numpy.std(importances, ddof=1)
            name_mean_std_pool.append([name, mean, std])

        name_mean_std_pool = sorted(name_mean_std_pool, key=lambda x: -x[1])

        name_pool, mean_pool, std_pool = [], [], []
        for name, mean, std in name_mean_std_pool[:first_k]:
            name_pool.append(name)
            mean_pool.append(mean)
            std_pool.append(std)

        fig, ax_features = pyplot.subplots(figsize=(10, 10))
        ax_features.bar(name_pool, mean_pool, yerr=std_pool)
        ax_features.set_xticklabels(
            name_pool, rotation_mode='anchor', rotation=45,
            horizontalalignment='right'
        )
        ax_features.set(
            title="Feature importances(with stand deviation as error bar)",
            xlabel='Feature name', ylabel='Importance'
        )

        self.feature_importance_hist = (fig, ax_features)

    def save_to(self, save_path="./"):
        """Save configs, results and etc. to disk"""

        time_stamp = self.TIME_STAMP
        save_path = os.path.join(save_path, time_stamp)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_path = os.path.join(save_path, "feature_importances.pkl")
        save_file(file_path, self.feature_importance_pool)

        file_path = os.path.join(save_path, "feature_importances_hist.png")
        save_file(file_path, self.feature_importance_hist[0])

        file_path = os.path.join(save_path, "AUC_false_true_positive_matrix.pkl")
        save_file(file_path, self.auc_false_true_pool)

        file_path = os.path.join(save_path, "roc_curve.png")
        save_file(file_path, self.auc_false_true_curve[0])

        file_path = os.path.join(save_path, "training_report.pkl")
        save_file(file_path, self.learning_report)

        file_path = os.path.join(save_path, "learning_curve.png")
        save_file(file_path, self.learning_line[0])

        file_path = os.path.join(save_path, time_stamp + "_object.pkl")
        with open(file_path, 'wb') as opfh:
            pickle.dump(self, opfh)
