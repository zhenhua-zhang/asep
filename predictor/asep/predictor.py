#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predicting Allele-specific expression effect"""

import copy
import os
import pickle
import sys
import time
from multiprocessing import Process, Queue

import numpy
import pandas
import scipy
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     learning_curve)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTENC

from .utilities import format_print, set_sed, timmer

try:
    from matplotlib import pyplot
except ImportWarning as warning:
    print(warning, file=sys.stderr)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot



def save_file(filename, target):
    """Save your file smartly"""
    with open(filename, "wb") as opfh:
        if hasattr(target, "savefig"):
            target.savefig(opfh)
        else:
            pickle.dump(target, opfh)


class ASEPredictor:
    """A class implementing prediction of ASE variance of a variant"""

    def __init__(self, file_name, config, sed=3142):
        """Set up basic variables """
        set_sed(sed)
        self.time_stamp = None

        self.input_file_name = file_name

        self.estimators_list = config.estimators_list
        self.optim_params = config.optim_params

        self.raw_dataframe = None
        self.work_dataframe = None

        self.x_matrix = None
        self.y_vector = None

        self.pipeline = None
        self.estimator = None

        self.model_pool = None
        self.training_report_pool = None

        self.feature_importance_pool = None
        self.feature_importance_hist = None

        self.area_under_curve_pool = None
        self.area_under_curve_curve = None

        self.learning_line = None
        self.learning_report = None

        self.label_encoder_matrix = None
        self.mask_query = None
        self.dropped_cols = None

    # trainer
    @timmer
    def trainer(self, limit=None, mask=None, response="bb_ASE", drop_cols=None,
                biclass_=True, outer_cvs=6, mings=2, maxgs=None,
                outer_n_jobs=5, with_lc=False, lc_space_size=10, lc_n_jobs=5,
                lc_cvs=5, nested_cv=False, resampling=None):
        """Execute a pre-designed construct pipeline"""

        self.time_stamp = time.strftime("%Y_%b_%d_%H_%M_%S", time.gmtime())
        self.raw_dataframe = self.read_file(self.input_file_name, limit)

        self.setup_work_dataframe()

        if maxgs:
            __gs_mask = "((group_size >= {:n}) & (group_size <= {:n}))"
            gs_mask = __gs_mask.format(mings, maxgs)
        else:
            gs_mask = "group_size >= {:n}".format(mings)

        self.mask_query = mask
        self.dropped_cols = drop_cols

        self.work_dataframe = self.slice_dataframe(
            self.work_dataframe, mask=gs_mask, remove=False
        )
        self.work_dataframe = self.simple_imputer(self.work_dataframe)
        self.work_dataframe = self.slice_dataframe(
            self.work_dataframe, cols=drop_cols)
        self.work_dataframe[response] = self.work_dataframe[response].apply(
            abs)

        self.label_encoder()
        self.setup_xy(y_col=response, resampling=resampling)
        self.setup_pipeline(self.estimators_list, biclass=biclass_)

        self.outer_validation(
            cvs=outer_cvs, n_jobs=outer_n_jobs, nested_cv=nested_cv
        )

        self.draw_roc_curve_cv()
        self.draw_k_main_features_cv()
        if with_lc:
            self.draw_learning_curve(
                estimator=self.estimator, cvs=lc_cvs, n_jobs=lc_n_jobs,
                space_size=lc_space_size
            )

    @staticmethod
    def read_file(file_name, nrows=None):
        """Read input file into pandas DataFrame."""
        try:
            file_handle = open(file_name)
        except PermissionError as err:
            print('File IO error: ', err, file=sys.stderr)
        else:
            return pandas.read_table(
                file_handle, nrows=nrows, low_memory=False,
                na_values=['NA', '.']
            )

    @timmer
    def setup_work_dataframe(self):
        """Deep copy the raw DataFrame into work DataFrame"""
        try:
            self.work_dataframe = copy.deepcopy(self.raw_dataframe)
        except Exception('Failed to deepcopy raw_df to work_df') as exp:
            raise exp

    @staticmethod
    def slice_dataframe(dataframe, rows=None, cols=None, mask=None,
                        remove=True, mask_first=True):
        """Slice the DataFrame base on rows, columns, and mask."""
        # XXX: mask and rows, cols could be conflict
        if not isinstance(remove, bool):
            raise TypeError('remove should be bool')

        def do_mask(work_dataframe, mask, remove):
            if mask is not None:
                if remove:
                    reverse_mask = "~({})".format(mask)  # reverse of mask
                    work_dataframe.query(reverse_mask, inplace=True)
                else:
                    work_dataframe.query(mask, inplace=True)
            return work_dataframe

        def do_trim(work_dataframe, cols, rows, remove):
            if remove:
                if cols is not None or rows is not None:
                    work_dataframe.drop(
                        index=rows, columns=cols, inplace=True
                    )
            else:
                if rows is None:
                    rows = work_dataframe.index
                if cols is None:
                    cols = work_dataframe.columns
                work_dataframe = work_dataframe.loc[rows, cols]
            return work_dataframe

        if cols:
            cols = [x for x in cols if x in dataframe.columns]

        if mask_first:
            dataframe = do_mask(dataframe, mask=mask, remove=remove)
            dataframe = do_trim(dataframe, cols=cols, rows=rows, remove=remove)
        else:
            dataframe = do_trim(dataframe, cols=cols, rows=rows, remove=remove)
            dataframe = do_mask(dataframe, mask=mask, remove=remove)

        return dataframe

    @timmer
    def setup_xy(self, x_cols=None, y_col=None, resampling=False,
                 cg_features=None):
        """Set up predictor variables and target variables. """
        if cg_features is None:
            cg_features = [
                "ref_encoded", "alt_encoded", "oAA_encoded", "nAA_encoded",
                "motifEHIPos", "CCDS_encoded", "Exon_encoded", "gene_encoded",
                "Type_encoded", "group_encoded", "Segway_encoded",
                "effect_encoded", "impact_encoded", "Intron_encoded",
                "Domain_encoded", "SIFTcat_encoded", "AnnoType_encoded",
                "ConsDetail_encoded", "motifEName_encoded",
                "Dst2SplType_encoded", "Consequence_encoded",
                "PolyPhenCat_encoded",
            ]

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
        x_matrix.sort_index(1, inplace=True)
        y_vector = copy.deepcopy(self.work_dataframe.loc[:, y_col])

        if resampling:
            features = [x_matrix.columns.get_loc(x) for x in cg_features]
            resampler = SMOTENC(features)
            x_matrix, y_vector = resampler.fit_resample(
                x_matrix.values, y_vector.values
            )
            self.x_matrix = pandas.DataFrame(x_matrix, columns=x_cols)
            self.y_vector = pandas.Series(y_vector, name=y_col, dtype='int8')
        else:
            self.x_matrix, self.y_vector = x_matrix, y_vector

    @timmer
    def label_encoder(self, target_cols=None, skip=None, remove=False):
        """Encode category columns """
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
            format_print(
                "Deleted columns (require encoding)", "\n".join(target_cols)
            )
            self.work_dataframe.drop(columns=target_cols, inplace=True)
            self.label_encoder_matrix = {
                (x, x): "removed" for x in target_cols}
        else:
            format_print("Encoded columns", ", ".join(target_cols))
            target_cols_encoded = [n + '_encoded' for n in target_cols]

            if self.label_encoder_matrix is None:
                self.label_encoder_matrix = {}

            encoder = LabelEncoder()
            for _tag, _tag_enc in zip(target_cols, target_cols_encoded):
                try:
                    self.work_dataframe[_tag_enc] = encoder.fit_transform(
                        self.work_dataframe[_tag]
                    )
                    self.label_encoder_matrix[(
                        _tag, _tag_enc)] = copy.deepcopy(encoder)
                    del self.work_dataframe[_tag]
                except ValueError as err:
                    print(err, file=sys.stderr)

    @staticmethod
    def simple_imputer(dataframe, targets=(numpy.NaN, '.')):
        """A simple imputater based on pandas DataFrame.replace method."""
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
            'dbscSNV-ada_score': 0, 'dbscSNV-rf_score': 0
        }

        for target in targets:
            dataframe.replace(target, defaults, inplace=True)

        return dataframe

    @timmer
    def setup_pipeline(self, estimator=None, biclass=True):
        """Setup a training pipeline """
        if biclass:
            self.pipeline = Pipeline(estimator)
        else:
            self.pipeline = OneVsOneClassifier(Pipeline(estimator))

    @timmer
    def draw_learning_curve(self, estimator, cvs=5, n_jobs=5, space_size=10,
                            **kwargs):
        """Draw the learning curve of specific estimator or pipeline"""
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=estimator, X=self.x_matrix, y=self.y_vector,
            train_sizes=numpy.linspace(.1, 1., space_size), cv=cvs,
            n_jobs=n_jobs, **kwargs
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
            label='Training score'
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
            title='Learning curve',
            xlabel='Training examples', ylabel='Score'
        )
        ax_learning.legend(loc='best')

        self.learning_line = (fig, ax_learning)

    @timmer
    def randomized_search_cv(self, estimator, split):  # Nested cv
        """Hyper-parameters optimization by RandomizedSearchCV """
        train_idx, test_idx = split
        x_train_matrix = copy.deepcopy(self.x_matrix.iloc[train_idx])
        y_train_vector = copy.deepcopy(self.y_vector.iloc[train_idx])
        x_test_matrix = copy.deepcopy(self.x_matrix.iloc[test_idx])
        y_test_vector = copy.deepcopy(self.y_vector.iloc[test_idx])

        estimator.fit(x_train_matrix, y_train_vector)

        y_test_scores = estimator.predict_proba(x_test_matrix)[:, 1]
        auc = [
            roc_auc_score(y_test_vector, y_test_scores),
            roc_curve(y_test_vector, y_test_scores)
        ]

        if isinstance(estimator, RandomizedSearchCV):
            training_report = dict(
                Scorer=estimator.scorer_,
                Params=estimator.get_params(),
                Best_params=estimator.best_params_,
                Best_score=estimator.best_score_,
                Best_index=estimator.best_index_,
                Cross_validations=estimator.cv_results_,
                Best_estimator=estimator.best_estimator_,
                Estimator_score=estimator.score(x_test_matrix, y_test_vector)
            )

            estimator = estimator.best_estimator_
        else:
            training_report = None

        # XXX: need update if use more estimator
        first_k_name = x_train_matrix.columns
        first_k_importance = estimator.steps[-1][-1].feature_importances_
        feature_importance = {
            name: importance
            for name, importance in zip(first_k_name, first_k_importance)
        }

        return (training_report, auc, feature_importance, estimator)

    @timmer
    def outer_validation(self, cvs=6, n_jobs=5, nested_cv=False, **kwargs):
        """K-fold stratified validation by StratifiedKFold from scikit-learn"""
        def worker(input_queue, output_queue):
            for func, estimator, split in iter(input_queue.get, 'STOP'):
                output_queue.put(func(estimator, split))

        skf = StratifiedKFold(n_splits=cvs, **kwargs)
        split_pool = skf.split(self.x_matrix, self.y_vector)

        model = RandomizedSearchCV(self.pipeline, **self.optim_params)
        if nested_cv:
            self.estimator = model
        else:
            model.fit(self.x_matrix, self.y_vector)
            self.estimator = model.best_estimator_

            if self.training_report_pool is None:
                self.training_report_pool = [
                    dict(
                        Scorer=model.scorer_,
                        Params=model.get_params(),
                        Best_params=model.best_params_,
                        Best_score=model.best_score_,
                        Best_index=model.best_index_,
                        Cross_validations=model.cv_results_,
                        Best_estimator=model.best_estimator_,
                        Estimator_score=None
                    )
                ]

        task_pool = [
            (self.randomized_search_cv, copy.deepcopy(self.estimator), split)
            for split in split_pool
        ]  # XXX: Deepcopy estimator, memory intensive but much safer ??

        task_queue = Queue()
        for task in task_pool:
            task_queue.put(task)

        result_queue = Queue()
        for _ in range(n_jobs):
            time.sleep(20)
            process = Process(target=worker, args=(task_queue, result_queue))
            process.start()

        if self.model_pool is None:
            self.model_pool = []

        if self.area_under_curve_pool is None:
            self.area_under_curve_pool = []

        if self.training_report_pool is None:
            self.training_report_pool = []

        if self.feature_importance_pool is None:
            self.feature_importance_pool = {
                name: [0] * cvs for name in self.x_matrix.columns
            }

        for cv_idx in range(cvs):
            training_report, auc, feature_importance, model = result_queue.get()

            self.model_pool.append(model)
            self.area_under_curve_pool.append(auc)

            if training_report:
                self.training_report_pool.append(training_report)

            for name, importance in feature_importance.items():
                self.feature_importance_pool[name][cv_idx] = importance

        for _ in range(n_jobs):
            task_queue.put('STOP')

    @timmer
    def draw_roc_curve_cv(self):
        """Draw ROC curve with cross-validation"""
        fig, ax_roc = pyplot.subplots(figsize=(10, 10))
        auc_pool, fpr_pool, tpr_pool = [], [], []
        space_len = 0
        for auc_area, (fpr, tpr, _) in self.area_under_curve_pool:
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

        self.area_under_curve_curve = (fig, ax_roc)

    @timmer
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

    @timmer
    def save_to(self, save_path="./", run_flag=''):
        """Save configs, results and etc. to disk"""
        time_stamp = self.time_stamp
        time_stamp = self.time_stamp + "_" + run_flag
        save_path = os.path.join(save_path, time_stamp)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.feature_importance_pool:
            file_path = os.path.join(save_path, "feature_importances.pkl")
            save_file(file_path, self.feature_importance_pool)

        if self.feature_importance_hist:
            file_path = os.path.join(save_path, "feature_importances_hist.png")
            save_file(file_path, self.feature_importance_hist[0])

        if self.area_under_curve_pool:
            file_path = os.path.join(save_path, "auc_fpr_tpr.pkl")
            save_file(file_path, self.area_under_curve_pool)

        if self.area_under_curve_curve:
            file_path = os.path.join(save_path, "roc_curve.png")
            save_file(file_path, self.area_under_curve_curve[0])

        if self.training_report_pool:
            file_path = os.path.join(save_path, "training_report.pkl")
            save_file(file_path, self.training_report_pool)

        if self.learning_line:
            file_path = os.path.join(save_path, "learning_curve.png")
            save_file(file_path, self.learning_line[0])

        file_path = os.path.join(save_path, time_stamp + "_object.pkl")
        with open(file_path, 'wb') as opfh:
            pickle.dump(self, opfh)

    # Predictor
    @timmer
    def predictor(self, input_file, output_dir="./", model=None,
                  output_file=None, nrows=None):
        """Predict ASE effects for raw dataset"""
        if model is None:
            model = self.fetch_model()
        else:
            if not hasattr(model, "predict"):
                raise AttributeError("Model need `predict` method.")
            if not hasattr(model, "predict_prob"):
                raise AttributeError("Model need `predict_proba` method.")

        with open(input_file) as file_handle:
            target_dataframe = pandas.read_table(
                file_handle, nrows=nrows, low_memory=False,
                na_values=['NA', '.']
            )

        processed_dataframe = self.setup_input_matrix(target_dataframe)

        pre_prob = model.predict_proba(processed_dataframe)
        target_dataframe['pre_prob0'] = pre_prob[:, 0]
        target_dataframe['pre_prob1'] = pre_prob[:, 1]
        target_dataframe['pre'] = model.predict(processed_dataframe)

        if output_file is None:
            _, input_file_name = os.path.split(input_file)
            name, ext = os.path.splitext(input_file_name)
            output_file = "".join([output_dir, name, "_pred", ext])
        else:
            output_file = "".join([output_dir, output_file])

        # XXX: maybe need state data type for each column explicitly
        target_dataframe.to_csv(output_file, sep="\t", index=False)

    @timmer
    def fetch_model(self, best=False):
        """Use specific model to predict new dataset"""
        if self.model_pool is None:
            print("Please train a model first.", file=sys.stderr)
            sys.exit(1)
        else:
            # XXX: add method to get best model
            if best:
                pass
            return copy.deepcopy(self.model_pool[0].steps[-1][-1])

    @timmer
    def setup_input_matrix(self, dataframe, missing_val=1e9-1):
        """Preprocessing inputs to predict"""
        dataframe = self.slice_dataframe(
            dataframe, mask=self.mask_query, remove=False
        )
        # XXX: one of the longest
        dataframe = self.slice_dataframe(dataframe, cols=self.dropped_cols)
        dataframe = self.simple_imputer(dataframe)

        # Encoding new dataframe
        # XXX: one of the longest
        for (_tag, _tag_enc), _encoder in self.label_encoder_matrix.items():
            if _encoder == "removed":
                del dataframe[_tag]
            else:
                classes = _encoder.classes_
                _tmp_dict = dict(zip(classes, _encoder.transform(classes)))
                dataframe[_tag_enc] = dataframe[_tag].apply(
                    lambda x: _tmp_dict.get(x, missing_val)
                )
                del dataframe[_tag]

        return dataframe

    # Validator
    @time
    def validator(self, input_file, output_dir="./", model=None,
                  output_file=None, nrows=None, cv=6):
        """Validate the model using another dataset"""
        self.time_stamp = time.strftime("%Y_%b_%d_%H_%M_%S", time.gmtime())
        self.raw_dataframe = self.read_file(self.input_file_name, limit)

        self.setup_work_dataframe()

        if maxgs:
            __gs_mask = "((group_size >= {:n}) & (group_size <= {:n}))"
            gs_mask = __gs_mask.format(mings, maxgs)
        else:
            gs_mask = "group_size >= {:n}".format(mings)

        self.mask_query = mask
        self.dropped_cols = drop_cols

        self.work_dataframe = self.slice_dataframe(
            self.work_dataframe, mask=gs_mask, remove=False
        )
        self.work_dataframe = self.simple_imputer(self.work_dataframe)
        self.work_dataframe = self.slice_dataframe(
            self.work_dataframe, cols=drop_cols)
        self.work_dataframe[response] = self.work_dataframe[response].apply(
            abs)

        self.label_encoder()
        self.setup_xy(y_col=response, resampling=resampling)
        skf = StratifiedKFold(n_splits=cvs)
        split_pool = skf.split(x_matrix, y_vector)

        y_test_scores = model.predict_proba(processed_dataframe)[:, 1]
        auc = [
            roc_auc_score(y_test_vector, y_test_scores),
            roc_curve(y_test_vector, y_test_scores)
        ]
