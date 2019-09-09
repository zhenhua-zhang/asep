#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predicting Allele-specific expression effect"""

import os
import copy
import time
import pickle

from sys import stderr as STDE
from sys import exit as EXIT

import numpy
import scipy
import joblib
import pandas

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     learning_curve)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .utils import format_print, set_sed, timmer

try:
    from matplotlib import pyplot
except ImportWarning as warning:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot


def save_file(filename, target, svmtd="pickle"):
    """Save your file smartly"""
    with open(filename, "wb") as opfh:
        if hasattr(target, "savefig"):
            target.savefig(opfh)
        elif svmtd == "pickle":
            pickle.dump(target, opfh)
        else:
            joblib.dump(target, opfh)


def check_model_sanity(models):
    """Check the sanity of given model"""
    if not isinstance(models, (list, tuple)):
        models = [models]

    for _model in models:
        if not hasattr(_model, "predict"):
            raise AttributeError("Model require `predict` method.")

        if not hasattr(_model, "predict_prob"):
            raise AttributeError("Model require `predict_proba` method.")

    return True


class ASEP:
    """A class implementing prediction of ASE effect for a variant"""

    def __init__(self, file_name, config, sed=3142):
        """Set up basic variables """
        set_sed(sed)
        self.time_stamp = None

        self.input_file_name = file_name

        self.estimators_list = config.estimators_list
        self.optim_params = config.optim_params

        self.work_dataframe = None
        self.raw_dataframe = None

        self.x_matrix = None
        self.y_vector = None

        self.estimator = None
        self.pipeline = None

        self.training_report_pool = None
        self.model_pool = None

        self.feature_importance_pool = None
        self.feature_importance_hist = None

        self.receiver_operating_characteristic_curve = None
        self.area_under_curve_pool = None

        self.learning_report = None
        self.learning_line = None

        self.label_encoder_matrix = None
        self.dropped_cols = None
        self.mask_query = None
        self.gs_mask = None

    # trainer
    @timmer
    def trainer(self, limit=None, mask=None, response="bb_ASE", drop_cols=None,
                biclass_=True, outer_cvs=6, mings=2, maxgs=None, with_lc=False,
                lc_space_size=10, lc_n_jobs=5, lc_cvs=5, nested_cv=False):
        """Execute a pre-designed construct pipeline"""
        self.time_stamp = time.strftime("%Y_%b_%d_%H_%M_%S", time.gmtime())

        if maxgs:
            _gs_mask = "((group_size >= {:n}) & (group_size <= {:n}))"
            gs_mask = _gs_mask.format(mings, maxgs)
        else:
            gs_mask = "group_size >= {:n}".format(mings)

        self.gs_mask = gs_mask
        self.mask_query = mask
        self.dropped_cols = drop_cols

        (
            self.raw_dataframe, self.work_dataframe,
            self.x_matrix, self.y_vector
        ) = self.preprocessing(
            self.input_file_name, limit, gs_mask, mask, drop_cols, response,
        )

        self.setup_pipeline(self.estimators_list, biclass=biclass_)
        self.outer_validation(cvs=outer_cvs, nested_cv=nested_cv)

        self.receiver_operating_characteristic_curve = self.draw_roc_curve_cv(
            self.area_under_curve_pool
        )

        self.feature_importance_hist = self.draw_k_main_features_cv(
            self.feature_importance_pool
        )

        if with_lc:
            self.draw_learning_curve(
                estimator=self.estimator, cvs=lc_cvs, n_jobs=lc_n_jobs,
                space_size=lc_space_size
            )

    @timmer
    def preprocessing(self, file_name, limit=None, gs_mask=None, mask=None,
                      drop_cols=None, response="bb_ASE"):
        """Preprocessing input data set"""
        if mask:
            pass
        raw_dataframe = self.read_file(file_name, limit)
        dataframe = self.setup_work_dataframe(raw_dataframe)
        dataframe = self.slice_dataframe(dataframe, mask=gs_mask, remove=False)
        dataframe = self.simple_imputer(dataframe)
        dataframe = self.slice_dataframe(dataframe, cols=drop_cols)
        dataframe[response] = dataframe[response].apply(abs)
        dataframe = self.label_encoder(dataframe)
        x_matrix, y_vector = self.setup_xy(dataframe, y_col=response)

        return raw_dataframe, dataframe, x_matrix, y_vector

    @staticmethod
    def read_file(file_name, nrows=None):
        """Read input file into pandas DataFrame."""
        try:
            file_handle = open(file_name)
        except PermissionError as err:
            print('File IO error: ', err, file=STDE)
        else:
            return pandas.read_table(
                file_handle, nrows=nrows, low_memory=False,
                na_values=['NA', '.']
            )

    @staticmethod
    def setup_work_dataframe(raw_dataframe):
        """Deep copy the raw DataFrame into work DataFrame"""
        try:
            return copy.deepcopy(raw_dataframe)
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

    @staticmethod
    def setup_xy(work_dataframe, x_cols=None, y_col=None, cg_features=None):
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

        cols = work_dataframe.columns
        if x_cols is None and y_col is None:
            x_cols, y_col = cols[:-1], cols[-1]
        elif x_cols is None:
            x_cols = cols.drop(y_col)
        elif y_col is None:
            y_col = cols[-1]
            if y_col in x_cols:
                raise ValueError('Target column is in predictor columns')

        x_matrix = copy.deepcopy(work_dataframe.loc[:, x_cols])
        x_matrix.sort_index(1, inplace=True)
        y_vector = copy.deepcopy(work_dataframe.loc[:, y_col])

        return (x_matrix, y_vector)

    def label_encoder(self, work_dataframe, target_cols=None, skip=None,
                      remove=False):
        """Encode category columns """
        if target_cols is None:
            col_types = work_dataframe.dtypes
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
                    print('{} isn\'t in list...'.format(skip), file=STDE)

        if remove:
            format_print(
                "Deleted columns (require encoding)", "\n".join(target_cols)
            )
            work_dataframe.drop(columns=target_cols, inplace=True)
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
                    work_dataframe[_tag_enc] = encoder.fit_transform(
                        work_dataframe[_tag]
                    )
                    self.label_encoder_matrix[(
                        _tag, _tag_enc)] = copy.deepcopy(encoder)
                    del work_dataframe[_tag]
                except ValueError as err:
                    print(err, file=STDE)

        return work_dataframe

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
            'dbscSNV-ada_score': 0, 'dbscSNV-rf_score': 0, 'gnomAD_AF': 0.0,
            'pLI_score': 0.303188,
            # Using the mean of
            # fordist_cleaned_exac_r03_march16_z_pli_rec_null_data.txt.gz;
            # A new feature 4th Jul, 2019
        }

        for target in targets:
            dataframe.replace(target, defaults, inplace=True)

        return dataframe

    def setup_pipeline(self, estimator=None, biclass=True):
        """Setup a training pipeline """
        if biclass:
            self.pipeline = Pipeline(estimator)
        else:
            self.pipeline = OneVsOneClassifier(Pipeline(estimator))

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
    def outer_validation(self, cvs=6, nested_cv=False, **kwargs):
        """K-fold stratified validation by StratifiedKFold from scikit-learn"""
        skf = StratifiedKFold(n_splits=cvs, **kwargs)
        split_pool = skf.split(self.x_matrix, self.y_vector)

        model = RandomizedSearchCV(self.pipeline, **self.optim_params)
        if nested_cv:
            self.estimator = model
        else:
            model.fit(self.x_matrix, self.y_vector)
            self.estimator = model.best_estimator_

            # XXX: possible wrong logics
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

        for cv_idx, split in enumerate(split_pool):
            estimator = copy.deepcopy(self.estimator)
            training_report, auc, feature_importance, model \
                    = self.randomized_search_cv(estimator, split)

            self.model_pool.append(model)
            self.area_under_curve_pool.append(auc)

            if training_report:
                self.training_report_pool.append(training_report)

            for name, importance in feature_importance.items():
                self.feature_importance_pool[name][cv_idx] = importance

    @staticmethod
    def draw_roc_curve_cv(area_under_curve_pool):
        """Draw ROC curve with cross-validation"""
        fig, ax_roc = pyplot.subplots(figsize=(10, 10))
        auc_pool, fpr_pool, tpr_pool = [], [], []
        space_len = 0
        for auc_area, (fpr, tpr, _) in area_under_curve_pool:
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

        return (fig, ax_roc)

    @staticmethod
    def draw_k_main_features_cv(feature_importance_pool, first_k=20):
        """Draw feature importance for the model with cross-validation"""
        name_mean_std_pool = []
        for name, importances in feature_importance_pool.items():
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

        return (fig, ax_features)

    @timmer
    def save_to(self, save_path="./", run_flag='', save_method="pickle"):
        """Save configs, results and model to the disk

        Keyword Arguments:
            save_path {str} -- The path of the saved files (default: {"./"})
            run_flag {str} -- The suffix for current run (default: {''})
            save_method {str} -- The method to save serialize model, specific output, etc. (default: {"pickle"})
        """
        # TODO: Finish the save_method parameters
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

        if self.receiver_operating_characteristic_curve:
            file_path = os.path.join(save_path, "roc_curve.png")
            save_file(file_path, self.receiver_operating_characteristic_curve[0])

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
    def predictor(self, input_file, output_dir="./", nrows=None, models=None):
        """Predict ASE effects for raw dataset"""
        with open(input_file) as file_handle:
            dataframe = pandas.read_table(
                file_handle, nrows=nrows, low_memory=False,
                na_values=['NA', '.']
            )

        x_matrix = self.setup_x_matrix(dataframe)

        _new_cols = ["prob0_mean", "prob0_var", "prob1_mean", "prob1_var"]
        _new_vals, *_ = self.get_predict_proba(x_matrix, models)
        for col_key, col_val in zip(_new_cols, _new_vals):
            dataframe[col_key] = col_val

        _, input_file_name = os.path.split(input_file)
        name, ext = os.path.splitext(input_file_name)
        output_file = "".join([name, "_pred", ext])
        output_path = os.path.join(output_dir, output_file)

        dataframe.to_csv(output_path, sep="\t", index=False)

    def get_predict_proba(self, x_matrix, models=None):
        """Get the predicted probability"""

        if models is None:
            models = self.fetch_models()
        else:
            check_model_sanity(models)

        _pre_prob0, _pre_prob1 = [], []
        for model in models:
            _pre_prob = model.predict_proba(x_matrix)
            _pre_prob0.append(_pre_prob[:, 0])
            _pre_prob1.append(_pre_prob[:, 1])

        prob0 = numpy.array(_pre_prob0)
        prob1 = numpy.array(_pre_prob1)

        prob_mean = (
            prob0.mean(axis=0), prob0.var(axis=0),
            prob1.mean(axis=0), prob1.var(axis=0)
        )

        return prob_mean, prob1, prob0

    def fetch_models(self):
        """Use specific model to predict new dataset"""
        if self.model_pool is None:
            print("Please train a model first.", file=STDE)
            EXIT(1)
        else:
            return [copy.deepcopy(m.steps[-1][-1]) for m in self.model_pool]

    def setup_x_matrix(self, dataframe, missing_val=1e9-1):
        """Preprocessing inputs to predict"""
        dataframe = self.slice_dataframe(
            dataframe, mask=self.mask_query, remove=False
        )
        dataframe = self.slice_dataframe(dataframe, cols=self.dropped_cols)
        dataframe = self.simple_imputer(dataframe)

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
    @timmer
    def validator(self, input_file, output_dir="./", limit=None,
                  response="bb_ASE", models=None):
        """Validate the model using another dataset"""
        mask = self.mask_query
        gs_mask = self.gs_mask
        drop_cols = self.dropped_cols

        _, dataframe, x_matrix, y_vector = self.preprocessing(
            input_file, limit, gs_mask, mask, drop_cols, response
        )

        _new_cols = ["prob0_mean", "prob0_var", "prob1_mean", "prob1_var"]
        _new_vals, prob1, _ = self.get_predict_proba(x_matrix, models)
        for col_key, col_val in zip(_new_cols, _new_vals):
            dataframe[col_key] = col_val

        _, input_file_name = os.path.split(input_file)
        name, ext = os.path.splitext(input_file_name)
        output_file = "".join(["validation_", name, "_pred", ext])
        output_path = os.path.join(output_dir, output_file)
        dataframe.to_csv(output_path, sep="\t", index=False)

        auc = [
            [roc_auc_score(y_vector, _prob1), roc_curve(y_vector, _prob1)]
            for _prob1 in prob1
        ]

        auc_opt = os.path.join(output_dir, "validation_roc_auc.pkl")
        with open(auc_opt, 'wb') as auc_opth:
            pickle.dump(auc, auc_opth)

        fig, _ = self.draw_roc_curve_cv(auc)
        file_path = os.path.join(output_dir, "validation_roc_auc.png")
        save_file(file_path, fig)
