#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main interface for asep

TODO: 1. A module to parse configuration file, which could make life easier.
      2. The `mask` argument in predictor.train() func doesn't function at all.
"""
import os
import sys
import pdb
import copy
import time
import pickle
import argparse

import joblib
import prince
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

from imblearn.ensemble import BalancedRandomForestClassifier


def format_print(title, main_content, pipe=sys.stderr):
    """A method format the prints"""
    head = '-' * 3
    flag = ' '.join([head, title, ": "])
    print(flag, '\n   ', main_content, "\n\n", file=pipe)


def print_flag(subc=None, flag=None):
    """A method to print running flags

    Keyword Arguments:
        subc {String} -- Name of subcommand (default: {None})
        flag {String} -- Any strings used as a flag of current run (default: {None})
    """

    if subc and flag:
        run_flag = "".join([" Subcommand: ", subc, ". Run flag: ", flag, " "])
    elif subc:
        run_flag = "".join([" Subcommand: ", subc, " "])
    elif flag:
        run_flag = "".join([" Run flag: ", flag, " "])
    else:
        run_flag = "-" * 80

    print("{:-^80}".format(run_flag), file=sys.stderr)


def print_args(args, fwd=-1):
    """Print arguments form command lines

    Arguments:
        args {NameSpace} -- A NameSpace containing commandline arguments

    Keyword Arguments:
        fwd {int} -- The number of space used to fill the argument and
        parameter, using an optimized fill-width if it's default (default: {-1})
    """

    print("Arguments for current run: ", file=sys.stderr)
    args_pair = [(_d, _a) for _d, _a in vars(args).items()]
    args_pair = sorted(args_pair, key=lambda x: len(x[0]))

    if fwd == -1:
        fwd = len(args_pair[-1][0]) + 1

    for dst, arg in args_pair:
        print("  {d: <{w}}: {a: <{w}}".format(d=dst, a=str(arg), w=fwd), file=sys.stderr)


class ASEP:
    """A class implementing prediction of ASE effect for a variant"""

    def __init__(self, file_name, config, random_state=42, test_pp=0):
        """Set up basic variables """
        self.time_stamp = None
        self.random_state = random_state

        self.input_file_name = file_name

        self.estimators_list = config.estimators_list
        self.optim_params = config.optim_params

        self.work_dataframe = None
        self.raw_dataframe = None

        self.x_matrix = None
        self.y_vector = None

        self.test_pp = test_pp
        self.x_test_matrix = None
        self.y_test_vector = None

        self.conf_string = None
        self.test_roc_auc = None
        self.test_roc_auc_curve = None

        self.estimator = None
        self.pipeline = None

        self.training_report_pool = None
        self.model_pool = None

        self.feature_importance_pool = None
        self.feature_importance_hist = None

        self.roc_curve_pool = None
        self.area_under_curve_pool = None

        self.learning_report = None
        self.learning_line = None

        self.label_rename_mtrx = None
        self.dropped_cols = None
        self.mask_query = None
        self.gs_mask = None

    # train
    def train(self, limit=None, mask=None, response="bb_ASE", drop_cols=None,
              biclass_=True, outer_cvs=6, mings=2, maxgs=None, with_lc=False,
              lc_space_size=10, lc_n_jobs=5, lc_cvs=5, nested_cv=False,
              max_na_ratio=0.6):
        """Execute a pre-designed construct pipeline"""
        self.time_stamp = time.strftime("%Y_%b_%d_%H_%M_%S", time.gmtime())

        if maxgs:
            gs_mask = "((group_size>={:n})&(group_size<={:n}))".format(mings, maxgs)
        else:
            gs_mask = "group_size >= {:n}".format(mings)

        self.gs_mask = gs_mask
        self.mask_query = mask
        self.dropped_cols = drop_cols
        self.raw_dataframe, self.work_dataframe, self.x_matrix, self.y_vector \
                = self.preprocessing(self.input_file_name, limit, gs_mask, mask,
                                     drop_cols, response, max_na_ratio)
        if 0 < self.test_pp < 1:
            self.x_matrix, self.x_test_matrix, self.y_vector, self.y_test_vector \
                    = train_test_split(self.x_matrix, self.y_vector, test_size=self.test_pp, random_state=self.random_state, shuffle=True)

        self.setup_pipeline(self.estimators_list, biclass=biclass_)
        self.outer_validation(cvs=outer_cvs, nested_cv=nested_cv)
        self.roc_curve_pool = self.draw_roc_curve_cv(self.area_under_curve_pool)
        self.feature_importance_hist = self.draw_k_main_features_cv(self.feature_importance_pool)

        if with_lc:
            self.draw_learning_curve(estimator=self.estimator, cvs=lc_cvs, n_jobs=lc_n_jobs, space_size=lc_space_size)

        return self

    def test(self):
        """Do test on n% left out samples."""
        if self.x_test_matrix is None or self.y_test_vector is None:
            print("[E]: It looks you setted --test-proportion to 0, which means no test dataset will be setup from the whole input dataset.")
            return self

        y_true = self.y_test_vector
        y_pred = self.get_predict_label(self.x_test_matrix)[0]

        # Confusion matrix
        creport = classification_report(y_true, y_pred, labels=[1, 0], target_names=["ASE", "Non-ASE"])
        conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
        conf_matrix = pd.DataFrame(conf_matrix, index=pd.Index(["ASE", "Non-ASE"], name="True"), columns=pd.Index(["ASE", "Non-ASE"], name="Pred"))
        conf_matrix_list = ["Classification report:", creport, "Confusion matrix:", conf_matrix.to_string()]
        self.conf_string = "\n".join(conf_matrix_list) + "\n"  # output

        # Test results
        _new_cols = ["prob0_mean", "prob0_var", "prob1_mean", "prob1_var"]
        _new_vals, prob1, _ = self.get_predict_proba(self.x_test_matrix)
        for col_key, col_val in zip(_new_cols, _new_vals):
            self.x_test_matrix[col_key] = col_val
        self.x_test_matrix["ASE_true"] = y_true
        self.x_test_matrix["ASE_pred"] = y_pred

        # Test RUC and ROC
        self.test_roc_auc = [[roc_auc_score(y_true, _prob1), roc_curve(y_true, _prob1)] for _prob1 in prob1]
        self.test_roc_auc_curve, _ = self.draw_roc_curve_cv(self.test_roc_auc)

        return self

    def preprocessing(self, file_name, limit=None, gs_mask=None, mask=None,
                      drop_cols=None, response="bb_ASE", max_na_ratio=None):
        """Preprocessing input data set"""
        if mask:
            pass
        raw_dataframe = self.read_file(file_name, limit)
        dataframe = self.setup_work_dataframe(raw_dataframe)
        dataframe = self.slice_dataframe(dataframe, mask=gs_mask, remove=False)
        dataframe = self.slice_dataframe(dataframe, cols=drop_cols)

        if isinstance(max_na_ratio, float):
            dataframe = self.remove_high_na_features(dataframe, max_na_ratio)

        dataframe = self.simple_imputer(dataframe)
        dataframe[response] = dataframe[response].apply(abs)
        dataframe = self.label_encoder(dataframe)
        dataframe = shuffle(dataframe, random_state=self.random_state)
        x_matrix, y_vector = self.setup_xy(dataframe, y_col=response)

        return raw_dataframe, dataframe, x_matrix, y_vector

    @staticmethod
    def read_file(file_name, nrows=None):
        """Read input file into pandas DataFrame."""
        return pd.read_csv(file_name, sep="\t", compression="infer",
                           nrows=nrows, low_memory=False, na_values=['NA', '.'])

    @staticmethod
    def setup_work_dataframe(raw_dataframe, var_id=('Chrom', 'Pos', 'Ref', 'Alt')):
        """Deep copy the raw DataFrame into work DataFrame"""
        try:
            raw_dataframe.index = (raw_dataframe
                                   .loc[:, var_id]
                                   .apply(lambda x: tuple(x.values), axis=1))
            return copy.deepcopy(raw_dataframe)
        except Exception('Failed to deepcopy raw_df to work_df') as exp:
            raise exp

    @staticmethod
    def slice_dataframe(dataframe, rows=None, cols=None, mask=None,
                        remove=True, mask_first=True):
        """Slice the DataFrame base on rows, columns, and mask."""
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
                    work_dataframe.drop(index=rows, columns=cols, inplace=True)
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

    def remove_high_na_features(self, dataframe, max_na_ratio=0.4):
        """Remove features with high NA ratio"""
        na_count_table = dataframe.isna().sum()
        nr_rows, _ = dataframe.shape
        na_freq_table = na_count_table / float(nr_rows)
        _dropped_cols = na_freq_table.loc[na_freq_table >= max_na_ratio].index

        if isinstance(self.dropped_cols, list):
            self.dropped_cols.extend(_dropped_cols)
        else:
            self.dropped_cols = _dropped_cols
        dataframe = dataframe.loc[:, na_freq_table <= max_na_ratio]
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
            target_cols = [n for n, t in col_types.items() if t is np.dtype('O')]

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
            format_print("Deleted cols(require encode)", "\n".join(target_cols))
            work_dataframe.drop(columns=target_cols, inplace=True)
            self.label_rename_mtrx = {(x, x): "removed" for x in target_cols}
        else:
            format_print("Encoded columns", ", ".join(target_cols))
            target_cols_encoded = [n + '_encoded' for n in target_cols]

            if self.label_rename_mtrx is None:
                self.label_rename_mtrx = {}

            encoder = LabelEncoder()
            for _tag, _tag_enc in zip(target_cols, target_cols_encoded):
                try:
                    work_dataframe[_tag_enc] = encoder.fit_transform(work_dataframe[_tag])
                    self.label_rename_mtrx[(_tag, _tag_enc)] = copy.deepcopy(encoder)
                    del work_dataframe[_tag]
                except ValueError as err:
                    print(err, file=sys.stderr)

        return work_dataframe

    @staticmethod
    def simple_imputer(dataframe, targets=(np.NaN, '.')):
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

    def factor_analysis_decomp(self):
        """Do a factor analysis to find latent variables.
        """
        famd = prince.FAMD(
            n_components=2,
            n_iter=3,
            copy=True,
            check_input=True,
            engine="auto",
        )
        famd = famd.fit(self.x_matrix)
        print(famd.row_coordinates)

    def setup_pipeline(self, estimator=None, biclass=True):
        """Setup a training pipeline """
        if biclass:
            self.pipeline = Pipeline(estimator)
        else:
            self.pipeline = OneVsOneClassifier(Pipeline(estimator))

    def draw_learning_curve(self, estimator, cvs=5, n_jobs=5, space_size=10, **kwargs):
        """Draw the learning curve of specific estimator or pipeline"""
        train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=self.x_matrix, y=self.y_vector, train_sizes=np.linspace(.1, 1., space_size), cv=cvs, n_jobs=n_jobs, **kwargs)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        fig, ax_learning = plt.subplots(figsize=(10, 10))

        ax_learning.fill_between(train_sizes, train_scores_mean + train_scores_std, train_scores_mean - train_scores_std, alpha=0.1)
        ax_learning.plot(train_sizes, train_scores_mean, color='r', label='Training score')

        ax_learning.fill_between(train_sizes, test_scores_mean + test_scores_std, test_scores_mean - test_scores_std, alpha=0.1)
        ax_learning.plot(train_sizes, test_scores_mean, color='g', label='Cross-validation score')
        ax_learning.set(title='Learning curve', xlabel='Training examples', ylabel='Score')
        ax_learning.legend(loc='best')

        self.learning_line = (fig, ax_learning)

    def randomized_search_cv(self, estimator, split):  # Nested cv
        """Hyper-parameters optimization by RandomizedSearchCV """
        train_idx, test_idx = split
        x_train_matrix = copy.deepcopy(self.x_matrix.iloc[train_idx])
        y_train_vector = copy.deepcopy(self.y_vector.iloc[train_idx])
        x_test_matrix = copy.deepcopy(self.x_matrix.iloc[test_idx])
        y_test_vector = copy.deepcopy(self.y_vector.iloc[test_idx])

        estimator.fit(x_train_matrix, y_train_vector)

        y_test_scores = estimator.predict_proba(x_test_matrix)[:, 1]
        auc = [roc_auc_score(y_test_vector, y_test_scores), roc_curve(y_test_vector, y_test_scores)]

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
        feature_importance = {name: importance for name, importance in zip(first_k_name, first_k_importance)}

        return (training_report, auc, feature_importance, estimator)

    def outer_validation(self, cvs=6, nested_cv=False, **kwargs):
        """K-fold stratified validation by StratifiedKFold from scikit-learn"""
        skf = StratifiedKFold(n_splits=cvs, random_state=self.random_state, **kwargs)
        split_pool = skf.split(self.x_matrix, self.y_vector)

        model = RandomizedSearchCV(self.pipeline, random_state=self.random_state, **self.optim_params)
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
                name: [0]*cvs for name in self.x_matrix.columns}

        for cv_idx, split in enumerate(split_pool):
            estimator = copy.deepcopy(self.estimator)
            tn_report, auc, ft_impot, model = self.randomized_search_cv(estimator, split)

            self.model_pool.append(model)
            self.area_under_curve_pool.append(auc)

            if tn_report:
                self.training_report_pool.append(tn_report)

            for name, importance in ft_impot.items():
                self.feature_importance_pool[name][cv_idx] = importance

    @staticmethod
    def draw_roc_curve_cv(area_under_curve_pool):
        """Draw ROC curve with cross-validation"""
        fig, ax_roc = plt.subplots(figsize=(10, 10))
        auc_pool, fpr_pool, tpr_pool = [], [], []
        space_len = 0
        for auc_area, (fpr, tpr, _) in area_under_curve_pool:
            auc_pool.append(auc_area)
            fpr_pool.append(fpr)
            tpr_pool.append(tpr)

            if len(fpr) > space_len:
                space_len = len(fpr)

        lspace = np.linspace(0, 1, space_len)
        interp_fpr_pool, interp_tpr_pool = [], []
        for fpr, tpr in zip(fpr_pool, tpr_pool):
            fpr_interped = sp.interp(lspace, fpr, fpr)
            fpr_interped[0], fpr_interped[-1] = 0, 1
            interp_fpr_pool.append(fpr_interped)

            tpr_interped = sp.interp(lspace, fpr, tpr)
            tpr_interped[0], tpr_interped[-1] = 0, 1
            interp_tpr_pool.append(tpr_interped)

        for fpr, tpr in zip(interp_fpr_pool, interp_tpr_pool):
            ax_roc.plot(fpr, tpr, lw=0.5)

        fpr_mean = np.mean(interp_fpr_pool, axis=0)
        tpr_mean = np.mean(interp_tpr_pool, axis=0)
        tpr_std = np.std(interp_tpr_pool, axis=0)

        # A 95% confidence interval for the mean of AUC by Bayesian mvs
        mean, *_ = sp.stats.bayes_mvs(auc_pool)
        auc_mean, (auc_min, auc_max) = mean.statistic, mean.minmax

        label = "Mean: AUC={:0.3}, [{:0.3}, {:0.3}]".format(auc_mean,
                                                            auc_min, auc_max)
        ax_roc.plot(fpr_mean, tpr_mean, color="r", lw=2, label=label)

        mean_upper = np.minimum(tpr_mean + tpr_std, 1)
        mean_lower = np.maximum(tpr_mean - tpr_std, 0)
        ax_roc.fill_between(fpr_mean, mean_upper, mean_lower, color='green',
                            alpha=0.1, label="Standard deviation")
        ax_roc.set(title="ROC curve", xlabel='False positive rate',
                   ylabel='True positive rate')
        ax_roc.plot([0, 1], color='grey', linestyle='--')
        ax_roc.legend(loc="best")

        return (fig, ax_roc)

    @staticmethod
    def draw_k_main_features_cv(feature_importance_pool, first_k=20):
        """Draw feature importance for the model with cross-validation.
        """
        name_mean_std_pool = []
        for name, importances in feature_importance_pool.items():
            mean = np.mean(importances)
            std = np.std(importances, ddof=1)
            name_mean_std_pool.append([name, mean, std])

        name_mean_std_pool = sorted(name_mean_std_pool, key=lambda x: -x[1])

        name_pool, mean_pool, std_pool = [], [], []
        for name, mean, std in name_mean_std_pool[:first_k]:
            name_pool.append(name)
            mean_pool.append(mean)
            std_pool.append(std)

        fig, ax_features = plt.subplots(figsize=(10, 10))
        ax_features.bar(name_pool, mean_pool, yerr=std_pool)
        ax_features.set_xticklabels(name_pool, rotation_mode='anchor',
                                    rotation=45, horizontalalignment='right')
        ax_features.set(
            title="Feature importances(with stand deviation as error bar)",
            xlabel='Feature name', ylabel='Importance')

        return (fig, ax_features)

    def save_to(self, save_path="./", run_flag=''):
        """Save configs, results and model to the disk.
        """
        time_stamp = self.time_stamp + "_" + run_flag
        save_path = os.path.join(save_path, time_stamp)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.feature_importance_pool:
            file_path = os.path.join(save_path, "train_feature_importances.pkl")
            save_file(file_path, self.feature_importance_pool)

        if self.feature_importance_hist:
            file_path = os.path.join(save_path, "train_feature_importances_hist.png")
            save_file(file_path, self.feature_importance_hist[0])

        if self.area_under_curve_pool:
            file_path = os.path.join(save_path, "train_auc_fpr_tpr.pkl")
            save_file(file_path, self.area_under_curve_pool)

        if self.roc_curve_pool:
            file_path = os.path.join(save_path, "train_roc_curve.png")
            save_file(file_path, self.roc_curve_pool[0])

        if self.training_report_pool:
            file_path = os.path.join(save_path, "train_report.pkl")
            save_file(file_path, self.training_report_pool)

        if self.learning_line:
            file_path = os.path.join(save_path, "train_learning_curve.png")
            save_file(file_path, self.learning_line[0])

        if self.conf_string:
            file_path = os.path.join(save_path, "test_report.txt")
            with open(file_path, "w") as opfh:
                opfh.write(self.conf_string)

        if self.x_test_matrix is not None:
            file_path = os.path.join(save_path, "test_output.tsv")
            self.x_test_matrix.to_csv(file_path)

        if self.test_roc_auc:
            file_path = os.path.join(save_path, "test_auc_roc.pkl")
            save_file(file_path, self.test_roc_auc)

        if self.test_roc_auc_curve:
            file_path = os.path.join(save_path, "test_auc_roc.png")
            save_file(file_path, self.test_roc_auc_curve)

        file_path = os.path.join(save_path, "train_model.pkl")
        with open(file_path, 'wb') as opfh:
            pickle.dump(self, opfh)

    # Predictor
    def predictor(self, input_file, output_prefix="./", nrows=None, models=None):
        """Predict ASE effects for raw dataset.
        """
        with open(input_file) as file_handle:
            dataframe = pd.read_table(file_handle, nrows=nrows,
                                      low_memory=False, na_values=['NA', '.'])

        x_matrix = self.setup_x_matrix(dataframe)

        _new_cols = ["prob0_mean", "prob0_var", "prob1_mean", "prob1_var"]
        _new_vals, *_ = self.get_predict_proba(x_matrix, models)
        for col_key, col_val in zip(_new_cols, _new_vals):
            dataframe[col_key] = col_val

        _, input_file_name = os.path.split(input_file)
        name, ext = os.path.splitext(input_file_name)
        output_file = "".join([name, "_pred", ext])
        output_path = output_prefix + output_file

        dataframe.to_csv(output_path, sep="\t", index=False)

    def get_predict_proba(self, x_matrix, models=None):
        """Get the predicted probability.
        """
        if models is None:
            models = self.fetch_models()
        else:
            check_model_sanity(models)

        _pre_prob0, _pre_prob1 = [], []
        for model in models:
            _pre_prob = model.predict_proba(x_matrix)
            _pre_prob0.append(_pre_prob[:, 0])
            _pre_prob1.append(_pre_prob[:, 1])

        prob0 = np.array(_pre_prob0)
        prob1 = np.array(_pre_prob1)
        prob_mean = (prob0.mean(axis=0), prob0.var(axis=0), prob1.mean(axis=0), prob1.var(axis=0))

        return prob_mean, prob1, prob0

    def get_predict_label(self, x_matrix, models=None):
        """Get the predicted labels.
        """
        if models is None:
            models = self.fetch_models()
        else:
            check_model_sanity(models)

        _pre_label = []
        for model in models:
            _pre_label.append(model.predict(x_matrix))

        return _pre_label

    def fetch_models(self):
        """Use specific model to predict new dataset.
        """
        if self.model_pool is None:
            print("Please train a model first.", file=sys.stderr)
            sys.exit(1)
        else:
            return [copy.deepcopy(m.steps[-1][-1]) for m in self.model_pool]

    def setup_x_matrix(self, dataframe, missing_val=1e9-1):
        """Preprocessing inputs to predict.
        """
        dataframe = self.slice_dataframe(dataframe, mask=self.mask_query, remove=False)
        dataframe = self.slice_dataframe(dataframe, cols=self.dropped_cols)
        dataframe = self.simple_imputer(dataframe)

        for (_tag, _tag_enc), _encoder in self.label_rename_mtrx.items():
            if _encoder == "removed":
                del dataframe[_tag]
            else:
                classes = _encoder.classes_
                tmp_dict = dict(zip(classes, _encoder.transform(classes)))
                dataframe[_tag_enc] = dataframe[_tag].apply(lambda x: tmp_dict[x] if x in tmp_dict else missing_val)
                del dataframe[_tag]

        return dataframe

    # Validate
    def validate(self, input_file, output_prefix="./", limit=None, response="bb_ASE", models=None):
        """Validate the model using another dataset.
        """
        mask = self.mask_query
        gs_mask = self.gs_mask
        drop_cols = self.dropped_cols
        if response in drop_cols:
            drop_cols.remove(response)

        rawdtfm, dataframe, x_matrix, y_true = self.preprocessing(input_file, limit, gs_mask, mask, drop_cols, response)

        y_pred = self.get_predict_label(x_matrix, models)[0]
        creport = classification_report(y_true, y_pred, labels=[1, 0], target_names=["ASE", "Non-ASE"])

        creport = classification_report(y_true, y_pred, labels=[1, 0], target_names=["ASE", "Non-ASE"])
        clsf_report = output_prefix + "validation_report.txt"
        conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
        conf_matrix = pd.DataFrame(conf_matrix, index=pd.Index(["ASE", "Non-ASE"], name="True"), columns=pd.Index(["ASE", "Non-ASE"], name="Pred"))

        with open(clsf_report, "w") as fh:
            fh.write("Classification report:\n")
            fh.write(creport)
            fh.write("\nConfusion matrix:\n")
            fh.write(conf_matrix.to_string())
            fh.write("\n")

        _new_cols = ["prob0_mean", "prob0_var", "prob1_mean", "prob1_var"]
        _new_vals, prob1, _ = self.get_predict_proba(x_matrix, models)
        output_dataframe = copy.deepcopy(rawdtfm.loc[dataframe.index, :])
        for col_key, col_val in zip(_new_cols, _new_vals):
            output_dataframe[col_key] = col_val

        _, input_file_name = os.path.split(input_file)
        name, ext = os.path.splitext(input_file_name)
        output_file = "".join(["validation_", name, "_pred", ext])
        output_path = output_prefix + output_file
        output_dataframe.to_csv(output_path, sep="\t", index=False)

        auc = [[roc_auc_score(y_true, _prob1), roc_curve(y_true, _prob1)] for _prob1 in prob1]
        auc_opt = output_prefix + "validation_roc_auc.pkl"

        with open(auc_opt, 'wb') as auc_opth:
            pickle.dump(auc, auc_opth)

        fig, _ = self.draw_roc_curve_cv(auc)
        file_path = output_prefix + "validation_roc_auc.png"
        save_file(file_path, fig)


class Config:
    """Configs module for the ASEPredictor.
    """

    def __init__(self, random_state=42):
        """Initializing configuration metrics.
        """
        self.random_state = random_state

        self.estimators_list = None
        self.optim_params = dict()

        self.searcher_params = None
        self.init_params = None
        self.classifier = None
        self.scorers = None

    def set_init_params(self, classifier="rfc"):
        """A mathod get initial params for classifier.
        """
        if classifier == "abc":  # For AdaboostClassifier
            self.init_params = dict(
                abc__n_estimators=list(range(50, 1000, 50)),
                abc__learning_rate=np.linspace(.01, 1., 50),
                abc__algorithm=["SAMME", "SAMME.R"],
            )
        elif classifier == "gbc":  # For GradientBoostingClassifier
            self.init_params = dict(
                gbc__learning_rate=np.linspace(.01, 1., 50),
                gbc__n_estimators=list(range(50, 1000, 50)),
                gbc__min_samples_split=list(range(2, 12)),
                gbc__min_samples_leaf=list(range(1, 11)),
                gbc__max_depth=list(range(3, 11)),
                gbc__max_features=['sqrt', 'log2', None],
            )
        elif classifier == 'rfc':  # For RandomForestClassifier
            self.init_params = dict(
                rfc__n_estimators=list(range(50, 500, 50)),
                rfc__min_samples_split=list(range(2, 10, 2)),
                rfc__min_samples_leaf=list(range(2, 10, 2)),
                rfc__max_depth=list(range(10, 50, 10)),
                rfc__class_weight=['balanced'],
                # rfc__bootstrap=[False, True],
                # rfc__max_features=['sqrt', 'log2', None],
            )
        elif classifier == 'brfc':  # For BalancedRandomForestClassifier
            self.init_params = dict(
                brfc__n_estimators=list(range(50, 500, 50)),
                brfc__min_samples_split=list(range(2, 10, 2)),
                brfc__min_samples_leaf=list(range(2, 10, 2)),
                brfc__max_depth=list(range(10, 50, 10)),
                brfc__class_weight=['balanced'],
                # brfc__bootstrap=[False, True],
                # brfc__max_features=['sqrt', 'log2', None],
            )
        else:
            raise ValueError("Unknow classifier, choice: abc, gbc, rfc, brfc.")

        return self

    def set_classifier(self, classifier="rfc"):
        """Set classifier"""
        self.set_init_params(classifier=classifier)

        if classifier == "abc":  # For AdaboostClassifier
            self.classifier = ('abc', AdaBoostClassifier(random_state=self.random_state))
        elif classifier == "gbc":  # For GradientBoostingClassifier
            self.classifier = ('gbc', GradientBoostingClassifier(random_state=self.random_state))
        elif classifier == 'rfc':  # For RandomForestClassifier
            self.classifier = ('rfc', RandomForestClassifier(random_state=self.random_state))
        elif classifier == 'brfc':  # For BalancedRandomForestClassifier
            self.classifier = ('brfc', BalancedRandomForestClassifier(random_state=self.random_state))
        else:
            raise ValueError("Unknow classifier, choice [abc, gbc, rfc, brfc]")

        return self

    def set_scorers(self):
        """Set scorer"""
        self.scorers = [
            'f1', 'recall', 'roc_auc', "accuracy", 'f1_micro', "f1_weighted",
            "precision",
        ]
        return self

    def set_searcher_params(self, cvs=None, ncvs=10, n_jobs=5, n_iter=25,
                            refit=None):
        """Set params for the searcher.
        """
        if cvs is None:
            cvs = StratifiedKFold(n_splits=ncvs, shuffle=True, random_state=self.random_state)

        if refit is None:
            refit = 'accuracy'

        self.searcher_params = dict(
            cv=cvs, iid=False, n_jobs=n_jobs, n_iter=n_iter, refit=refit,
            return_train_score=True,
        )

        return self

    def assembly(self):
        """Set up default configuration.
        """
        if self.classifier is None:
            self.set_classifier()

        if self.init_params is None:
            self.set_init_params()

        if self.scorers is None:
            self.set_scorers()

        if self.searcher_params is None:
            self.set_searcher_params()

        if self.estimators_list is None:
            self.estimators_list = [self.classifier]

        self.optim_params['param_distributions'] = self.init_params
        self.optim_params['scoring'] = self.scorers
        self.optim_params.update(self.searcher_params)

        return self


def save_file(filename, target, svmtd="pickle"):
    """Save your file smartly.
    """
    with open(filename, "wb") as opfh:
        if hasattr(target, "savefig"):
            target.savefig(opfh)
        elif svmtd == "pickle":
            pickle.dump(target, opfh)
        else:
            joblib.dump(target, opfh)


def check_model_sanity(models):
    """Check the sanity of given model.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]

    for _model in models:
        if not hasattr(_model, "predict"):
            raise AttributeError("Model require `predict` method.")

        if not hasattr(_model, "predict_prob"):
            raise AttributeError("Model require `predict_proba` method.")

    return True


def cli_parser():
    """A method to get arguments from the command line.
    """
    # default_discarded_cols = ["bb_p", "bb_p_adj", "bn_ASE", "bn_p",
    # "bn_p_adj", "group_size", "log2FC", "Chrom", "Pos", "Annotype",
    # "ConsScore", "ConsDetail", "motifEName", "FeatureID", "GeneID",
    # "GeneName", "CCDS", "Intron", "Exon", "EncExp", "gnomAD_AF"]

    parser = argparse.ArgumentParser()
    _group = parser.add_argument_group("Global") # Global-wide configs
    _group.add_argument("--run-flag", dest="run_flag", default="new_task", help="Flags for current run. The flag will be added to the name of the output dir. Default: %(default)s")
    _group.add_argument("--random-state", dest="random_state", default=None, type=int, help="The random seed. Default: %(default)s")

    subparser = parser.add_subparsers(dest="subcmd") # Arguments parser for sub-command `train`
    train_argparser = subparser.add_parser("train", help="Train a model")

    _group = train_argparser.add_argument_group("Input") # Arguments for Input
    _group.add_argument("-i", "--input-file", dest="input_file", default=None, help="The path to file of training dataset. [Required]")

    _group = train_argparser.add_argument_group("Filter") # Arguments for Filter
    _group.add_argument("-f", "--first-k-rows", dest="first_k_rows", default=None, type=int, help="Only read first k rows as input from input file. Default: %(default)s")
    _group.add_argument("-m", "--mask-as", dest="mask_as", default=None, type=str, help="Pattern will be kept. Default: %(default)s")
    _group.add_argument("-M", "--mask-out", dest="mask_out", default=None, type=str, help="Pattern will be masked. Default: %(default)s")
    _group.add_argument("--min-group-size", dest="min_group_size", default=2, type=lambda x: int(x) > 1 and int(x) or parser.error("--min-group-size must be >= 2"), help="The minimum individuals bearing the same variant(>=2). Default: %(default)s")
    _group.add_argument("--max-group-size", dest="max_group_size", default=1.0E5, type=lambda x: int(x) <= 1e4 and int(x) or parser.error("--max-group-size must be <= 10,000"), help="The maximum number of individuals bearing the same variant (<= 10,000). Default: %(default)s")
    _group.add_argument("--max-na-ratio", dest="max_na_ratio", default=0.6, type=float, help="The maximum ratio of NA in each feature, otherwise, the feature will be abundant")
    _group.add_argument("--drop-cols", dest="drop_cols", default=[], nargs='*', help="The columns will be dropped. Seperated by semi-colon and quote them by ','. if there are more than one columns. Default: %(default)s")
    _group.add_argument("--response-col", dest="response_col", default='bb_ASE', help="The column name of response variable or target variable. Default: %(default)s")

    _group = train_argparser.add_argument_group("Configuration") # Arguments for configuration
    _group.add_argument("--classifier", dest="classifier", default='gbc', type=str, choices=["abc", "gbc", "rfc", "brfc"], help="Algorithm. Choices: [abc, gbc, rfc, brfc]. Default: %(default)s")
    _group.add_argument("--nested-cv", dest="nested_cv", default=False, action="store_true", help="Use nested cross validation or not. Default: %(default)s")
    _group.add_argument("--inner-cvs", dest="inner_cvs", default=6, type=int, help="Fold of cross-validations for RandomizedSearchCV. Default: %(default)s")
    _group.add_argument("--inner-n-jobs", dest="inner_n_jobs", default=5, type=int, help="Number of jobs for RandomizedSearchCV. Default: %(default)s")
    _group.add_argument("--inner-n-iters", dest="inner_n_iters", default=50, type=int, help="Number of iters for RandomizedSearchCV. Default: %(default)s")
    _group.add_argument("--outer-cvs", dest="outer_cvs", default=6, type=int, help="Fold of cross-validation for outer_validation. Default: %(default)s")
    _group.add_argument("--with-learning-curve", dest="with_learning_curve", default=False, action='store_true', help="Whether draw learning curve. Default: %(default)s")
    _group.add_argument("--learning-curve-cvs", dest="learning_curve_cvs", default=4, type=int, help="Number of folds to draw learning curve. Default: %(default)s")
    _group.add_argument("--learning-curve-n-jobs", dest="learning_curve_n_jobs", default=5, type=int, help="Number of jobs to draw learning curves. Default: %(default)s")
    _group.add_argument("--learning-curve-space-size", dest="learning_curve_space_size", default=10, type=int, help="Number of splits created in learning curve. Default: %(default)s")
    _group.add_argument("--test-proportion", dest="test_pp", default=0, type=float, help="Percentage of whole dataset will be used as test dataset.")

    _group = train_argparser.add_argument_group("Output") # Arguments for Output
    _group.add_argument("-o", "--output-prefix", dest="output_prefix", default='./', type=str, help="The directory including output files. Default: ./")
    _group.add_argument("--save-method", dest="save_method", default="pickle", choices=["pickle", "joblib"], help="The library used to save the model and other data set. Choices: pickle, joblib. Default: %(default)s")

    validate_argparser = subparser.add_parser("validate", help="Validate the model.") # Argument parser for subcommand `validate`
    _group = validate_argparser.add_argument_group("Input") # Arguments for Input
    _group.add_argument("-i", "--input-file", dest="input_file", type=str, help="Path to file of validation dataset. [Required]")
    _group.add_argument("-m", "--model-file", dest="model_file", type=str, help="Model to be validated. [Required]")

    _group = validate_argparser.add_argument_group("Filter") # Arguments for Filter
    _group.add_argument("-f", "--first-k-rows", dest="first_k_rows", default=None, type=int, help="Only read first k rows as input from input file. Default: %(default)s")
    _group.add_argument("--response-col", dest="response_col", default='bb_ASE', help="The column name of response variable or target variable. Default: %(default)s")

    _group = validate_argparser.add_argument_group("Output") # Arguments for Output
    _group.add_argument("-o", "--output-prefix", dest="output_prefix", default="./", type=str, help="The directory including output files. Default: ./")

    predict_argparser = subparser.add_parser("predict", help="Apply the model on new data set") # Argument parser for subcommand `predict`
    _group = predict_argparser.add_argument_group("Input") # Arguments for Input
    _group.add_argument("-i", "--input-file", dest="input_file", type=str, required=True, help="New dataset to be predicted. [Required]")
    _group.add_argument("-m", "--model-file", dest="model_file", type=str, required=True, help="Model to be used. [Required]")

    _group = predict_argparser.add_argument_group("Filter") # Arguments for Filter
    _group.add_argument("-f", "--first-k-rows", dest="first_k_rows", default=None, type=int, help="Only read first k rows as input from input file. Default: %(default)s")

    _group = predict_argparser.add_argument_group("Output") # Arguments for Output
    _group.add_argument("-o", "--output-prefix", dest="output_prefix", type=str, help="Output directory for input file. [Reqired]")

    return parser


def train(args):
    """Wrapper entry for `train` subcommand.
    """
    my_config = Config(args.random_state) \
            .set_searcher_params(n_jobs=args.inner_n_jobs,
                                 n_iter=args.inner_n_iters,
                                 ncvs=args.inner_cvs) \
            .set_classifier(args.classifier) \
            .assembly()

    ASEP(args.input_file, my_config, args.random_state, args.test_pp) \
            .train(mask=args.mask_out,
                   mings=args.min_group_size,
                   maxgs=args.max_group_size,
                   limit=args.first_k_rows,
                   response=args.response_col,
                   drop_cols=args.drop_cols,
                   outer_cvs=args.outer_cvs,
                   nested_cv=args.nested_cv,
                   with_lc=args.with_learning_curve,
                   lc_space_size=args.learning_curve_space_size,
                   lc_n_jobs=args.learning_curve_n_jobs,
                   lc_cvs=args.learning_curve_cvs,
                   max_na_ratio=args.max_na_ratio) \
            .test() \
            .save_to(args.output_prefix,
                     run_flag=args.run_flag,
                     save_method=args.save_method)


def validate(args):
    """Validate the model using extra dataset.
    """
    with open(args.model_file, 'rb') as model_file_handle:
        model_obj = pickle.load(model_file_handle)

    model_obj.validate(args.input_file, args.output_prefix, args.first_k_rows,
                       args.response_col)


def predict(args):
    """Predict new dataset based on constructed model.
    """
    with open(args.model_file, 'rb') as model_file_handle:
        model_obj = pickle.load(model_file_handle)
    model_obj.predictor(args.input_file, args.output_prefix, args.first_k_rows)


def main():
    """Main function to run the module.
    """
    parser = cli_parser()
    cli_args = parser.parse_args()

    run_flag = cli_args.run_flag
    cli_args.run_flag = run_flag
    subcmd = cli_args.subcmd

    random_state = cli_args.random_state
    np.random.seed(random_state)

    if subcmd not in ["train", "validate", "predict"]:
        parser.print_help()
    else:
        print_flag(subcmd, run_flag)
        print_args(cli_args)

        if subcmd == "train":
            train(cli_args)
        elif subcmd == "validate":
            validate(cli_args)
        elif subcmd == "predict":
            predict(cli_args)

        print_flag(subcmd, run_flag)


if __name__ == '__main__':
    main()
