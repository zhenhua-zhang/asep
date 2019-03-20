#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""configs module"""

import numpy

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

class Config:
    """configs module for the ASEPredictor

    A class to configure the ASEPredictor class. You can use the default
    configuration by using attributes estimators_list, and optim_params. You
    can also load your own configurations by load_config(YOUR-FILE-NAME), but
    please note it will covert the current configurations(`set_default` will
    get you back to the default configs). If you want change the default
    settings, please make a instance and modify it like modifying a Python dict
    object, but please note the structure and import essential modules either
    built-in or third-party ones.

    Attributes:
        estimators_list (list): required, no default
            A list of 2D-tuple, where tuple is (NAME, sklearn_estimator)
        optim_params (dict): options, default dict()
            A `dict` form built-in `collections` module

    Methods:
        set_default(self):
        get_configs(self):
        dump_configs(self, fn=None): dump configurations into a pickle file
        load_configs(self, fn=None):

    Examples:
        >>> from configs import Config
        >>> config = Config()
    """
    def __init__(self):
        """Initializing configuration metrics"""
        self.estimators_list = None
        self.optim_params = dict()
        self.set_default()

    def insert_estimator(self, estimator, position=0):
        """Add more estimator in to estimators_list"""

    def insert_scorer(self, estimator, position=0):
        """Add more scorer for end estimator"""

    def set_params(self, param_name, param_value):
        """Set parameters in optim_params"""

    def set_default(self, estimators_list=None, scoring_dict=None,
                    optim_params=None):
        """Set up default configuration"""

        # For GradientBoostingClassifier
        estimator_params = dict(
            gbc__leaning_rate=numpy.linspace(.01, 1., 50),
            gbc__n_estimators=list(range(50, 1000, 50)),
            gbc__min_samples_split=range(2, 12),
            gbc__min_samples_leaf=range(1, 11),
            gbc__max_features=['sqrt', 'log2', None],
        )

        self.estimators_list = [
            ('fs', SelectKBest()),
            # ('rfc', RandomForestClassifier()),
            # ('gbc', GradientBoostingClassifier()),
            ('brfc', BalancedRandomForestClassifier()),
        ]

        scoring_dict = dict(
            roc_auc_score=make_scorer(roc_auc_score, needs_proba=True),
            precision=make_scorer(precision_score, average="micro"),
            f1_score=make_scorer(f1_score, needs_proba=True),
            accuracy=make_scorer(accuracy_score),
        )

        self.optim_params.update(
            dict(
                cv=StratifiedKFold(n_splits=10, shuffle=True),
                n_jobs=5,
                n_iter=25,
                iid=False,
                scoring=scoring_dict,
                refit="accuracy",
                return_train_score=True,
                param_distributions=dict(
                    fs__k=list(range(3, 90)),
                    fs__score_func=[mutual_info_classif],
                    brfc__min_samples_split=range(2, 10),
                    brfc__min_samples_leaf=range(2, 10),
                    brfc__max_features=['sqrt', 'log2', None],
                    brfc__n_estimators=list(range(50, 1000, 50)),
                    brfc__class_weight=['balanced'],
                    brfc__max_depth=list(range(10, 111, 10)),
                    brfc__bootstrap=[False, True],
                ),
            )
        )

    def get_configs(self):
        """Get current configs"""
        return dict(
            estimators_list=self.estimators_list,
            random_search_parameters=self.optim_params
        )
