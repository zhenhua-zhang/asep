#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""configs module"""

import pickle

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score


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
        pass

    def insert_scorer(self, estimator, position=0):
        """Add more scorer for end estimator"""
        pass

    def set_params(self, param_name, param_value):
        """Set parameters in optim_params"""
        pass

    def set_default(self):
        """Set up default configuration"""
        self.estimators_list = [
            ('feature_selection', SelectKBest()),
            ('rfc', RandomForestClassifier())
        ]

        scoring_dict = dict(
            precision=make_scorer(precision_score, average="micro"),
            accuracy=make_scorer(accuracy_score),
            roc_auc_score=make_scorer(roc_auc_score, needs_proba=True),
        )

        self.optim_params.update(
            dict(
                cv=StratifiedKFold(n_splits=10, shuffle=True),
                n_jobs=5,
                n_iter=20,
                iid=False,  # TODO: need more knowledge to understand iid here
                scoring=scoring_dict,
                refit="roc_auc_score",
                return_train_score=True,
                param_distributions=dict(
                    feature_selection__score_func=[mutual_info_classif],
                    feature_selection__k=list(range(3, 80, 2)),
                    rfc__n_estimators=list(range(50, 1000, 50)),
                    rfc__max_features=['auto', 'sqrt'],
                    rfc__max_depth=list(range(10, 111, 10)),
                    rfc__min_samples_split=[2, 4, 6, 8, 10],
                    rfc__min_samples_leaf=[2, 4, 6, 8, 10]
                ),
            )
        )

    def get_configs(self):
        """Get current configs"""
        return dict(
            estimators_list=self.estimators_list,
            random_search_parameters=self.optim_params
        )

    def dump_configs(self, file_name):
        """Write the config into a file to make life easier.

        Args:
            file_name (str or None): required; default None
        """
        with open(file_name, 'wb') as fnh:
            pickle.dump(self.config_dict, fnh)
