#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""configs module"""

import pickle

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer


class Config:
    """configs module for the ASEPredictor

    A class to configure the ASEPredictor class. You can use the default
    configuration by using attributes estimators_list, and
    optim_params. You can also load your own configurations by
    load_config(YOUR-FILE-NAME), but please note it will covert the current
    configurations(`set_default` will get you back to the default configs). If
    you want change the default settings, please make a instance and modify it
    like modifying a Python dict object, but please note the structure and
    import essential modules either built-in or third-party ones.

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
        self.config_file_name = None
        self.estimators_list = None
        self.optim_params = dict()

        self.config_dict = dict(
            estimators_list=self.estimators_list,
            random_search_parameters=self.optim_params
        )
        self.set_default()

    def set_default(self):
        """Set up default configuration"""
        self.estimators_list = [
            ('feature_selection', SelectKBest()),
            ('rfc', RandomForestClassifier())
        ]

        scoring_dict = dict(
            precision=make_scorer(precision_score, average="micro"),
            accuracy=make_scorer(accuracy_score)
        )

        self.optim_params.update(
            dict(
                cv=10,
                n_jobs=3,
                n_iter=15,
                iid=False,
                refit="accuracy",
                scoring=scoring_dict,
                param_distributions=dict(
                    feature_selection__score_func=[mutual_info_classif],
                    feature_selection__k=list(range(3, 110, 2)),
                    rfc__n_estimators=list(range(50, 1000, 10)),
                    rfc__max_features=['auto', 'sqrt'],
                    rfc__max_depth=list(range(10, 111, 10)),
                    rfc__min_samples_split=[2, 4, 6, 8, 10],
                    rfc__min_samples_leaf=[2, 4, 6, 8],
                    rfc__bootstrap=[True, False]
                ),
                return_train_score=True,
            )
        )

    def get_configs(self):
        """Get current configs"""
        return self.config_dict

    def dump_configs(self, file_name):
        """Write the config into a file to make life easier.

        Args:
            file_name (str or None): required; default None
        """
        with open(file_name, 'wb') as fnh:
            pickle.dump(self.config_dict, fnh)

    def load_configs(self, file_name):
        """Load saved configurations into memory

        Args:
            file_name (str): required; default None

        Raises:
            IOError: when argument fn is None, raise IOError.
        """
        with open(file_name, 'rb') as file_handler:
            config_dict = pickle.load(file_handler)

            self.config_dict = config_dict
            self.estimators_list = config_dict['estimators_list']
            self.optim_params = config_dict['optim_params']

            self.config_file_name = file_name
