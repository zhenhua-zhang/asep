#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""configs module"""

import numpy

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import accuracy_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold


class Config:
    """configs module for the ASEPredictor

    Examples:
        >>> from config import Config
        >>> config = Config()
        >>> config.init()
    """

    def __init__(self, random_state=31415):
        """Initializing configuration metrics"""
        self.random_state = random_state
        self.estimators_list = None
        self.optim_params = dict()

        self.searcher_params = None
        self.init_params = None
        self.classifier = None
        self.scorers = None

    def set_init_params(self, classifier="rfc"):
        """A mathod get initial params for classifier"""
        if classifier == "abc":  # For AdaboostClassifier
            self.init_params = dict(
                abc__n_estimators=list(range(50, 1000, 50)),
                abc__learning_rate=numpy.linspace(.01, 1., 50),
                abc__algorithm=["SAMME", "SAMME.R"],
            )
        elif classifier == "gbc":  # For GradientBoostingClassifier
            self.init_params = dict(
                gbc__learning_rate=numpy.linspace(.01, 1., 50),
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
        elif classifier == 'brfc':  # For RandomForestClassifier
            self.init_params = dict(
                rfc__n_estimators=list(range(50, 500, 50)),
                rfc__min_samples_split=list(range(2, 10, 2)),
                rfc__min_samples_leaf=list(range(2, 10, 2)),
                rfc__max_depth=list(range(10, 50, 10)),
                rfc__class_weight=['balanced'],
                # rfc__bootstrap=[False, True],
                # rfc__max_features=['sqrt', 'log2', None],
            )
        else:
            raise ValueError("Unknow classifier, choice: abc, gbc, rfc, brfc.")

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

    def set_scorers(self):
        """Set scorer"""
        self.scorers = [
            'f1', 'recall', 'roc_auc', "accuracy", 'f1_micro', "f1_weighted",
            "precision",
        ]

    def __set_scorers(self, extra_scorer=None):
        """Set scorer"""
        basic_scorers = dict(
            precision=make_scorer(precision_score, average="micro"),
            accuracy=make_scorer(accuracy_score),
        )

        if extra_scorer:
            basic_scorers['extra_scorer'] = extra_scorer
        self.scorers = basic_scorers

    def set_searcher_params(self, cvs=None, ncvs=10, n_jobs=5, n_iter=25,
                            refit=None):
        """Set params for the searcher"""
        if cvs is None:
            cvs = StratifiedKFold(n_splits=ncvs, shuffle=True)

        if refit is None:
            refit = 'accuracy'

        self.searcher_params = dict(
            cv=cvs, iid=False, n_jobs=n_jobs, n_iter=n_iter, refit=refit,
            return_train_score=True,
        )

    def assembly(self):
        """Set up default configuration"""

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
