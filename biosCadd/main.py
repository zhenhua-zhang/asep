#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#   File Name  : main.py
##  Author     : zhzhang
### E-mail     : zhzhang2015@sina.com
### Created on : Mon 12 Nov 2018 09:19:18 PM CET
##  Version    : <unknown>
#   License    : MIT


 The best parameter values should ALWAYS be CROSS-VALIDATED. Seting up `seed`
 by `numpy.random.seed()`, since scikit-learn used `numpy.random()` throughout.
 A trick to speed up is to use `memory` parameter to cache the estimators
"""


# Load basic modules
import os
import sys
import time
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

ct = time.clock()  # Time counting starts

# Create stream handler of logging
# Logging info formatter
FORMATTER = '%(asctime)s <%(name)s> %(levelname)s: %(message)s'
formatter = logging.Formatter(FORMATTER, '%Y-%m-%d,%H:%M:%S')

# Set up main logging stream and formatter
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

# Set up logging
lg = logging.getLogger()
lg.setLevel(logging.INFO)         # default logging level INFO
lg.addHandler(ch)
lg.info("=== Start ... ===")


# Load necessay modules
lg.info("Load necessay modules...")

# Arrange working dirs and files
pjDir = '/home/umcg-zzhang/Documents/projects/ASEpredictor'
pjOpDir = join(pjDir, 'output')
ipfd = join(pjDir, 'biosGavinOverlapCov10')
ipf = join(pjOpDir, 'biosGavinOverlapCov10Anno.tsv')

# Loading training datasets.
ipdf = pd.read_table(ipf, header=0)


# Setting up random seed
np.random.seed(1234)

# Predictors v.s. responsible variables
X_c, y_c = ['chr', 'pos', 'ref', 'alt', 'cadd', 'group'], ['log2FC']

X, y = ipdf.loc[:, X_c], ipdf.loc[:, y_c]

# Encoding categorical feature?
le = LabelEncoder()
le.fit(X['ref'].append(X['alt']))
X['refEncoded'] = le.transform(X['ref'])
X['altEncoded'] = le.transform(X['alt'])
X['groupEncoded'] = le.fit_transform(X['group'])

XData_c = ['chr', 'pos', 'refEncoded', 'altEncoded', 'cadd', 'groupEncoded']
X_data = X.loc[:, XData_c]

# Split of training datasets.
X_train, X_test, y_train, y_test = train_test_split(X_data, y, random_state=42)


# Check the shape of splitted dataset
print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)
print("y_train.shape: ", y_train.shape)
print("y_test.shape: ", y_test.shape)


# Preprocessing or transformers
# Since we have some outliers, the RobustScaler will be used.
rs = RobustScaler(quantile_range=(25, 75))

# Normalizatoin?
nm = Normalizer()


# Load learning algorithm
rfr = RandomForestRegressor(n_estimators=10)


# Aggregate several steps into a pipeline by Pipeline
rfr_estimators = [
    # ('mi', mi),    # MissingIndicator
    # ('ohe', ohe),  # OneHotEncoder
    ('rs', rs),     # RobustScaler
    ('nm', nm),     # Normalizer
    ('rfr', rfr),   # RandomForestClassifier
]

rfr_pipeline = Pipeline(rfr_estimators)
rfrpf = rfr_pipeline.fit(X_data, y)
rfrm = rfrpf.named_steps['rfr']
rfrm.feature_importances_


# Cross validation metohd 1
# Using k-fold CV (plain CV) or other strategies
plainCV = False

if plainCV:
    cv = 10
else: 
    # optimization need
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

cvs = cross_val_score(rfr_pipeline, X_data, y, cv=cv)
lc = learning_curve(rfr_pipeline, X_data, y, cv=cv, n_jobs=5,
                    train_sizes=np.linspace(0.1, 1.0, 10))

def plot_learning_curve(lc, title, ylim=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    lc : tuple, including train_sizes_abs, train_scores, test_scores
    train_sizes_abs : array, shape (n_unique_ticks,), dtype int
    Numbers of training examples that has been used to generate the learning
    curve. Note that the number of ticks might be less than n_ticks because
    duplicate entries will be removed.

    train_scores : array, shape (n_ticks, n_cv_folds)
    Scores on training sets.

    test_scores : array, shape (n_ticks, n_cv_folds)
    Scores on test set.

    title : string
    Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
    Defines minimum and maximum yvalues plotted.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
    Relative or absolute numbers of training examples that will be used to
    generate the learning curve. If the dtype is float, it is regarded as a
    fraction of the maximum size of the training set (that is determined
    by the selected validation method), i.e. it has to be within (0, 1].
    Otherwise it is interpreted as absolute sizes of the training sets.
    Note that for classification the number of samples usually have to
    be big enough to contain at least one sample from each class.
    (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = lc
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


plot_learning_curve(lc, '7K sample test').show()

# Cross validation method 2
skf = KFold(n_splits=10)
for train, test in skf.split(X_data, y):
    rfrpf = rfr_pipeline.fit(X_data.loc[train, ], y.loc[train, ])
    rfrpfp = rfrpf.predict(X_data.loc[test, ])
    print(explained_variance_score(y.loc[test, ], rfrpfp))
