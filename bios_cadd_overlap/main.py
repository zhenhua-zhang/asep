#!/usr/bin/env python
# coding: utf-8

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

import os
import sys
import math
import time
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from os.path import join
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit


warnings.filterwarnings("ignore")

def plot_learning_curve(lc, title, ylim=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):

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

    fig = plt.figure()
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

    fig.set_facecolor('white')

    plt.legend(loc="best")
    return plt


# In[17]:


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

debug = True


# Arrange working dirs and files
pjDir = '/home/umcg-zzhang/Documents/projects/ASEpredictor'
pjOpDir = join(pjDir, 'outputs')
ipfd = join(pjOpDir, 'biosGavinOverlapCov10')

if debug:
    f = 'tmp.tsv'
else:
    f = 'biosGavinOverlapCov10Anno.tsv'
    
ipf = join(ipfd, f)

# Loading training datasets.
if debug:
    nrows = 100000
else:
    nrows = None
    
ipdf = pd.read_table(ipf, header=0, nrows=nrows)
print("Shape of input dataframe(rows, columns):    ", *ipdf.shape, sep="\t")


ipg = ipdf.groupby(['chr', 'pos', 'alt', 'ref'])

# (10, 72358811, 'T', 'G')

# for index, (g_name, group) in enumerate(ipg):
#     if group.group.head(1).all() == 'PATHOGENIC':
#         print(g_name)
#         print(group.group)


# Check the double peaks after filtering by |log2FC| >= 1
# for index, (g_name, group) in enumerate(ipg):
#     if index > 20:
#         break

#     l_, _ = group.shape

#     subGroup = group[((group.log2FC <= -1) | (group.log2FC >= 1))]
#     l, w = subGroup.shape
#     title = ' '.join([str(x) for x in g_name])
#     xlabel = 'log2FC(altAlleleCounts / refAlleleCounts)'
#     if l > 20:
#         fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
#         ax1.set_title(title + "; n: %d" % l_)
#         ax1.set_xlim((-4, 4))
#         group.log2FC.plot.density(ax=ax1)
#         group.log2FC.plot.hist(
#               ax=ax1, density=True, bins=10, color='grey', edgecolor='w')
#         ax1.set_xlabel(xlabel)

#         ax2.set_title(title + '; n: %d (|log2FC| >= 1)' % l)
#         ax2.set_xlim((-4, 4))
#         subGroup.log2FC.plot.density(ax=ax2)
#         subGroup.log2FC.plot.hist(
#               ax=ax2, density=True, bins=10, color='grey', edgecolor='w')
#         ax2.set_xlabel(xlabel)

#         fig.set_dpi(150)
#         fig.set_figwidth(12)
#         fig.set_tight_layout(tight=True)
#         fig.set_facecolor('white')
#         # group.cadd.plot.text(ax=ax)


ipdf = ipdf[((ipdf['log2FC'] >= 1) | (ipdf['log2FC'] <= -1))]
print("Shape of filtered dataframe(rows, columns): ", *ipdf.shape, sep="\t")


# Setting up random seed
np.random.seed(1234)

# Predictors v.s. responsible variables
X_c, y_c = ['chr', 'pos', 'ref', 'alt', 'cadd', 'group', 'FDRPerVariant'], ['log2FC']
X, y = ipdf.loc[:, X_c], ipdf.loc[:, y_c]
X['FDRPerVariant'] = X.FDRPerVariant.apply(lambda x: -math.log10(x))

# Encoding categorical feature
le = LabelEncoder()
le.fit(X['ref'].append(X['alt']))
X['refEncoded'] = le.transform(X['ref'])
X['altEncoded'] = le.transform(X['alt'])
X['groupEncoded'] = le.fit_transform(X['group'])

X_data_c = ['chr', 'pos', 'refEncoded', 'altEncoded', 'cadd', 'groupEncoded']
X_data = X.loc[:, X_data_c]

# Split of training datasets.
X_train, X_test, y_train, y_test = train_test_split(X_data, y, random_state=42)


# Check the shape of splitted dataset
print("X_train.shape: ", *X_train.shape)
print("X_test.shape:  ", *X_test.shape)
print("y_train.shape: ", *y_train.shape)
print("y_test.shape:  ", *y_test.shape)


# Check the dataframe
# X_train.head(10)
# y_train.head(10)


# Preprocessing or transformers
# Since we have some outliers, the RobustScaler will be used.
rs = RobustScaler(quantile_range=(25, 75))

# Normalizatoin?
nm = Normalizer()


# Aggregate several steps into a pipeline by Pipeline
sgdr = SGDRegressor(
    penalty='elasticnet', alpha=0.01, fit_intercept=True
)  # Stochastic Gradient Descent regression

sgdr_estimators = [
    ('rs', rs),     # RobustScaler
    ('nm', nm),     # Normalizer
    ('sgdr', sgdr),   # SGDRegressor
]

sgdr_pipeline = Pipeline(sgdr_estimators)
sgdrpf = sgdr_pipeline.fit(X_data, y)
sgdrm = sgdrpf.named_steps['sgdr']


# Cross validation metohd 1
# Using k-fold CV (plain CV) or other strategies
plainCV = False

if plainCV:
    cv = 10
else:
    # optimization need
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

# 10-fold validation
cvs = cross_val_score(sgdr_pipeline, X_data, y, cv=cv)
print("Cross validation score:", cvs)

lc = learning_curve(sgdr_pipeline, X_data, y, cv=cv, n_jobs=5,
                    train_sizes=np.linspace(0.1, 1.0, 10))

plot_learning_curve(
    lc, '12K samples(Stochastic gradient descent regression)').show()


# Aggregate several steps into a pipeline by Pipeline
lasso = Lasso(alpha=0.01)  # Least Absolute Shrinkage and Oelection Operator

lasso_estimators = [
    ('rs', rs),     # RobustScaler
    ('nm', nm),     # Normalizer
    ('lasso', lasso),   # SGDRegressor
]

lasso_pipeline = Pipeline(lasso_estimators)
lassorpf = lasso_pipeline.fit(X_data, y)
lassorm = lassorpf.named_steps['lasso']


# Cross validation metohd 1
# Using k-fold CV (plain CV) or other strategies
plainCV = False

if plainCV:
    cv = 10
else:  # optimization need
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

# 10-fold validation
cvs = cross_val_score(lasso_pipeline, X_data, y, cv=cv)
print("Cross validation score:", cvs)

lc = learning_curve(lasso_pipeline, X_data, y, cv=cv, n_jobs=5,
                    train_sizes=np.linspace(0.1, 1.0, 10))

plot_learning_curve(lc, '12K sample test(LASSO)').show()


# Aggregate several steps into a pipeline by Pipeline
rfr = RandomForestRegressor(n_estimators=10)  # Random forest regression

rfr_estimators = [
    ('rs', rs),     # RobustScaler
    ('nm', nm),     # Normalizer
    ('rfr', rfr),   # RandomForestClassifier
]

rfr_pipeline = Pipeline(rfr_estimators)
rfrpf = rfr_pipeline.fit(X_data, y)
rfrm = rfrpf.named_steps['rfr']

featureNames = X_train.columns
featureImportance = rfrm.feature_importances_

print("Importance of each feature: ")
for index, value in enumerate(featureNames):
    print(value.replace("Encoded", ''), ":", featureImportance[index])

# Cross validation metohd 1
# Using k-fold CV (plain CV) or other strategies
plainCV = False

if plainCV:
    cv = 10
else:
    # optimization need
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

# 10-fold validation
cvs = cross_val_score(rfr_pipeline, X_data, y, cv=cv)
print("Cross validation score:", cvs)

lc = learning_curve(rfr_pipeline, X_data, y, cv=cv, n_jobs=5,
                    train_sizes=np.linspace(0.1, 1.0, 10))

plot_learning_curve(lc, '12K sample test(Random forest regression)').show()


# Cross validation method 2
skf = KFold(n_splits=10)

for train, test in skf.split(X_data, y):
    rfrpf = rfr_pipeline.fit(X_data.loc[train, ], y.loc[train, ])
    rfrpfp = rfrpf.predict(X_data.loc[test, ])


# Check the variance of log2FC for one variant
# alt_counts / ref_counts
ipg = ipdf.groupby(['chr', 'pos', 'alt', 'ref'])

ax_nums = 5

counter = 0
percentage_dict = {}


for index, (g_name, group) in enumerate(ipg):
    length, _ = group.shape
    percentage_keys = percentage_dict.keys()

    if length in percentage_keys:
        percentage_dict[length]
    else:
        percentage_dict[length] = {
            'altPreferenceCounter': 0,
            'refPreferenceCounter': 0,
            'mixedCounter': 0}

    negativeLog2FC = group[group['log2FC'] <= -1]
    positiveLog2FC = group[group['log2FC'] >= 1]

    lNegativeLog2FC, wNegativeLog2FC = negativeLog2FC.shape
    lPositiveLog2FC, wPositiveLog2FC = positiveLog2FC.shape

    if lNegativeLog2FC == 0:
        percentage_dict[length]['altPreferenceCounter'] += 1
    elif lPositiveLog2FC == 0:
        percentage_dict[length]['refPreferenceCounter'] += 1
    else:
        percentage_dict[length]['mixedCounter'] += 1

#     if length > 300 and lPositiveLog2FC != 0 and lNegativeLog2FC==0:
# the outlier is (10, 88820789, 'C', 'T')
#         print(group.log2FC, g_name)

for x, y in percentage_dict.items():
    sumPerVariant = percentage_dict[x]['altPreferenceCounter'] \
            + percentage_dict[x]['refPreferenceCounter'] \
            + percentage_dict[x]['mixedCounter']

    percentage_dict[x]['altPreferenceCounter'] /= sumPerVariant / 100.0
    percentage_dict[x]['refPreferenceCounter'] /= sumPerVariant / 100.0
    percentage_dict[x]['mixedCounter'] /= sumPerVariant / 100.0


percentage_dataframe = pd.DataFrame(percentage_dict)
FIG, (ax1, ax2) = plt.subplots(nrows=2)

x = list(percentage_dataframe.columns,)
y = np.vstack(percentage_dataframe)

ax1.bar(x, percentage_dataframe.loc['mixedCounter', ])
ax1.bar(x, percentage_dataframe.loc['refPreferenceCounter', ],
        bottom=percentage_dataframe.loc['mixedCounter', :])
ax1.bar(x, percentage_dataframe.loc['altPreferenceCounter', :],
        bottom=percentage_dataframe.loc['mixedCounter', :]
        + percentage_dataframe.loc['refPreferenceCounter', :])
ax1.set_xlim((0, ax1.get_xlim()[-1]/2))

ax2.bar(x, percentage_dataframe.loc['mixedCounter', ])
ax2.bar(x, percentage_dataframe.loc['refPreferenceCounter', ],
        bottom=percentage_dataframe.loc['mixedCounter', :])

ax2.bar(x, percentage_dataframe.loc['altPreferenceCounter', :],
        bottom=percentage_dataframe.loc['mixedCounter', :] +
        percentage_dataframe.loc['refPreferenceCounter', :])

ax2.set_xlim((ax2.get_xlim()[-1]/2, ax2.get_xlim()[-1]))

FIG.set_figwidth(25)
plt.legend(('mixedCounter', 'refPreferenceCounter', 'altPreferenceCounter'))
plt.xlabel('Num of samples supporting one variant')
plt.show()

#     if counter >= ax_nums:
#         break

#     if length == 5:
#         fig, ax = plt.subplots(1)

#         print(g_name, '| %3.2f' % group.cadd.mode(), '| %d' % length)
#         group.log2FC.hist(alpha=0.9, grid=False, ax=ax)
#         plt.title('_'.join([str(x) for x in g_name]))
#         counter += 1

# Some instance.
# Variant id               | CADD  | num of samples | log2FC
# (10, 72358655, 'A', 'G') | 0.07  | 3              | >= 1

# 如图所示, 数据呈现一对多的特点, 即存在多个具有不同log2FC的样本具有相同的注释
# 结果(如cadd). 因而需要新的features.

# As shown in the figure. For one variant with the identical annotations(e.g
#   CADD score), the log2FC could be less than -1 or greater than 1. Therefore, 
#   more specific features are needed. By specific, I mean: information that
#   can make the observered variant in one sample different from the one from
#   another sample. For instance: combination of up and down stream information
#   of the observed variant.

# 可以添加的 feature 举例:
#  1) 位于观测variant上游和下游且与之位于相同的原件(intro或exon)的variant
#    (1~2个), 同时添加一列两者之间的遗传距离或碱基距离
#  2) 上游最近的snp(pos, ref, alt, cadd, log2FC, genetic distance, physical
#    distance, consistent exon or intron?)

# Potential features could be, for instance, information of variants that is
#  1)in the same elenment(e.g intron/exon) with the observed variant,
#  2)from the up or down-stream of the target variant. Meanwhile the genetic
#    distance or/and physical distance between the observed variant and
#    adjacent variants.
