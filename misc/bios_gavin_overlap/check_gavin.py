#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load necessary modules
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import statsmodels.stats.multitest as mlt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import math
import sys
import os


# In[ ]:


# Arrange working dirs
hmDir = '/home/umcg-zzhang'
pjDir = os.path.join(hmDir, 'scripts', 'allele_specific_expression',
    'distinguishPathogenicAndNonpathogenic', 'biosGavinOverlapPlots')
pjIpDir = os.path.join(pjDir, 'inputs')
pjOpDir = os.path.join(pjDir, 'outputs')
pjSpDir = os.path.join(pjDir, 'scripts')
pjMcDir = os.path.join(pjDir, 'miscellanies')


# In[ ]:


# Loading file
pjIpFile = os.path.join(pjIpDir, 'biosGavinOverlapCov10AnnoHead1M.tsv')
df = pd.read_table(pjIpFile, header=0, low_memory=False)


# In[ ]:


df.head(2)


# In[ ]:


# Filtering data
covSum = 10
fil = (df.refCountsBios + df.altCountsBios >= covSum) & (df.refCountsBios > 0) & (df.altCountsBios > 0)
df = df.loc[fil, :]


# In[ ]:


# Do exact binomial test
df.loc[:, 'pVal'] = df.loc[:, ['refCountsBios', 'altCountsBios']].apply(sp.stats.binom_test, axis=1)


# In[ ]:


# Two strategy for correction. One for all p values; two for each site.
## Str 1. Considering all p values.
df.loc[:, 'FDROverall'] = mlt.multipletests(df.loc[:, 'pVal'], method='bonferroni')[1]


# In[ ]:


## Str 2. Considering each variant
dfGroups = df.groupby(['chr', 'pos', 'ref', 'alt'])
for name, group in dfGroups:
    FDRPerVariant = mlt.multipletests(group['pVal'], method='bonferroni')[1]
    index = group.index
    df.loc[index, 'FDRPerVariant'] = FDRPerVariant


# In[ ]:


# Add column of log2 fold change
df.loc[:, 'log2FC'] = (df.loc[:, 'altCountsBios']/df.loc[:, 'refCountsBios']).apply(math.log2)


# In[ ]:


# Conbinations pool of colors or markers
pool = [
    ['POPULATION', 'navy'],
    ['BENIGN', 'turquoise'],
    ['PATHOGENIC', 'darkorange']
]
markerDict = [10:'.', 11:'x']


# In[ ]:


# Plots of raw read counts
df = df[(df['log2FC'] >= 1) & (df['FDRPerVariant'] <= 0.05)]

fig, axs = plt.subplots(ncols=3, sharey=True, sharex=True)
fig.set_size_inches((30, 10))
for index, (group, color) in enumerate(pool):
    axs[index].scatter(df.loc[df.loc[:, 'group']==group, 'altCountsBios'], 
               df.loc[df.loc[:, 'group']==group, 'refCountsBios'], 
               color=color, alpha=0.5, s=df.loc[df.loc[:, 'group']==group, 'cadd'])
plt.show()


# In[132]:


# PCA and plots
X = df.loc[:, ['cadd', 'FDROverall', 'log2FC']]
pca = PCA(3, whiten=True)
XFit = pca.fit(X)
Xreduced = pca.transform(X)
covar = pca.get_covariance()

fig, ax = plt.subplots()
for group, color in pool:
    ax.scatter(Xreduced[df['group']==group, 0], Xreduced[df['group']==group, 1], color=color, marker='.', alpha=0.5)
plt.show()

