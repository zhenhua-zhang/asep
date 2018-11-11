
# coding: utf-8

# In[1]:

#!/usr/bin/env python3

#
##
### author: zhzhang
### e-mail: zhzhang2015@sina.com / zhenhua.zhang@sina.com
### date  : 2018.10.25
##
################################################################################


# In[2]:

# Load necessary modules
import os
import time
import logging


# In[3]:

ct = time.clock()  # Time counting starts


# In[4]:

# Create stream handler of logging
## Logging info formatter
FORMATTER = '%(asctime)s <%(name)s> %(levelname)s: %(message)s'
formatter = logging.Formatter(FORMATTER, '%Y-%m-%d,%H:%M:%S')

## Set up main logging stream and formatter
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

# Set up logging
lg = logging.getLogger()
lg.setLevel(logging.INFO)         # default logging level INFO
lg.addHandler(ch)
lg.info("=== Start ... ===")


# In[5]:

# Load necessay modules
lg.info('Load necessay modules...')
import statsmodels.stats.multitest as mlt
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import pandas as pd
import scipy as sp
import numpy as np
import math

from sklearn.decomposition import PCA
from scipy import stats


# In[6]:

# Arrange working dirs
lg.info('Arrange working dirs...')
hmDir = '/home/umcg-zzhang'
pjDir = os.path.join(hmDir, 'projects', 'ASEpredictor')
pjIpDir = os.path.join(pjDir, 'inputs')
pjOpDir = os.path.join(pjDir, 'outputs', 'biosGavinOverlapCov10')
pjSpDir = os.path.join(pjDir, 'scripts')
pjMcDir = os.path.join(pjDir, 'miscellanies')


# In[7]:

# Arrange input and output files
lg.info('Arrange input and output files...')
pjIpFile = os.path.join(pjOpDir, 'biosGavinOverlapCov10Anno.tsv')
pjOpUfFile = os.path.join(pjOpDir, 'biosGavinOverlapCov10AnnoUnFiltered.tsv')
pjOpFcFile = os.path.join(pjOpDir, 'biosGavinOverlapCov10AnnoFilteredByLog2FC.tsv')
pjOpAfFile = os.path.join(pjOpDir, 'biosGavinOverlapCov10AnnoFilteredByAf0.001.tsv')
rawRdCtPlot = os.path.join(pjOpDir, 'rawReadCounts.png')
rawLsRdCtPlot = os.path.join(pjOpDir, 'rawLess500ReadCounts.png')
PCAPlot = os.path.join(pjOpDir, 'PCA.png')


# In[8]:

# Conbinations pool of colors or markers
pool = [['POPULATION', 'navy'], 
        ['BENIGN', 'turquoise'], 
        ['PATHOGENIC', 'darkorange'] ]
markerDict = {10:'.', 11:'x'}

if True:
    # Loading file
    lg.info("Reading file ...")
    df = pd.read_table(pjIpFile, header=0, low_memory=False)


    # Filtering data
    covSum = 10
    fil = (df.refCountsBios + df.altCountsBios >= covSum)             & (df.refCountsBios > 0)             & (df.altCountsBios > 0)
    lg.info("Filtering ...")
    df = df.loc[fil, :]


    # Do exact binomial test
    lg.info("Binomial test ...")
    df['pVal'] = df.loc[:, ['refCountsBios', 'altCountsBios']
                       ].apply(stats.binom_test, axis=1)


    # Two strategies for correction. One for all p values; two for each site.
    ## Str 1. Considering all p values.
    lg.info("Overall adjust ...")
    df['FDROverall'] = mlt.fdrcorrection(df.pVal)[1]

    ## Str 2. Considering each variant
    lg.info("Variant-wised adjust ...")
    dfGroups = df.groupby(['chr', 'pos', 'ref', 'alt'])
    for name, group in dfGroups:
        index = group.index

        ## Adjustment of p-values because of multiple test
        FDRPerVariant = mlt.fdrcorrection(group.pVal)[1]
        df.loc[index, 'FDRPerVariant'] = FDRPerVariant
        
        ## chi2_contigency test for the identical variant.
        ctgTable = group.loc[:, ['refCountsBios', 'altCountsBios']]
        g, p, dof, expctd = stats.chi2_contingency(ctgTable, lambda_='log-likelihood')
        df.loc[index, 'varInsideChi2Pval'] = p


    # Add coloumn of log2 fold change
    lg.info("Calculating log 2 fold change ...")
    df['log2FC'] = ( df.loc[:, "altCountsBios"] / df.loc[:, 'refCountsBios']).apply(math.log2)
    
    # Write Unfiltered Df in to a file
    lg.info('Writing unfiltered file in to the drive ...')
    df.to_csv(pjOpUfFile, header=True, index=False, sep='\t')
    
    # Write DF filtered by AF and FDRPerVariant
    af = 0.001
    lg.info("Applying FILTER on unfiltered dataset ...")
    dfFltAF = df[((df.gnomad_AF <= af)  & (df.FDRPerVariant <= 0.05))]

    lg.info('Writting DF filtered by AF and FDRPerVariant into the drive ...')
    dfFltAF.to_csv(pjOpAfFile, header=True, index=False, sep='\t')
    lg.info('Removing extra variables ...')
    del dfFltAF
    
    # Write the filtered DF into a file
    lg.info("Applying FILTER on unfiltered dataset ...")
    dfFltLog2FC = df[((df.log2FC >= 1) | (df.log2FC <= -1)) & (df.FDRPerVariant <= 0.05)]
    
    lg.info("Writing DF filtered by log2FC into the drive ...")
    dfFltLog2FC.to_csv(pjOpFcFile, header=True, index=False, sep='\t')
    
else:
    lg.info('Skipping data preprocessing, willl use output of last run ...')
    df = pd.read_table(pjOpFcFile, header=0, low_memory=False)


# In[15]:

# Plots of raw read counts
lg.info("Start drawing ...")
fig, axs = plt.subplots(ncols=4, sharey=True, sharex=True)

fig.set_size_inches((40, 10))

for i, (g, c) in enumerate(pool):
    x = df.loc[df.group==g, 'altCountsBios']
    y = df.loc[df.group==g, 'refCountsBios']
    s = df.loc[df.group==g, 'cadd']
    axs[i].scatter(x, y, c=c, s=s, alpha=0.5)
    axs[3].scatter(x, y, c=c, s=s, alpha=0.5)
    
plt.savefig(rawRdCtPlot)


# In[16]:

# Plots of raw read counts(<=500)
lg.info('Drawing plots of records with less than 500 reads toally...')
dfLess500Rd = df[(df.altCountsBios + df.refCountsBios) <= 500]
fig, ax = plt.subplots()
fig.set_size_inches((40, 10))

for i, (g, c) in enumerate(pool):
    x = dfLess500Rd.loc[dfLess500Rd.group==g, 'altCountsBios']
    y = dfLess500Rd.loc[dfLess500Rd.group==g, 'refCountsBios']
    s = dfLess500Rd.loc[dfLess500Rd.group==g, 'cadd']
    ax.scatter(x, y, c=c, s=s, alpha=0.5)
plt.savefig(rawLsRdCtPlot)


# In[17]:

# PCA analysis
lg.info('Doing PCA...')
cpPool = ['cadd', 'FDROverall', 'log2FC']
cpNum = len(cpPool)

pca = PCA(cpNum, whiten=True)

X = df.loc[:, cpPool]
XFit = pca.fit(X)

Xreduced = pca.transform(X)
covar = pca.get_covariance()

lg.info('Plotting PCA results...')
# Plots for PCA analysis
fig, ax = plt.subplots()
for g, c in pool:
    ax.scatter(
        Xreduced[df.group==g, 0], Xreduced[df.group==g, 1], 
        c=c, marker='.', alpha=0.5)

plt.savefig(PCAPlot)

