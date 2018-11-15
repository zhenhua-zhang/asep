#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#   File Name  : check_log2fc_distribution.py
##  Author     : zhzhang
### E-mail     : zhzhang2015@sina.com
### Created on : Thu 15 Nov 2018 02:06:53 PM CET
##  Version    : v0.0.1
#   License    : MIT
"""

from os.path import join
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# input files
PJDIR = '/home/umcg-zzhang/Documents/projects/ASEpredictor'
IPDIR = join(PJDIR, 'outputs/biosGavinOverlapCov10')

DEBUG = False
if DEBUG:
    IPFILE = join(IPDIR, 'tmp.tsv')
else:
    IPFILE = join(IPDIR, 'biosGavinOlCv10AntUfltCst.tsv')

RAW_DF = pd.read_table(IPFILE, header=0, low_memory=False)
RAW_BENIGN_DF = RAW_DF[RAW_DF.group == 'BENIGN']
RAW_PATHOGENIC_DF = RAW_DF[RAW_DF.group == 'PATHOGENIC']
RAW_POPULATION_DF = RAW_DF[RAW_DF.group == 'POPULATION']

GROUPBY_COLS = ['chr', 'pos', 'ref', 'alt']
RAW_DF_GROUP = RAW_DF.groupby(GROUPBY_COLS)

RAW_BENIGN_DF_GROUP = RAW_BENIGN_DF.groupby(GROUPBY_COLS)
RAW_PATHOGENIC_DF_GROUP = RAW_PATHOGENIC_DF.groupby(GROUPBY_COLS)
RAW_POPULATION_DF_GROUP = RAW_POPULATION_DF.groupby(GROUPBY_COLS)


# FILTER = ((abs(RAW_DF.log2FC) >= 1) & (RAW_DF.FDRPerVariant <= 0.05))
# FIL_DF = RAW_DF[FILTER]
# FIL_BENIGN_DF = FIL_DF[FIL_DF.group == 'BENIGN']
# FIL_PATHOGENIC_DF = FIL_DF[FIL_DF.group == 'PATHOGENIC']
# FIL_POPULATION_DF = FIL_DF[FIL_DF.group == 'POPULATION']

# FIL_DF_GROUP = FIL_DF.groupby(GROUPBY_COLS)
# FIL_BENIGN_DF_GROUP = FIL_BENIGN_DF.groupby(GROUPBY_COLS)
# FIL_PATHOGENIC_DF_GROUP = FIL_PATHOGENIC_DF.groupby(GROUPBY_COLS)
# FIL_POPULATION_DF_GROUP = FIL_POPULATION_DF.groupby(GROUPBY_COLS)


FIG, (BENIGN_AX, POPULATION_AX, PATHOGENIC_AX) = plt.subplots(
    nrows=3, sharex=True)

for index, (gn, g) in enumerate(RAW_BENIGN_DF_GROUP):
    gl, _ = g.shape
    indexs = [index for x in range(gl)]
    BENIGN_AX.scatter(g.log2FC, indexs, c='r', s=0.5, marker='.')

for index, (gn, g) in enumerate(RAW_POPULATION_DF_GROUP):
    gl, _ = g.shape
    indexs = [index for x in range(gl)]
    POPULATION_AX.scatter(g.log2FC, indexs, s=0.2, c='g', marker='x')
    POPULATION_AX.axhline(index, c='g', alpha=0.2)
    POPULATION_AX.axvline(1, alpha=0.1, linestyle='dotted')
    POPULATION_AX.axvline(-1, alpha=0.1, linestyle='dotted')

for index, (gn, g) in enumerate(RAW_PATHOGENIC_DF_GROUP):
    gl, _ = g.shape
    indexs = [index for x in range(gl)]
    PATHOGENIC_AX.scatter(g.log2FC, indexs, s=0.2, c='b', marker='o')
    PATHOGENIC_AX.axhline(index, c='b', alpha=0.2)
    PATHOGENIC_AX.axvline(1, alpha=0.1, linestyle='dotted')
    PATHOGENIC_AX.axvline(-1, alpha=0.1, linestyle='dotted')

BENIGN_AX.set_title('Distribution of BENIGN')
POPULATION_AX.set_title('Distribution of POPULATION')
PATHOGENIC_AX.set_title('Distribution of PATHOGENIC')

FIG.set_figwidth(25)
FIG.set_figheight(30)
FIG.set_dpi(150)
FIG.savefig('log2FC_distribution.png')
