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


BENIGN_FIG, BENIGN_AX = plt.subplots()
POPULATION_FIG, POPULATION_AX = plt.subplots()
PATHOGENIC_FIG, PATHOGENIC_AX = plt.subplots()

for index, (gn, g) in enumerate(RAW_BENIGN_DF_GROUP):
    gl, _ = g.shape
    indexs = [index for x in range(gl)]
    BENIGN_AX.scatter(g.log2FC, indexs, c='r', s=0.5, marker='.')

BENIGN_AX.set_title('Distribution of BENIGN')
BENIGN_FIG.set_figwidth(25)
BENIGN_FIG.set_figheight(90)
BENIGN_FIG.set_dpi(150)
BENIGN_FIG.savefig('log2FC_distribution_BENIGN.pdf')
BENIGN_FIG.savefig('log2FC_distribution_BENIGN.png')

for index, (gn, g) in enumerate(RAW_POPULATION_DF_GROUP):
    gl, _ = g.shape
    indexs = [index for x in range(gl)]
    POPULATION_AX.scatter(g.log2FC, indexs, s=0.2, c='g', marker='x')
    POPULATION_AX.axhline(index, c='g', alpha=0.2, markersize=0.1)
    POPULATION_AX.axvline(1, alpha=0.1, linestyle='dotted')
    POPULATION_AX.axvline(-1, alpha=0.1, linestyle='dotted')

POPULATION_AX.set_title('Distribution of POPULATION')
POPULATION_FIG.set_figwidth(25)
POPULATION_FIG.set_figheight(60)
POPULATION_FIG.set_dpi(150)
POPULATION_FIG.savefig('log2FC_distribution_POPULATION.pdf')
POPULATION_FIG.savefig('log2FC_distribution_POPULATION.png')

for index, (gn, g) in enumerate(RAW_PATHOGENIC_DF_GROUP):
    gl, _ = g.shape
    indexs = [index for x in range(gl)]
    PATHOGENIC_AX.scatter(g.log2FC, indexs, s=0.2, c='b', marker='o')
    PATHOGENIC_AX.axhline(index, c='b', alpha=0.2, markersize=0.1)
    PATHOGENIC_AX.axvline(1, alpha=0.1, linestyle='dotted')
    PATHOGENIC_AX.axvline(-1, alpha=0.1, linestyle='dotted')

PATHOGENIC_AX.set_title('Distribution of PATHOGENIC')
PATHOGENIC_FIG.set_figwidth(25)
PATHOGENIC_FIG.set_figheight(40)
PATHOGENIC_FIG.set_dpi(150)
PATHOGENIC_FIG.savefig('log2FC_distribution_PATHOGENIC.pdf')
PATHOGENIC_FIG.savefig('log2FC_distribution_PATHOGENIC.png')
