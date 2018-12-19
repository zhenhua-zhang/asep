#!/usr/bin/env
# -*- coding: utf-8 -*-
#
# File Name  : pythonScriptTemplement
# Author     : zhzhang
# E-mail     : zhzhang2015@sina.com
# Created on : Sun 11 Nov 2018 06:10:46 PM CET
# Version    : v0.0.1
# License    : MIT
#

# Load basic modules
import time
import logging
import numpy as np
import pandas as pd

from os.path import join

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def transform():
    HOME_DIR = '/home/umcg-zzhang/'
    PROJECT_DIR = join(HOME_DIR, 'Documents/projects/ASEpredictor')

    INPUT_DIR = join(PROJECT_DIR, 'outputs', 'biosGavinOverlapCov10')
    INPUT_FILE = join(INPUT_DIR, 'biosGavinOlCv10AntUfltCst.tsv')

    OUTPUT_DIR = join(PROJECT_DIR, 'outputs', 'biosGavinOverlapCov10')
    OUTPUT_FILE = join(OUTPUT_DIR, 'biosGavinOlCv10AntUfltCstLog2FCVarZsc.tsv')

    GROUP_COLS = ['chr', 'pos', 'ref', 'alt']
    raw_df = pd.read_table(INPUT_FILE, header=0, low_memory=False)
    raw_df = raw_df.sort_values(GROUP_COLS)

    kick_off_index = -13
    log2FCVar_df_column = raw_df.columns[:kick_off_index]

    log2FCVar_df = pd.DataFrame(columns=log2FCVar_df_column)
    raw_df_group = raw_df.groupby(GROUP_COLS)
    for index, (gn, g) in enumerate(raw_df_group):
        g_l, _ = g.shape
        var = g.log2FC.var()
        centralized = g.log2FC - g.log2FC.mean()
        z_score = centralized / g.log2FC.var(ddof=0)
        if np.isnan(var):
            var = 0
            z_score = np.nan

        g_features = g.iloc[0, :kick_off_index]
        g_features['log2FCVar'] = var
        g_features['log2FCZScore'] = z_score
        log2FCVar_df = log2FCVar_df.append(g_features)

    log2FCVar_df.to_csv(OUTPUT_FILE, header=True, index=False, sep='\t')

    fig, (ax0, ax1) = plt.subplots(ncols=2)
    ax0.hist(log2FCVar_df.log2FCVar, bins=100)
    ax1.hist(log2FCVar_df[log2FCVar_df.log2FCVar != 0].log2FCVar, bins=100)
    fig.set_edgecolor('white')
    fig.set_figwidth(10)
    fig.savefig('log2FCVar.png')


def main():
    # Start a timer.
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
    lg.setLevel(logging.INFO)              # default logging level INFO
    lg.addHandler(ch)
    lg.info("=== Start ... ===")

    transform()

    lg.info("=== Done  ... ===\nTime elapsed: %0.5f" % (time.clock()-ct))


if __name__ == '__main__':
    main()
    # Finished the logging & time counting ends
