#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File name : average_feature_importance.py
# Author    : Zhenhua Zhang
# E-mail    : zhenhua.zhang217@gmail.com
# Created   : Fri 26 Jun 2020 09:36:51 AM CEST
# Version   : v0.1.0
# License   : MIT
#

"""
Dependency:
    Python: 3.7.0
    pandas: 1.0.3
    seaborn: 0.10.1
    matplotlib: 3.2.1
"""


import os
import pickle
import argparse

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

def getargs():
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--input-files", dest="input_files", nargs="*",
                   help="Input files. Accept more than one files.")
    p.add_argument("-F", "--input-sample-name", dest="input_sample_name", nargs="*",
                   help="Sample name for the input files. Accept more than one files, but the number should be the same to the one of --input-files")
    p.add_argument("-b", "--base-src", dest="base_src", default="bios",
                   help="The baseline source. Default: %(default)s")
    p.add_argument("-n", "--first-n", dest="first_n", default=30, type=int,
                   help="The first n most important features sorted by mean. Default: %(default)s")
    p.add_argument("-o", "--output-prefix", dest="output_prefix", default="feature_importance",
                   help="Output file. Path is acceptable, but require the path existancs")
    return p


def load_input_file(file_pool, sample_name_pool, base_src="bios", first_n=30):
    # base_src 目前暂定为bios，后续需要根据你命令行参数确定。
    fidtfm = None
    for input_file, sample_name in zip(file_pool, sample_name_pool):
        with open(input_file, "rb") as infh:
            fidict = pickle.load(infh)

            idx_len = len(fidict[list(fidict.keys())[0]])
            idx_tuple = [(sample_name, "cv_{:0>2}".format(i+1)) for i in range(idx_len)]
            idx = pd.MultiIndex.from_tuples(idx_tuple)

            if fidtfm is None:
                fidtfm = pd.DataFrame(fidict, index=idx)
            else:
                fidtfm = fidtfm.append(pd.DataFrame(fidict, index=idx))

    first_n_feature = fidtfm.mean(level=0, axis=0) \
            .loc[base_src] \
            .sort_values(ascending=False) \
            .index[:first_n]
    
    fidtfm = fidtfm.loc[:, first_n_feature]

    return fidtfm


def prepare_for_pair_plot(fidtfm):
    fidtfm = fidtfm.stack().reset_index(level=(0, 1, 2))
    fidtfm.columns = ["source", "cv", "feature", "value"]

    return fidtfm


def draw_pairs(fidtfm, output_prefix, plot_type="box", fmt="png"):
    """Draw paired barplot."""

    if plot_type == "bar":
        ax = sb.barplot(x="feature", y="value", hue="source", data=fidtfm)
    elif plot_type == "box":
        ax = sb.boxplot(x="feature", y="value", hue="source", data=fidtfm)
    else:
        raise TypeError("Unsupported type of plot {}".format(plot_type))

    labels = fidtfm.feature.drop_duplicates()
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")

    fig = ax.get_figure()
    fig.tight_layout()
    n_fi = len(labels)
    fig.set_figwidth(n_fi*0.25)
    fig.set_figheight(5)

    fig.savefig("{}.paired_{}plot.{}".format(output_prefix, plot_type, fmt))

    plt.close()


def prepare_for_slope_plot(fidtfm, sample_name_pool):
    _, n_fi = fidtfm.shape
    src1, src2 = sample_name_pool
    src1_rank, src2_rank = src1 + "_rank", src2 + "_rank" 

    fidtfm_mean = fidtfm.mean(level=0, axis=0)
    fidtfm_mean = fidtfm_mean.sort_values(by=src1, axis=1, ascending=False)
    fidtfm_mean.loc[src1_rank] = range(n_fi)

    fidtfm_mean = fidtfm_mean.sort_values(by=src2, axis=1, ascending=False)
    fidtfm_mean.loc[src2_rank] = range(n_fi)

    return fidtfm_mean.sort_values(by=src1 + "_rank", axis=1).sort_index()


def draw_shift(fidtfm, output_prefix, sample_name_pool, fmt="png"):
    _, n_fi = fidtfm.shape
    src1, src2 = sample_name_pool
    src1_rank, src2_rank = src1 + "_rank", src2 + "_rank" 

    ax = host_subplot(111)
    ax.scatter(fidtfm.loc[src1_rank, :], [1]*n_fi)
    ax.scatter(fidtfm.loc[src2_rank, :], [0]*n_fi)

    for x1, x2 in zip(fidtfm.loc[src1_rank], fidtfm.loc[src2_rank]):
        ax.plot([x1, x2], [1, 0])

    ax_ = ax.twin()
    ax_.set_xticks(range(n_fi))
    ax_.set_xticklabels(fidtfm.columns, rotation=-45, ha="right", rotation_mode="anchor")
    ax_.set_xlabel(src1)

    labels = fidtfm.loc[src2_rank].sort_values().index
    ax.set_xticks(range(n_fi))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel(src2)

    ax.axis["right"].set_visible(False)
    ax.axis["left"].set_visible(False)
    ax_.axis["right"].set_visible(False)
    ax_.axis["left"].set_visible(False)
    ax_.axis["bottom"].set_visible(False)

    fig = ax.get_figure()
    fig.set_figwidth(n_fi*0.25)
    fig.set_figheight(5)
    plt.tight_layout()
    fig.savefig("{}.feature_shift.{}".format(output_prefix, fmt))
    plt.close()


def main():
    parser = getargs()
    args = parser.parse_args()
    input_files = args.input_files
    input_sample_name = args.input_sample_name
    base_src = args.base_src
    first_n = args.first_n
    output_prefix = args.output_prefix

    if len(input_files) != len(input_sample_name):
        raise ValueError("The number of --input-files should be the same as --input-sample-name")

    if base_src not in input_sample_name:
        raise ValueError("--base-src should be one of --input-sample-name")

    output_path, file_prefix = os.path.split(output_prefix)
    output_path = "./" if output_path == "" else output_path
    if not os.path.exists(output_path):
        raise FileExistsError("The path does not exist!")

    fidtfm = load_input_file(input_files, input_sample_name, base_src, first_n=first_n)

    pair_fidtfm = prepare_for_pair_plot(fidtfm)
    draw_pairs(pair_fidtfm, output_prefix)

    slope_fidtfm = prepare_for_slope_plot(fidtfm, input_sample_name)
    draw_shift(slope_fidtfm, output_prefix, input_sample_name)


if __name__ == "__main__":
    main()
