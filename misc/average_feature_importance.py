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

import os
import pickle
import argparse

import pandas as pd
import matplotlib.pyplot as plt

def getargs():
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--input-files", dest="input_files", nargs="*",
                   help="Input files. Accept more than one files.")
    p.add_argument("-n", "--first-n", dest="first_n", default=20, type=int,
                   help="The first n most important features sorted by mean. Default: %(default)s")
    p.add_argument("-o", "--output-prefix", dest="output_prefix", default="feature_importance",
                   help="Output file. Path is acceptable, but require the path existancs")
    return p


def load_input_file(input_file_pool, output_prefix):
    fidict = None
    for input_file in input_file_pool:
        with open(input_file, "rb") as infh:
            _fidict = pickle.load(infh)
            _fidict = {key: [sum(val)/len(val)] for key, val in _fidict.items()}
            if fidict is None:
                fidict = _fidict
            else:
                fidict = {key: fidict[key] + _fidict[key] for key in _fidict}

    fidtfm = pd.DataFrame(fidict)
    fidtfm.loc["mean", :] = fidtfm.mean(axis=0)
    fidtfm.loc["std", :] = fidtfm.std(axis=0)

    fidtfm.to_csv(output_prefix + ".tsv", sep="\t")

    return fidtfm


def draw_hist(fidtfm, output_prefix, first_n):
    length, width = fidtfm.shape
    name = fidtfm.columns
    mean = fidtfm.loc["mean", :]
    std = fidtfm.loc["std", :]

    feature_importance = sorted(list(zip(name, mean, std)), key=lambda x: -x[1])

    name = [x[0] for x in feature_importance[:first_n]]
    yerr = [x[2] for x in feature_importance[:first_n]]
    height = [x[1] for x in feature_importance[:first_n]]

    fig, ax = plt.subplots()
    ax.bar(x=range(first_n), height=height, yerr=yerr)
    ax.set_xticks(range(first_n))
    ax.set_xticklabels(name, fontdict={"rotation": 45, "rotation_mode": "anchor", "ha": "right"})
    ax.set_title("First {} most important features from {} models".format(first_n, length-2))

    fig.tight_layout()
    fig.savefig(output_prefix + ".png")

    plt.close()


def main():
    parser = getargs()
    args = parser.parse_args()
    input_files = args.input_files
    first_n = args.first_n
    output_prefix = args.output_prefix

    output_path, file_prefix = os.path.split(output_prefix)
    output_path = "./" if output_path == "" else output_path
    if not os.path.exists(output_path):
        raise FileExistsError("The path does not exist!")

    fidtfm = load_input_file(input_files, output_prefix)
    draw_hist(fidtfm, output_prefix, first_n)


if __name__ == "__main__":
    main()
