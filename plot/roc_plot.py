#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Draw ROC plot from asepred.py

'''
import pickle
import argparse

import numpy as nmp
import pandas as pds
import seaborn as sns
import matplotlib.pyplot as plt

def getargs():
    '''Parse CLI arguments'''
    parser = argparse.ArgumentParser(description="Draw ROC plot.")
    parser.add_argument('-i', '--input-file', dest='input_file', required=True, help='Input file.')
    parser.add_argument('-t', '--plot-title', dest='plot_title', default='ROC', help='The title. Default: %(default)s')
    parser.add_argument('-F', '--output-fmt', nargs='*', choices=['png', 'svg', 'pdf'], default=['svg'], dest='output_fmt', help='The output fmt. Default: %(default)s')
    parser.add_argument('-o', '--output-pref', dest='output_pref', default='./roc_curve', help='Prefix of output file.  Default: %(default)s')

    return parser


def load_pkl(pkl_path):
    '''Load Pickle object of ROC and AUC.'''
    with open(pkl_path, 'rb') as pklh:
        roc_pl = pickle.load(pklh)

    return roc_pl


def draw_roc(auc_roc_pl, output_path, title="ROC", figsize=(10, 10), prec=3,
             null_line=True, roc_mean=True):
    '''Draw a line plot to show the ROC.
    '''
    figwidth, figheight = figsize

    auc_pl, roc_pl = [], []
    for idx, (auc, roc) in enumerate(auc_roc_pl):
        auc_pl.append(auc)
        _temp_df = pds.DataFrame(dict(zip(["fpr", "tpr", "thr"], roc)))
        _temp_df["cv_idx"] = idx
        roc_pl.append(_temp_df)

    auc_mean, auc_std = nmp.mean(auc_pl), nmp.std(auc_pl)
    label = "AUC-ROC: {:.{prec}} [{:.{prec}}, {:.{prec}}]".format(
        auc_mean, auc_mean - auc_std, auc_mean + auc_std, prec=prec)

    roc_df = pds.concat(roc_pl)

    line_ax = sns.lineplot(x="fpr", y="tpr", hue="cv_idx", data=roc_df,
                           palette=sns.light_palette("navy", len(auc_pl)),
                           legend=False, linewidth=0.1)
    line_ax.text(0., 1., label, fontsize="xx-large")

    if null_line:
        line_ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1)

    if roc_mean:
        roc_pv_df = (roc_df
                     .pivot(columns='cv_idx', values=['fpr', 'tpr'])
                     .interpolate()
                     .mean(axis=1, level=0))
        sns.lineplot(x=roc_pv_df['fpr'], y=roc_pv_df['tpr'], color='red', ax=line_ax)

    line_ax.set_title(title, fontsize='xx-large')
    line_ax.set_xlabel("False positive rate", fontsize='xx-large')
    line_ax.set_ylabel("True positive rate", fontsize='xx-large')

    fig = line_ax.get_figure()
    fig.set_figwidth(figwidth)
    fig.set_figheight(figheight)

    if isinstance(output_path, str):
        output_path = [output_path]

    for optpath in output_path:
        fig.savefig(optpath)

    plt.close(fig)


def main():
    '''The main entry.
    '''
    args = getargs().parse_args()
    input_file = args.input_file
    plot_title = args.plot_title
    output_fmt = args.output_fmt
    output_pref = args.output_pref

    roc_pl = load_pkl(input_file)
    output_path = ['{}-roc_curve.{}'.format(output_pref, fmt) for fmt in output_fmt]
    draw_roc(roc_pl, output_path, title=plot_title)

if __name__ == '__main__':
    main()
