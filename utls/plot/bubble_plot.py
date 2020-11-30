#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import numpy as np
import pandas as pds
import seaborn as sns
import matplotlib as mpl


def getargs():
    '''Get CLI arguments.
    '''
    parser = argparse.ArgumentParser(description='Plot bubble plot for functional enrichment results.')

    parser.add_argument('-i', '--input-file', dest='input_file', required=True, help='The input file.')
    parser.add_argument('-d', '--input-sep', dest='input_sep', default='\t', help='The filed splitter of input file. Default: %(default)s')
    parser.add_argument('-p', '--plot-cols', dest='plot_cols', default=['Adjusted P-value', 'Odds Ratio', 'Combined Score', 'Term'], metavar=('P-VALUE', 'ODDS-RATIO', 'SCORE', 'TERM'), nargs=4, help='Columns used to plot bubble plot. Default: %(default)s')
    parser.add_argument('-t', '--p-val-threshold', dest='p_val_thr', default=0.05, type=float, help='The threshold of p-value to plot. Default: %(default)s')
    parser.add_argument('-n', '--first-n-rows', dest='first_n_rows', default=20, type=int, help='The first N rows will be ploted. Default: %(default)s')
    parser.add_argument('-f', '--output-fmt', dest='output_fmt', default=['svg'], choices=['png', 'svg', 'pdf'], nargs='*', help='Output file format. Default: %(default)s')
    parser.add_argument('-o', '--output-pref', dest='output_pref', default='./bubble_plot', help='Output prefix. Default: %(default)s')

    return parser


def load_file(fpath, **kwargs):
    '''Load input files.
    '''
    return pds.read_csv(fpath, **kwargs)


def bubble_plot(dtfm: pds.DataFrame, output_path, figsize=(8, 8),
                plot_cols=None, title='Enrichment bubble plot',
                sort_by=None, max_p_val=0.05, log10_p_val=True,
                term_col_func=None):
    '''Draw bubble plots.
    '''

    p_val_col, odds_ratio_col, score_col, term_col = plot_cols

    dtfm = dtfm.query('`{}` < {}'.format(p_val_col, max_p_val))

    if term_col_func is not None:
        dtfm.loc[:, term_col] = dtfm.loc[:, term_col].apply(term_col_func)

    if log10_p_val:
        dtfm.loc[:, p_val_col] = dtfm.loc[:, p_val_col].apply(lambda x: -np.log10(x))

    if sort_by is None:
        dtfm = dtfm.sort_values(by=odds_ratio_col)
    else:
        dtfm = dtfm.sort_values(by=sort_by)

    sizes = dtfm.loc[:, score_col].min(), dtfm.loc[:, score_col].max()

    mpl.rcParams['legend.fontsize'] = 'large'
    bubble_ax = sns.scatterplot(x=odds_ratio_col, y=term_col, hue=p_val_col,
                                size=score_col, sizes=sizes, legend='brief',
                                data=dtfm)
    bubble_ax.set_title(title, fontsize='large')
    bubble_ax.set_xlabel('Odds ratio')
    bubble_ax.set_ylabel('Term')
    bubble_ax.grid(which='major', axis='x', linestyle='--')

    figwidth, figheight = figsize
    bubble_fig = bubble_ax.get_figure()
    bubble_fig.set_figheight(figheight)
    bubble_fig.set_figwidth(figwidth)
    bubble_fig.set_tight_layout(True)

    if isinstance(output_path, str):
        output_path = [output_path]

    for optpath in output_path:
        bubble_fig.savefig(optpath)


def main():
    args = getargs().parse_args()

    input_file = args.input_file
    input_sep = args.input_sep
    plot_cols = args.plot_cols
    p_val_thr = args.p_val_thr
    first_n_rows = args.first_n_rows
    output_fmt = args.output_fmt
    output_pref = args.output_pref


    dtfm = load_file(input_file, sep=input_sep).head(first_n_rows)
    output_path = ['{}-bubble_plot.{}'.format(output_pref, fmt) for fmt in output_fmt]
    bubble_plot(dtfm, output_path=output_path, plot_cols=plot_cols, term_col_func=lambda x: x.split("(")[0])


if __name__ == '__main__':
    main()
