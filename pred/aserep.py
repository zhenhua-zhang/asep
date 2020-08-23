#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Draw Manhattan plot.

Usages:

Options:

Notes:

Todo:
'''

import argparse

import pandas as pd
import matplotlib.pyplot as plt

from numpy.lib import scimath


CHROM_LEN_GRCH37 = {
    'chr1' : 249250621, 'chr2' : 243199373, 'chr3' : 198022430,
    'chr4' : 191154276, 'chr5' : 180915260, 'chr6' : 171115067,
    'chr7' : 159138663, 'chr8' : 146364022, "chr9" : 141213431,
    "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
    "chr13": 115169878, "chr14": 107349540, "chr15": 102531392,
    "chr16":  90354753, "chr17":  81195210, "chr18":  78077248,
    "chr19":  59128983, "chr20":  63025520, "chr21":  48129895,
    "chr22":  51304566, "chrx" : 155270560, "chry" :  59373566,
}

ASE_QUANT_COLS = [
    'log2FC', 'bn_p', 'bn_p_adj', 'bb_p', 'bb_p_adj', 'group_size', 'bn_ASE',
    'bb_ASE', 'prob0_mean', 'prob0_var', 'prob1_mean', 'prob1_var'
]


def getargs():
    parser = argparse.ArgumentParser(description='Draw Manhattan plot.')
    parser.add_argument('-f', '--pred-files', dest='pred_files', metavar=('FILE_1', 'FILE_2'), nargs=2, required=True, help='Files including predicted ASE.')
    parser.add_argument('-d', '--pred-files-delimter', dest='pred_files_delimiter', metavar='DELIMITER', default='\t', help='The file delimiter in pred-files. Default: %(default)s')
    parser.add_argument('-s', '--dataset-suffix', dest='dataset_suffix', metavar=('SUFFIX_1', 'SUFFIX_2'), nargs=2, default=('bios', 'gtex'), help='Suffix used for two input dataset respectively.  Default: %(default)s')
    parser.add_argument('-c', '--coord-cols', dest='coord_cols', metavar=('CHROM', 'POS', 'REF', "ALT"), nargs=4, default=('Chrom', 'Pos', 'Ref', 'Alt'), help='The  columns used to represent coordinations, including chromsome, postion, reference allele, and alternative allele. Default: %(default)s')

    parser.add_argument('-e', '--extra-cols-to-save', dest='extra_cols', default=('GeneID', 'GeneName', 'Consequence'), nargs='*', help='Other columns will be saved together with --coord-cols and ASE quantification columns. Default: %(default)s')
    parser.add_argument('-o', '--output-fref', dest='output_pref', default='./bios.gtex.ase_quant_pred', help='Output prefix. Default: %(default)s')
    parser.add_argument('-F', '--plot-fmt', dest='plot_fmt', default='png', help='The output format of plot. Default: %(default)s')

    return parser


def load_dtfm(path):
    return pd.read_csv(path, sep='\t', header=0)


def partial_join(left, right, both_on, outer_asis, how='outer',
                 suffixes=('_l', '_r')):
    """Partially join two DataFrames.
    """
    if isinstance(both_on, str):
        both_on = [both_on]

    if set(both_on) & set(outer_asis):
        raise ValueError("The elements in both_on should not be in outer_asis")

    asis_cols = list(both_on) + list(outer_asis)
    left_asis, right_asis = left.loc[:, asis_cols], right.loc[:, asis_cols]
    asis_dtfm = pd.concat([left_asis, right_asis], ignore_index=True) \
            .drop_duplicates()

    comb_cols = [x for x in left.columns if x not in outer_asis]
    left_comb, right_comb = left.loc[:, comb_cols], right.loc[:, comb_cols]
    comb_dtfm = pd.merge(left_comb, right_comb, 'outer', both_on, suffixes=suffixes)

    joint_dtfm = pd.merge(asis_dtfm, comb_dtfm, how, both_on)

    return joint_dtfm


def make_shift_dict(chosen_chr, chr_fmt_func=None):
    '''Make shift dict.
    '''
    if chr_fmt_func is None:
        chr_fmt_func = lambda y: ('chr{}'.format(x) for x in y)

    chr_pair = sorted(zip(chosen_chr, chr_fmt_func(chosen_chr)), key=lambda x: int(x[0]))

    pos = [CHROM_LEN_GRCH37[_chr] for _, _chr in chr_pair]
    shift_pos = [sum(pos[:i]) for i in range(len(pos))]

    return dict(zip([x[0] for x in chr_pair], shift_pos))


def trans_func(row, shift_dict):
    '''A transform function applied on per row.
    '''
    bb_p_adj_gtex, bb_p_adj_bios, prob1_mean_gtex, prob1_mean_bios \
            = row[['bb_p_adj_gtex', 'bb_p_adj_bios', 'prob1_mean_gtex',
                   'prob1_mean_bios']]

    if bb_p_adj_bios < 0.05 and bb_p_adj_gtex < 0.05:
        color = 'red'
    elif bb_p_adj_bios < 0.05:
        color = 'green'
    elif bb_p_adj_gtex < 0.05:
        color = 'blue'
    else:
        color = '0.5'

    # if bb_p_adj_gtex < 0.05 and prob1_mean_gtex >= 0.5:
    #     shape_gtex = 'D'
    # elif bb_p_adj_gtex < 0.05:
    #     shape_gtex = 'x'
    # elif prob1_mean_gtex >= 0.5:
    #     shape_gtex = 'v'
    # else:
    #     shape_gtex = '.'

    # if bb_p_adj_bios < 0.05 and prob1_mean_bios >= 0.5:
    #     shape_bios = 'D'
    # elif bb_p_adj_bios < 0.05:
    #     shape_bios = 'x'
    # elif prob1_mean_bios >= 0.5:
    #     shape_bios = 'v'
    # else:
    #     shape_bios = '.'

    pos = row['Pos'] * 0.85 + shift_dict[row['Chrom']]

    return [color, pos] #, shape_bios, shape_gtex]


def manhattan(pred_dtfm, x_col, y_col, output_path, chrom_col='Chrom', height=9,
              width=32, trans=None):
    '''Draw upper-lower Manhattan plot.
    '''
    mark_color_dict = {
        'D': 'Q,P', 'x': 'Q', 'v': 'P', '.': 'N', 'red': 'Both',
        'green': 'BIOS', 'blue': 'GTEx', '0.5': 'None'}

    shift_dict = make_shift_dict(range(1, 23))
    plot_cols = ['color', 'pos_shift'] #, 'shape_bios', 'shape_gtex']
    pred_dtfm[plot_cols] = pred_dtfm \
            .apply(trans, axis=1, result_type='expand', shift_dict=shift_dict)

    chrom_ticks = []
    for chrom in sorted(pred_dtfm.loc[:, chrom_col].drop_duplicates()):
        chrom_recs = pred_dtfm.loc[pred_dtfm[chrom_col] == chrom, 'pos_shift']
        chrom_ticks.append((max(chrom_recs) + min(chrom_recs)) / 2)

    y_col_upper, y_col_lower = y_col
    fig, (axes_upper, axes_lower) = plt.subplots(nrows=2)

    #color_shape_pairs = pred_dtfm.loc[:, ['shape_bios', 'color']].drop_duplicates()
    #color_shape_pairs = zip(color_shape_pairs['shape_bios'], color_shape_pairs['color'])
    snv_colors = pred_dtfm.loc[:, "color"].drop_duplicates()
    for color in snv_colors:
        subgroup_loc = pred_dtfm.loc[:, 'color'] == color
        subgroup = pred_dtfm.loc[subgroup_loc, :]
        label = '{}'.format(mark_color_dict[color])
        x_val = subgroup.loc[:, 'pos_shift']
        y_val = subgroup.loc[:, y_col_upper].apply(lambda x: -scimath.log10(x))
        axes_upper.scatter(x_val, y_val, c=color, label=label)

    axes_upper.axhline(-scimath.log10(5e-2), ls='--')
    axes_upper.spines['top'].set_visible(False)
    axes_upper.spines['right'].set_visible(False)
    axes_upper.spines['bottom'].set_visible(False)

    axes_upper.tick_params(axis='x', which='both', length=0)
    axes_upper.set_xticks(chrom_ticks)
    xticklables = ['chr{}'.format(i+1) for i, _ in enumerate(chrom_ticks)]
    axes_upper.set_xticklabels(xticklables, rotation=90)
    axes_upper.set_title('BIOS')

    handles, labels = axes_upper.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0].split(';', 1)))
    axes_upper.legend(handles, labels)

    color_shape_pairs = pred_dtfm.loc[:, ['shape_gtex', 'color']].drop_duplicates()
    color_shape_pairs = zip(color_shape_pairs['shape_gtex'], color_shape_pairs['color'])
    for marker, color in color_shape_pairs:
        subgroup_loc = ((pred_dtfm.loc[:, 'shape_gtex'] == marker)
                        & (pred_dtfm.loc[:, 'color'] == color))
        subgroup = pred_dtfm.loc[subgroup_loc, :]
        label = '{};{}'.format(mark_color_dict[color], mark_color_dict[marker])
        x_val = subgroup.loc[:, 'pos_shift']
        y_val = subgroup.loc[:, y_col_lower].apply(lambda x: -scimath.log10(x))
        axes_lower.scatter(x_val, y_val, c=color, marker=marker, label=label)

    axes_lower.axhline(-scimath.log10(5e-2), ls='--')
    axes_lower.invert_yaxis()
    axes_lower.spines['top'].set_visible(False)
    axes_lower.spines['right'].set_visible(False)
    axes_lower.spines['bottom'].set_visible(False)
    axes_lower.set_xticks([])
    axes_lower.legend(loc='lower left')
    axes_lower.set_title('GTEx')

    handles, labels = axes_lower.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0].split(';', 1)))
    axes_lower.legend(handles, labels)

    fig.set_figheight(height)
    fig.set_figwidth(width)
    fig.tight_layout(h_pad=0)
    fig.savefig(output_path)
    plt.cla()
    plt.clf()
    plt.close()


def scatter_plot(pred_dtfm, x_col, y_col, output_path,
                 trans_func=lambda x: -scimath.log10(x)):
    '''Scatter plot.
    '''
    fig, axes = plt.subplots()

    cond = (pred_dtfm[x_col] < 0.05) & (pred_dtfm[y_col] < 0.05)
    xy_vals = pred_dtfm.loc[cond, [x_col, y_col]].dropna()
    x_val = xy_vals.loc[:, x_col].apply(trans_func)
    y_val = xy_vals.loc[:, y_col].apply(trans_func)
    axes.scatter(x_val, y_val, c='r', s=1)

    cond = (pred_dtfm[x_col] > 0.05) | (pred_dtfm[y_col] > 0.05)
    xy_vals = pred_dtfm.loc[cond, [x_col, y_col]].dropna()
    x_val = xy_vals.loc[:, x_col].apply(trans_func)
    y_val = xy_vals.loc[:, y_col].apply(trans_func)
    axes.scatter(x_val, y_val, c='0.5', s=1, alpha=0.5)


    fig.savefig(output_path)
    plt.cla()
    plt.clf()
    plt.close()


def main():
    '''Main entry.
    '''
    parser = getargs().parse_args()
    pred_files = parser.pred_files
    coord_cols = parser.coord_cols
    extra_cols = parser.extra_cols
    output_pref = parser.output_pref
    plot_fmt = parser.plot_fmt
    dataset_suffix = parser.dataset_suffix

    lfile, rfile = pred_files
    ldtfm = load_dtfm(lfile)
    rdtfm = load_dtfm(rfile)

    outer_asis = list(set(ldtfm.columns)-set(coord_cols)-set(ASE_QUANT_COLS))
    pred_dtfm = partial_join(ldtfm, rdtfm, both_on=coord_cols,
                             outer_asis=outer_asis, suffixes=dataset_suffix)

    output_path = output_pref + '.csv'
    kept_cols = (list(coord_cols)
                 + list(extra_cols)
                 + [x + dataset_suffix[0] for x in ASE_QUANT_COLS]
                 + [x + dataset_suffix[1] for x in ASE_QUANT_COLS])
    pred_dtfm.loc[:, kept_cols].to_csv(output_path, index=False)

    output_path = '.'.join([output_pref, 'manhattan_plot', plot_fmt])
    manhattan(pred_dtfm, x_col='Pos', y_col=['bb_p_adj_bios', 'bb_p_adj_gtex'],
              output_path=output_path, trans=trans_func)

    # output_path = '.'.join([output_pref, 'scatter_plot', plot_fmt])
    # scatter_plot(pred_dtfm, 'bb_p_adj_gtex', 'bb_p_adj_bios',
    # output_path=output_path)


if __name__ == '__main__':
    main()
