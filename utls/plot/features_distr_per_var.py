#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Plot featurest distribution of per variant in a given region.'''

import argparse

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def getargs():
    '''Get CLI options.'''
    parser = argparse.ArgumentParser(description='Plot featurest distribution of per variant in a given region.')

    parser.add_argument('-f', '--input-file', dest='input_file', required=True, help='The input file.')
    parser.add_argument('-i', '--index-cols', dest='index_cols', default=('Chrom', 'Pos', 'Ref', 'Alt'), nargs='*', help='The columns used as index.')
    parser.add_argument('-d', '--field-delim', dest='field_delim', default='\t', help='The delimiter of the input file. Default: %(default)s')
    parser.add_argument('-p', '--ase-prob-col', dest='ase_prob_col', default='prob1_mean', help='The column that indicate the probability of ASE effect. Default: %(default)s')
    parser.add_argument('-t', '--ase-prob-thr', dest='ase_prob_thr', default=0.5, type=float, help='The min probability that a variant is ASE. Default: %(default)s')
    parser.add_argument('-c', '--cols-as-class', dest='cols_as_class', default=[], nargs='*', help='Features to be treated as class variable. Default: %(default)s')
    parser.add_argument('-I', '--incl-features', dest='incl_features', default=None, nargs='*', help='Features to be included. All features will be used if is None (default). Default: %(default)s')
    parser.add_argument('-E', '--excl-features', dest='excl_features', default=None, nargs='*', help='Features to be excluded.  No feature will be excluded if is None (default). Default: %(default)s')
    parser.add_argument('-F', '--figure-format', dest='figure_format', default=['png'], choices=['png', 'svg', 'pdf'], nargs='*', help='The save format of the figure. Default: %(default)s')
    parser.add_argument('-s', '--figure-size', dest='figure_size', default=None, nargs=2, type=float, help='The size of output file. Default: %(default)s')
    parser.add_argument('-o', '--output-prefix', dest='output_prefix', default='./', help='The output prefix. Default: %(default)s')

    return parser


def load_dataset(dtst_path, index_cols, **kwargs):
    '''Load dataset.'''
    dtfm = pd.read_csv(dtst_path, engine='python', **kwargs)
    dtfm.index = [tuple(x) for x in dtfm.loc[:, index_cols].values]
    return dtfm


def prepare_dtfm(dtfm, incl_features, excl_features, cols_as_class, ase_prob_col,
                 true_cond_col='bb_ASE', true_cond_func=abs):
    '''A simple imputater based on pandas DataFrame.replace method.'''
    defaults = {
        'motifEName': 'unknown', 'GeneID': 'unknown', 'GeneName': 'unknown',
        'CCDS': 'unknown', 'Intron': 'unknown', 'Exon': 'unknown', 'ref': 'N',
        'alt': 'N', 'Consequence': 'UNKNOWN', 'GC': 0.42, 'CpG': 0.02,
        'motifECount': 0, 'motifEScoreChng': 0, 'motifEHIPos': 0, 'oAA':
        'unknown', 'nAA': 'unknown', 'cDNApos': 0, 'relcDNApos': 0, 'CDSpos': 0,
        'relCDSpos': 0, 'protPos': 0, 'relProtPos': 0, 'Domain': 'UD',
        'Dst2Splice': 0, 'Dst2SplType': 'unknown', 'minDistTSS': 5.5,
        'minDistTSE': 5.5, 'SIFTcat': 'UD', 'SIFTval': 0, 'PolyPhenCat':
        'unknown', 'PolyPhenVal': 0, 'priPhCons': 0.115, 'mamPhCons': 0.079,
        'verPhCons': 0.094, 'priPhyloP': -0.033, 'mamPhyloP': -0.038,
        'verPhyloP': 0.017, 'bStatistic': 800, 'targetScan': 0, 'mirSVR-Score':
        0, 'mirSVR-E': 0, 'mirSVR-Aln': 0, 'cHmmTssA': 0.0667, 'cHmmTssAFlnk':
        0.0667, 'cHmmTxFlnk': 0.0667, 'cHmmTx': 0.0667, 'cHmmTxWk': 0.0667,
        'cHmmEnhG': 0.0667, 'cHmmEnh': 0.0667, 'cHmmZnfRpts': 0.0667,
        'cHmmHet': 0.667, 'cHmmTssBiv': 0.667, 'cHmmBivFlnk': 0.0667,
        'cHmmEnhBiv': 0.0667, 'cHmmReprPC': 0.0667, 'cHmmReprPCWk': 0.0667,
        'cHmmQuies': 0.0667, 'GerpRS': 0, 'GerpRSpval': 0, 'GerpN': 1.91,
        'GerpS': -0.2, 'TFBS': 0, 'TFBSPeaks': 0, 'TFBSPeaksMax': 0,
        'tOverlapMotifs': 0, 'motifDist': 0, 'Segway': 'unknown', 'EncH3K27Ac':
        0, 'EncH3K4Me1': 0, 'EncH3K4Me3': 0, 'EncExp': 0, 'EncNucleo': 0,
        'EncOCC': 5, 'EncOCCombPVal': 0, 'EncOCDNasePVal': 0, 'EncOCFairePVal':
        0, 'EncOCpolIIPVal': 0, 'EncOCctcfPVal': 0, 'EncOCmycPVal': 0,
        'EncOCDNaseSig': 0, 'EncOCFaireSig': 0, 'EncOCpolIISig': 0,
        'EncOCctcfSig': 0, 'EncOCmycSig': 0, 'Grantham': 0, 'Dist2Mutation': 0,
        'Freq100bp': 0, 'Rare100bp': 0, 'Sngl100bp': 0, 'Freq1000bp': 0,
        'Rare1000bp': 0, 'Sngl1000bp': 0, 'Freq10000bp': 0, 'Rare10000bp': 0,
        'Sngl10000bp': 0, 'dbscSNV-ada_score': 0, 'dbscSNV-rf_score': 0,
        'gnomAD_AF': 0.0, 'pLI_score': 0.303188}

    incl_features = dtfm.columns if incl_features is None else incl_features
    excl_features = [] if excl_features is None else excl_features
    cand_features = [ft for ft in dtfm.columns
                     if ft in incl_features and ft not in excl_features]
    dtfm = dtfm.loc[:, cand_features]

    # Transform true conditions into binary [0, 1]
    if true_cond_col in dtfm.columns:
        dtfm.loc[:, true_cond_col] = dtfm.loc[:, true_cond_col].apply(true_cond_func)

    # Fill NAs
    dtfm.fillna(value=defaults, inplace=True)

    # Encode Lables
    encoder = LabelEncoder()
    tar_cols = list(set([col for col, dt in zip(dtfm.columns, dtfm.dtypes)
                         if str(dt) in ['object']] + cols_as_class))
    tar_enc_cols = ['{}_enc'.format(col) for col in tar_cols]
    for _tag, _tag_enc in zip(tar_cols, tar_enc_cols):
        dtfm.loc[:, _tag] = encoder.fit_transform(dtfm.loc[:, _tag])

    # Scale all values into [0, 1]
    prob1_mean = dtfm.loc[:, ase_prob_col]
    dtfm = dtfm.loc[:, [col for col in dtfm.columns if col != ase_prob_col]]

    scaler = MinMaxScaler()
    dtfm = pd.DataFrame(scaler.fit_transform(dtfm), columns=dtfm.columns,
                        index=dtfm.index)

    if true_cond_col in dtfm.columns:
        dtfm.insert(dtfm.shape[1] - 1, ase_prob_col, prob1_mean)
    else:
        dtfm.loc[:, ase_prob_col] = prob1_mean

    return dtfm


def plot_feature_dist(dtfm, ase_prob_col='prob1_mean', ase_prob_thr=0.5,
                      color_map=('black', 'red'), figure_size=None):
    '''Plot the distribution of features.'''
    width, height = dtfm.shape
    color = (dtfm.loc[ase_prob_col, :]
             .transform(lambda x: color_map[int(x >= ase_prob_thr)]))
    axe = dtfm.plot.line(legend=False, style='--', linewidth=0.1, marker='x',
                         markersize=width * 0.07, color=color)

    axe.set_xticks(range(width))
    axe.set_xticklabels(dtfm.index)
    axe.tick_params(axis='x', which='major', labelrotation=90)

    if figure_size is None:
        _width, _height = width * 0.6, min(height * 0.3, 15)
    else:
        _width, _height = figure_size

    fig = axe.get_figure()
    fig.set_figwidth(_width)
    fig.set_figheight(_height)
    fig.set_tight_layout(True)

    return fig, axe


def main():
    '''The main entry of the script.'''
    options = getargs().parse_args()
    input_file = options.input_file
    index_cols = options.index_cols
    field_delim = options.field_delim
    ase_prob_col = options.ase_prob_col
    ase_prob_thr = options.ase_prob_thr
    incl_features = options.incl_features
    excl_features = options.excl_features
    cols_as_class = options.cols_as_class
    figure_size = options.figure_size
    figure_format = options.figure_format
    output_prefix = options.output_prefix

    # dataframe column data type are numpy.dtype
    dtfm = load_dataset(input_file, sep=field_delim, index_cols=index_cols)
    dtfm = prepare_dtfm(dtfm, incl_features, excl_features, cols_as_class, ase_prob_col)
    fig, _ = plot_feature_dist(dtfm.T, ase_prob_col, ase_prob_thr, figure_size=figure_size)

    if isinstance(figure_format, str):
        figure_format = [figure_format]

    if output_prefix.endswith('/'):
        fig_name = 'feature_dist_plot'
    else:
        fig_name = '-feature_dist_plot'

    for fmt in figure_format:
        save_path = '{}{}.{}'.format(output_prefix, fig_name, fmt)
        fig.savefig(save_path)


if __name__ == '__main__':
    main()
