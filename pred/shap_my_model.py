#!/usr/bin/env python3
# -*- coding: utf-8 -*
'''A script to draw SHAP plot for given model.
'''

import pickle
import argparse

import shap
import pandas as pd
import matplotlib.pyplot as plt

from asepred import ASEP


_features = (
    'GerpN', 'bStatistic', 'Dist2Mutation', 'cDNApos', 'cHmmReprPCWk',
    'cHmmQuies', 'relcDNApos', 'cHmmTx', 'minDistTSE', 'CDSpos', 'GerpRS',
    'pLI_score', 'cHmmTxWk', 'minDistTSS', 'protPos', 'EncNucleo', 'relProtPos',
    'relCDSpos', 'RawScore', 'priPhCons'
)

_features = (
    'GerpN', 'bStatistic', 'cHmmReprPCWk', 'GerpRS', 'cHmmTxWk', 'minDistTSS',
    'cHmmTx', 'cDNApos', 'minDistTSE', 'cHmmQuies', 'relcDNApos', 'pLI_score',
    'chmmTxFlnk', 'CDSpos', 'protPos', 'relProtPos', 'EncH3K4Me3', 'priPhCons',
    'relCDSpos', 'EncH3K4Me1'
)


def getargs():
    '''Get command line options.
    '''
    parser = argparse.ArgumentParser(description="Draw SHAP plot for the model")
    parser.add_argument('-m', '--model', dest='model', required=True, help='The path to the model for which plot SHAP.')
    parser.add_argument('-n', '--n-samples', dest='n_samples', type=int, default=1000, help='The number of samples used to draw the plot.  Default: %(default)s')

    parser.add_argument('-c', '--color-map', dest='color_map', default='coolwarm', help='Matplotlib colormap used for the plots (violin and dot). Default: %(default)s')
    parser.add_argument('-W', '--plot-width', dest='plot_width', type=int, default=16, help='The width for the plot. Default: %(default)s')
    parser.add_argument('-H', '--plot-height', dest='plot_height', type=int, default=9, help='The height for the plot. Default: %(default)s')

    parser.add_argument('-F', '--features', dest='features', default=_features, nargs='*', help='Features will be used to draw the plots.')

    parser.add_argument('-o', '--output-prefix', dest='output_pref', default='./shap_plot', help='The prefix for output files.  Default: %(default)s')
    parser.add_argument('-f', '--figure-save-fmt', dest='figure_save_fmt', default='png', help='The file format in which to save.  Default: %(default)s')

    return parser


def shap_me(model, n_samples, features, output_pref, savefmt='pdf', cmap='spring',
            plot_width=16, plot_height=9):
    '''Calculate shap values and draw plots.
    '''
    with open(model, 'rb') as model_file_handle:
        my_object = pickle.load(model_file_handle)

    x_matrix_index = my_object.x_matrix.index
    my_x_matrix = my_object.x_matrix.loc[x_matrix_index[:n_samples], ]

    my_model = my_object.fetch_models()[0]
    plt.clf()

    explainer = shap.TreeExplainer(my_model)
    shap_values = explainer.shap_values(my_x_matrix)

    shap_values = (pd
                   .DataFrame(shap_values, columns=my_x_matrix.columns)
                   .loc[:, features]
                   .values)
    my_x_matrix = my_x_matrix.loc[:, features]

    for fig_type in ['dot', 'violin', 'bar']:
        output_path = '{}.summary_{}.{}'.format(output_pref, fig_type, savefmt)
        shap.summary_plot(shap_values, my_x_matrix, plot_type=fig_type,
                          cmap=plt.get_cmap(cmap), alpha=0.7, show=False)
        current_fig = plt.gcf()
        current_fig.set_figheight(plot_height)
        current_fig.set_figwidth(plot_width)
        plt.savefig(output_path)
        plt.clf()

    with open(output_pref + ".shap_values.pkl", 'wb') as pkl_opt:
        pickle.dump(shap_values, pkl_opt)


def main():
    '''The main entry.
    '''
    parser = getargs().parse_args()
    model = parser.model
    n_samples = parser.n_samples
    color_map = parser.color_map
    plot_width = parser.plot_width
    plot_height = parser.plot_height
    output_pref = parser.output_pref
    figure_save_fmt = parser.figure_save_fmt
    features = parser.features

    shap_me(model, n_samples, features, output_pref, figure_save_fmt, color_map,
            plot_width, plot_height)


if __name__ == '__main__':
    main()
