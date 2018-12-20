#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
from numpy import log10
from sys import stderr

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print(e)
    stderr.write('Importerror, while importing maptplotlib.pyplot\n')
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

input_file = 'candidates.json'
with open(input_file) as input_file_handle:
    candidates_pool = json.load(input_file_handle)

df = pd.DataFrame(candidates_pool)
del df['log2FCVar']

fig, ax = plt.subplots()
pvalue = [-log10(x) for x in df.loc['pvalue', :]]
corr = df.loc['correlation']
s = ax.scatter(corr, pvalue, s=2)

idx = 0
middle_panel_pool = {}
for i, text in enumerate(df.columns):
    if text == 'RawScore':
        continue

    h_shift = 0.01
    v_shift = 0

    if corr[i] < -0.15 or corr[i] > 0.15:
        if text == 'gnomad_AF':
            h_shift = -0.12
        if text == 'cadd':
            v_shift = -2
        ax.annotate(text + '({:.3f})'.format(corr[i]),
                    (corr[i]+h_shift, pvalue[i]+v_shift))
    else:
        idx += 4
        ax.annotate('{:.5f}: '.format(corr[i]) + text, (-0.05, 100-idx))

ax.set_xlabel('Correlation by Spearman')
ax.set_ylabel('-log10(p-value)')
ax.set_title('Spearman correlation between predictor and target')

fig.set_figwidth(8)
fig.set_figheight(8)
fig.savefig('candidate_spearman_pvalue_correlation.png')
