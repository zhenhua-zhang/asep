#!/usr/bin/env python3
# -*- coding: utf-8 -*-
methods = [
        'abs', 'add', 'add_prefix', 'add_suffix', 'agg', 'aggregate', 
        'align', 'all', 'altAllele', 'altCount', 'any', 'append', 'apply', 
        'applymap', 'as_matrix', 'asfreq', 'asof', 'assign', 'astype', 'at', 
        'at_time', 'axes', 'between_time', 'bfill', 'bool', 'boxplot', 'clip',
        'clip_lower', 'clip_upper', 'columns', 'combine', 'combine_first', 
        'compound', 'contig', 'copy', 'corr', 'corrwith', 'count', 'cov', 
        'cummax', 'cummin', 'cumprod', 'cumsum', 'describe', 'diff', 'div', 
        'divide', 'dot', 'drop', 'drop_duplicates', 'dropna', 'dtypes', 
        'duplicated', 'empty', 'eq', 'equals', 'eval', 'ewm', 'expanding', 
        'ffill', 'fillna', 'filter', 'first', 'first_valid_index', 'floordiv',
        'from_dict', 'from_records', 'ftypes', 'ge', 'get', 'get_dtype_counts',
        'get_ftype_counts', 'get_values', 'groupby', 'gt', 'head', 'hist', 
        'iat', 'idxmax', 'idxmin', 'iloc', 'index', 'infer_objects', 'info', 
        'insert', 'interpolate', 'isin', 'isna', 'isnull', 'items', 
        'iteritems', 'iterrows', 'itertuples', 'ix', 'join', 'keys', 'kurt', 
        'kurtosis', 'last', 'last_valid_index', 'le', 'loc', 'lookup', 'lt', 
        'mad', 'mask', 'max', 'mean', 'median', 'melt', 'memory_usage', 
        'merge', 'min', 'mod', 'mode', 'mul', 'multiply', 'ndim', 'ne', 
        'nlargest', 'notna', 'notnull', 'nsmallest', 'nunique', 'pct_change', 
        'pipe', 'pivot', 'pivot_table', 'plot', 'pop', 'position', 'pow', 
        'prod', 'product', 'quantile', 'query', 'radd', 'rank', 'rdiv', 
        'refAllele', 'refCount', 'reindex', 'reindex_axis', 'reindex_like', 
        'rename', 'rename_axis', 'reorder_levels', 'replace', 'resample', 
        'reset_index', 'rfloordiv', 'rmod', 'rmul', 'rolling', 'round', 
        'rpow', 'rsub', 'rtruediv', 'sample', 'select', 'select_dtypes', 'sem',
        'set_axis', 'set_index', 'shape', 'shift', 'size', 'skew', 
        'slice_shift', 'sort_index', 'sort_values', 'source', 'squeeze', 
        'stack', 'std', 'style', 'sub', 'subtract', 'sum', 'swapaxes', 
        'swaplevel', 'tail', 'take', 'to_clipboard', 'to_csv', 'to_dense', 
        'to_dict', 'to_excel', 'to_feather', 'to_gbq', 'to_hdf', 'to_html', 
        'to_json', 'to_latex', 'to_msgpack', 'to_panel', 'to_parquet', 
        'to_period', 'to_pickle', 'to_records', 'to_sparse', 'to_sql', 
        'to_stata', 'to_string', 'to_timestamp', 'to_xarray', 'totalCount', 
        'transform', 'transpose', 'truediv', 'truncate', 'tshift', 
        'tz_convert', 'tz_localize', 'unstack', 'update', 'values', 'var', 
        'variantID', 'where', 'xs'
]

import os
import math
# import glob
# import pyfaidx
# import pybedtools
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

fn = ('./chr1_30cov.txt')

df = pd.read_csv(fn, sep='\t', header=0)

df = df[(df['refCount'] != 0) & (df['altCount'] != 0)]
df['log10'] = df['altCount']/df['refCount']
df['log10'] = df['log10'].apply(math.log)

dfGrouped = df.groupby('variantID')

counter = 0
for indx, (name, group) in enumerate(dfGrouped):
    if len(group) >= 500:
        fig, ax = plt.subplots()
        counter += 1
        ax.hist(group['log10'])
        plt.savefig(name + '.png')
        plt.close()
        if counter > 30:
            break

