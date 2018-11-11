#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utls import *

fileName = [
    "BIOS_LLDeep_noRNAeditSites_phASER", "lldeep_FC7b_L4_I7_103001268811", 
    "chr1", "allelic_counts", "txt"
]

allelicPath = "/".join(PATH['Allelic'])
chr1inf = allelicPath + "/chr1/" + ".".join(fileName)

chr = pd.read_csv(chr1inf, header=0, sep="\t")
chr['refBias'] = chr.loc[:, 'refCount'] / chr.loc[:, 'altCount']

def slog(x):
    if x <= 0:
        return -3
    else:
        return math.log(x)

chr['refBias'] = chr.apply(lambda x: slog(x[-1]), axis=1)

chr['shiftColor'] = chr.apply(lambda x: colorPool[x[2][-3:]], axis=1)


fig, ax = plt.subplots()
ax.scatter(
    chr['position'], chr['refBias'], c=chr['shiftColor'], s=1, alpha=0.5
)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
plt.savefig("tmp.svg")
