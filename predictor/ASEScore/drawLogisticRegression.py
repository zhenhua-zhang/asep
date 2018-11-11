#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import pdb; pdb.set_trace()

import numpy as np
import scipy as sp

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

def expFunc(arr, shift):
    return 1/(1+np.exp(-arrays + shift))


shift = 15
scale = 5.5

arrays = np.linspace(-shift, shift, 500)

expVal = expFunc(arrays, -scale) + expFunc(arrays, scale) - 1
plt.axhline(0, c='r')
plt.plot(expVal)

plt.savefig('logistic_function.png')
