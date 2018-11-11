#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

encodeMatric = {
    'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 
    'N': [0, 0, 0, 0], '-': [0, 0, 0, 0], 'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 
    'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]
}

def encode(DNA, step=200) -> np.array:
    ''' An encoder translating DNA seq in to matrix

    For instance: 
        A    T    C    G
       1000 0001 0100 0010
    '''

    start = 0
    length = len(DNA)

    while start < length:
        end = start + step
        yield np.array([ encodeMatric[letter] for letter in DNA[start: end] ])
        start = end
