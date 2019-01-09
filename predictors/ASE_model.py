#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""


class ASEModel:
    """Modeling contribution of cis-regulatory regulatory variance of ASE

    Example:
    >>> import ASEModel
    >>> am = ASEModel('your-traing-set.tsv')
    >>> am.run()
    >>> am.train()
    >>> am.predict()

    """

    def __init__(self, fn):
        """Constrution of a ASEModel instance"""
        self.file_name = fn
