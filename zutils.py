#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File name : zutils.py
# Author    : Zhenhua Zhang
# E-mail    : zhenhua.zhang217@gmail.com
# Created   : Thu 02 Jul 2020 09:38:34 AM CEST
# Version   : v0.1.0
# License   : MIT
#


def print_header(title=None, version=None, author=None, email=None, institute=None, url=None):
    """A function to print a header including information of the package"""
    astr = "{: ^80}\n"
    bstr = "#{: ^48}#"
    head = astr.format("#" * 50)

    if title is None:
        title = "Allele-Specific Expression Predictor"
    head += astr.format(bstr.format(title))
    if version is None:
        version = 'Version 0.1.0'
    head += astr.format(bstr.format(version))
    if author is None:
        author = 'Zhen-hua Zhang'
    head += astr.format(bstr.format(author))
    if email is None:
        email = 'zhenhua.zhang217@gmail.com'
    head += astr.format(bstr.format(email))
    if institute is None:
        head += astr.format(bstr.format('Genomics Coordination Centre'))
        head += astr.format(bstr.format("University Medical Centre Groningen"))
    elif isinstance(institute, (tuple, list)):
        for i in institute:
            head += astr.format(bstr.format(i))
    if url is None:
        url = 'https://github.com/zhenhua-zhang/asep'
    head += astr.format(bstr.format(url))

    head += astr.format("#" * 50)
    print("\n", head, file=sys.stderr, sep="")

