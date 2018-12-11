#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#   File Name  : utls.py
##  Author     : zhzhang
### E-mail     : zhzhang2015@sina.com
### Created on : Mon 12 Nov 2018 09:19:51 PM CET
##  Version    : <unknown>
#   License    : MIT
"""

# Load basic modules
import os
import sys
import time
import logging

from os.path import join

ct = time.clock()  # Time counting starts

# Create stream handler of logging
# Logging info formatter
FORMATTER = '%(asctime)s <%(name)s> %(levelname)s: %(message)s'
formatter = logging.Formatter(FORMATTER, '%Y-%m-%d,%H:%M:%S')

# Set up main logging stream and formatter
CH = logging.streamhandler()
CH.setLevel(logging.INFO)
CH.setFormatter(formatter)

# Set up logging
lg = logging.getLogger()
lg.setLevel(logging.INFO)         # default logging level INFO
lg.addHandler(CH)
lg.info("=== Start ... ===")
