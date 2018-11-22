#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#   File Name  : gonlBiosGavin.py
##  Author     : zhzhang
### E-mail     : zhzhang2015@sina.com
### Created on : Thu 22 Nov 2018 09:15:53 PM CET
##  Version    : <unknown>
#   License    : MIT
"""

import time
import logging


# Start a timmer.
CT = time.clock()  # Time counting starts


# Create stream handler of logging
#  Logging info fromatter
F = '%(asctime)s <%(name)s> %(levelname)s: %(message)s'
FORMATTER = logging.Formatter(F, '%Y-%m-%d,%H:%M:%S')

#  Set up main logging stream and formatter
CH = logging.StreamHandler()
CH.setLevel(logging.INFO)
CH.setFormatter(FORMATTER)


# Set up logging
LG = logging.getLogger()
LG.setLevel(logging.INFO)
LG.addHandler(CH)
LG.info("=== Start ... ===")


"""
#
##
###   Your main scope here!!!
##
#
"""


# Finished the logging & time counting ends
LG.info("=== Done ... ===")
END = "Elapsed: %0.5f" % (time.clock() - CT)
LG.info(END)
