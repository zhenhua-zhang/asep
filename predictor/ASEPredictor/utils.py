#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Modified by: Zhenhua Zhang (zhzhang2015@sina.com)
# Author: Heng Li and Aaron Quinlan
# License: MIT/X11

import logging
from ctypes import cdll, CDLL
from ctypes.util import find_library


logFormat = logging.Formatter('%(name)s-%{levelname)s: %{message}')
sHandler  = logging.StreamHandler()
sHandler.setFormatter(logFormat)

logger    = logging.getLogger("Utls")
logger.addHandler(sHandler)
logger.setLevel(logging.WARNING)


class Tabix:
    """ A simple API for tabix by Heng Li
    """


    def __init__(self, fn, fnidx=0):
        self.logger = logging.getLogger("utls.Tabix")

        self.tabix = self.tabix_init()
        if(self.tabix == None):
            self.logger.error('Failed to load shared library.')
            return None
        self.fp = self.tabix.ti_open(fn, fnidx)


    def __del__(self):
        if (self.tabix): self.tabix.ti_close(self.fp)


    def tabix_init(self, lib='tabix', pth='./third_party', ver='*'):
        """Find tabix shared library(under Linux). By deault, current 
        directory is the expection("./third_party") no matter the version is.
        """

        libPth = find_library(lib)
        if libPth == None: 
            return None
        cdll.LoadLibrary(libPth)
        return CDLL(libPth)


    def fetch(self, chr, start=-1, end=-1):
        """ Fetch all records or specified records by chrx:start-end
        """
        if self.tabix == None: 
            return None

        if start < 0: 
            iter = self.tabix.ti_querys(self.fp, chr)
        else:
            iter = self.tabix.ti_query(self.fp, chr, start, end)

        if iter == None:
            self.logger.error("Failed to creat iterator...")
            return

        while 1:
            record = self.tabix.ti_read(self.fp, iter, 0)

            if record == None:
                break
            yield record

        self.tabix.ti_iter_destroy(iter)


class VCFParser:
    """ A simple API for vcftool 
    """
    def __init__(self):
        pass


class GTFParser:
    """ A simple API to parse GTF file
    """
    def __init__(self):
        pass


class GenotypParser:
    def __init__(self):
        pass
