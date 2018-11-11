#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pprint
from gtfparser import GTFParser


pp = pprint.PrettyPrinter(indent=4)
gtf = GTFParser('Homo_sapiens.GRCh37.75_head20k.gtf')
gtf.GTF2Json()
