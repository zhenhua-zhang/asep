#!/usr/bin/env python3
# -*- coding : utf-8 -*-
#
##
### author: zhzhang
### e-mail: zhzhang2015@sina.com / zhenhua.zhang217@gmail.com
### data  : 2018.11.5
##
#
################################################################################


# In[2]:

# Outline by Morris

## Extract BIOS variants list(chrom, pos, alt) from BIOS data
## Ask permission to sent this variant list through CADD
## Create a machine learning feature set based on CADD + benign/pathogenic/unknown classifications
## Training machine learning model targeting ASE as quantitative trait, or if sasier as binary trait(using threshold)
## In case pathogenicity is a feature with reasonable weight in the model THEN do pathogenic/benign/population seperately


# In[3]:

# Step 1 
# Filter the BIOS variant by the following:
# 1. Ensure the chr, pos and alt->ref are consistent
# 2. FDRPerVariant < 0.05
# 3. (optional) |log2FC| > 1.0; 
#    Chi square test > 0.01; 
#    in case of BENIGN group, gnomAD_AF < mean of BENIGN
## !!!!! DISCARD THE STEPS IN THIS CELL !!!!!


# In[4]:

# Step 1
# Filter the BIOS variant by the following:
# 1. Ensure the chr, pos and alt->ref are consistent.

# header = ['gene', 'chr', 'pos', 'ref', 'alt', 'group', 'effect', 'impact', 
# 'cadd', 'CaddChrom', 'CaddPos', 'CaddRef', 'CaddAlt', 'Type', 'Length', 
# 'AnnoType', 'Consequence', 'ConsScore', 'ConsDetail', 'GC', 'CpG', 'motifECount', 
# 'motifEName', 'motifEHIPos', 'motifEScoreChng', 'oAA', 'nAA', 'GeneID', 'FeatureID', 
# 'GeneName', 'CCDS', 'Intron', 'Exon', 'cDNApos', 'relcDNApos', 'CDSpos', 
# 'relCDSpos', 'protPos', 'relProtPos', 'Domain', 'Dst2Splice', 'Dst2SplType', 
# 'minDistTSS', 'minDistTSE', 'SIFTcat', 'SIFTval', 'PolyPhenCat', 'PolyPhenVal', 
# 'priPhCons', 'mamPhCons', 'verPhCons', 'priPhyloP', 'mamPhyloP', 'verPhyloP', 
# 'bStatistic', 'targetScan', 'mirSVR-Score', 'mirSVR-E', 'mirSVR-Aln', 
# 'cHmmTssA', 'cHmmTssAFlnk', 'cHmmTxFlnk', 'cHmmTx', 'cHmmTxWk', 'cHmmEnhG', 
# 'cHmmEnh', 'cHmmZnfRpts', 'cHmmHet', 'cHmmTssBiv', 'cHmmBivFlnk', 'cHmmEnhBiv', 
# 'cHmmReprPC', 'cHmmReprPCWk', 'cHmmQuies', 'GerpRS', 'GerpRSpval', 'GerpN', 
# 'GerpS', 'TFBS', 'TFBSPeaks', 'TFBSPeaksMax', 'tOverlapMotifs', 'motifDist', 
# 'Segway', 'EncH3K27Ac', 'EncH3K4Me1', 'EncH3K4Me3', 'EncExp', 'EncNucleo', 
# 'EncOCC', 'EncOCCombPVal', 'EncOCDNasePVal', 'EncOCFairePVal', 'EncOCpolIIPVal', 
# 'EncOCctcfPVal', 'EncOCmycPVal', 'EncOCDNaseSig', 'EncOCFaireSig', 'EncOCpolIISig', 
# 'EncOCctcfSig', 'EncOCmycSig', 'Grantham', 'Dist2Mutation', 'Freq100bp', 
# 'Rare100bp', 'Sngl100bp', 'Freq1000bp', 'Rare1000bp', 'Sngl1000bp', 'Freq10000bp', 
# 'Rare10000bp', 'Sngl10000bp', 'dbscSNV-ada_score', 'dbscSNV-rf_score', 'RawScore', 
# 'PHRED', 'gnomad_AF', 'chrBios', 'posBios', 'refAlleleBios', 'altAlleleBios', 
# 'refCountsBios', 'altCountsBios', 'sampleBios', 'pVal', 'FDROverall', 
# 'FDRPerVariant', 'varInsideChi2Pval', 'log2FC' ]


# In[8]:

# candidates = ['chr', 'pos', 'ref', 'alt', 'group', 'chrBios', 'posBios', 
#              'refAlleleBios', 'altAlleleBios']
# for x in candidates:
#    print(x, header.index(x) + 1)


# In[22]:

#!/usr/bin/env python3
# -*- coding : utf-8 -*-
#
##
### author: zhzhang
### e-mail: zhzhang2015@sina.com / zhenhua.zhang217@gmail.com
### data  : 2018.11.5
### name  : fgBiosVariants.py  filter and get BIOS variants
##
#
################################################################################

# Load libraries
import os
import sys                                                                      
import time
import logging

ct = time.clock()  # Time counting starts

# Create stream handler of logging
## Logging info formatter
FORMATTER = '%(asctime)s <%(name)s> %(levelname)s: %(message)s'
formatter = logging.Formatter(FORMATTER, '%Y-%m-%d,%H:%M:%S')

## Set up main logging stream and formatter
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

# Set up logging
lg = logging.getLogger()
lg.setLevel(logging.INFO)         # default logging level INFO
lg.addHandler(ch)
lg.info("=== Start ... ===")

# Load extra modules
lg.info("Load extra modules...")
from os.path import join

# Arrange working dirs
lg.info('Arrange working dirs...')
pjDir = '/home/umcg-zzhang/Documents/projects/ASEpredictor'
pjIpDir = join(pjDir, 'inputs')
pjOpDir = join(pjDir, 'outputs')

wkDir = join(pjOpDir, 'biosGavinOverlapCov10')
ipfn = join(wkDir, 'biosGavinOverlapCov10AnnoUnFiltered.tsv')
opfn = join(wkDir, 'biosGavinOlCv10AntUfltCst.tsv')            # bios GAVIN overlap coverage 10 annotated unfiltered consistent
epfn = join(wkDir, 'biosGavinOlCv10AntUfltUncst.tsv')          # bios GAVIN overlap coverage 10 annotated unfiltered unconsistent

# Filter the unconsistent recordes
descriptDict = { 
    0: 'no_css',             # all not consistent.
    1: 'chr_css',            # chr is consistent.
    2: 'pos_css',            # pos is consistent.
    3: 'chr_pos_css',        # chr and pos are consistent.                                              
    4: 'ref_css',            # ref is consistent.
    5: 'chr_ref_css',        # chr and ref and consistent.
    6: 'pos_ref_css',        # pos and ref are consistent.
    7: 'chr_pos_ref_css',    # chr, pos and ref are consistent.
    8: 'alt_css',            # alt is consistent.
    9: 'chr_alt_css',        # chr and alt are consistent.
    10: 'pos_alt_css',       # pos and alt are consistent.
    11: 'chr_pos_alt_css',   # chr, pos, and alt are consistent.                                         
    12: 'ref_alt_css',       # ref and alt are consistent.
    13: 'chr_ref_alt_css',   # chr, ref and alt are consistent.
    14: 'pos_ref_alt_css',   # pos, ref and alt are consistent.
    15: 'all_css'            # all are consistent.
} 


# ipfn / ipfh. input file name and file handle
# opfn / opfh. output file name and file handle
# epfn / epfh. error line file name and file handles

with open(ipfn, 'r') as ipfh, open(opfn, 'w') as opfh, open(epfn, 'w') as epfh:
    header = next(ipfh, "EOF").strip()
    et = 'errType'
    header += '\t' + et + '\n'
    epfh.write(header)
    opfh.write(header)
    
    lineStr = next(ipfh, "EOF").strip()
    while lineStr != "EOF":
        lpList = lineStr.split('\t')

        if (lpList[1]==lpList[117]): cChr = 1
        else: cChr = 0
            
        if (lpList[2]==lpList[118]): cPos = 2
        else: cPos = 0

        if (lpList[3]==lpList[119]): cRef = 4
        else: cRef = 0

        if (lpList[4]==lpList[110]): cAlt = 8
        else: cAlt = 0 
 
        check = cChr + cPos + cRef + cAlt;
        et = descriptDict[check]

        if check == 15:  # all are consistent
            opfh.write(lineStr + '\t' + et + '\n')
        else:            # not all consistent
            epfh.write(lineStr + '\t' + et + '\n')

        lineStr = next(ipfh, "EOF").strip()


# Finished the logging & time counting ends
lg.info("=== Done  ... ===\nTime elapsed: %0.2sf" %(time.clock()-ct) )

