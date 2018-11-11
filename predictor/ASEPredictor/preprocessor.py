#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import pdb; pdb.set_trace()
import os
import glob
import logging
import pybedtools
from pyfaidx import Fasta
from pybedtools import BedTool
from pybedtools import featurefuncs


FORMAT = logging.Formatter('>>[%(asctime)-15s] <%(name)s> %(message)s') 
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(FORMAT)
logger = logging.getLogger("preprocessor")
logger.addHandler(ch)
logger.setLevel(logging.INFO)


home = '/home/umcg-zzhang'
workDir = home + '/umcg-zzhang'
projectDir = workDir + '/projects/ASEpredictor'
inputDir = projectDir + '/inputs'
referenceDir = inputDir + '/reference'
gtfDir = referenceDir + '/gtf'
genomeDir = referenceDir + '/genome'
outputDir = projectDir + '/outputs'


logger.info("==== Start ====")
logger.info('Read BED file ...')
annotationF = gtfDir + '/Homo_sapiens.GRCh37.75.gene_interval.20.gtf'
genomeIntervals = genomeDir + '/genome_interval'
annotationGtf = BedTool(annotationF)

# logger.info("annotationBed")
annotationBed = annotationGtf.each(featurefuncs.gff2bed, name_field='gene_id')
annotationBed.saveas('tmp.bed')
annotationBed = BedTool('tmp.bed')  # print(annotationBed)

# logger.info("upstream2000")
upstream2000 = annotationBed.each(
    featurefuncs.five_prime, upstream=1000, downstream=10, 
    add_to_name='_upstream_2000'
)  #print(upstream2000)

# logger.info("downstream2000")
downstream2000 = annotationBed.each(
    featurefuncs.three_prime, upstream=10, downstream=1000, 
    add_to_name='_downstream_2000'
)  #print(downstream2000)

logger.info('Read FASTA file ...')
sequencesF = genomeDir + '/human_g1k_v37.fasta'
genome = Fasta(sequencesF)
sequences = BedTool(sequencesF)

logger.info('Fetch upstream sequences ...')
upseq = upstream2000.sequence(fi=sequences, name=True, s=True)
upseq.save_seqs('./upstream_2000.fasta')

logger.info('Fetch downstream sequences ...')
dwseq = downstream2000.sequence(fi=sequences, name=True, s=True)
dwseq.save_seqs('./downstream_2000.fasta')

logger.info('==== Done =====')

# featurePool = [
#     'append', 'attrs', 'chrom', 'count', 'deparse_attrs', 'end', 'fields', 
#     'file_type', 'length', 'name', 'o_amt', 'o_end', 'o_start', 'score', 
#     'start', 'stop', 'strand'
# ]
