#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Adjustment for reference bias

# Pick only auto and sexual chrommosomes
awk '{if ($1 ~ /^[1-9XY]+){print $0}} Homo_sapiens.GRCh37.71.gtf \
    > Homo_sapiens.GRCh37.71.Chr1-22.X.Y.gtf

# Sort the gtf on by chrommosome(column 1) and coordination(column 4 and 5),
# then compress it by bgzip
cat Homo_sapiens.GRCh37.71.Chr1-22.X.Y.gtf \
    | sort -k1,1d -k4,4n -k5,5n \
    | bgzip > Homo_sapiens.GRCh37.71.Chr1-22.X.Y.gtf.bgz

# Index the compressed file by tabix, also can be done by `pysam.tabix_index()`
tabix -p gff Homo_sapiens.GRCh37.71.Chr1-22.X.Y.gtf.bgz

"""

import pysam
import copy
import gzip


chromsomes = ['x', 'y']
chromsomes.extend([str(x) for x in range(1, 23)])
chromsomes.extend(['chr' + x for x in chromsomes])


G_LINE = ""


def recursive_join(it, sp="\t"):
    """Join the elements of a iterable object"""
    global G_LINE
    for element in it:
        if isinstance(element, (list, tuple)):
            recursive_join(element)
        elif isinstance(element, str):
            G_LINE += element + sp
        else:
            raise ValueError(
                'Cannot concatenate {} to a str'.format(element)
            )
    return G_LINE


def to_tsv(my_list, file_name, mode='w', sp="\t", extinfo=''):
    """Write a list into file"""
    if extinfo:
        bs, ext = os.path.splitext(file_name)
        file_name = bs + "_" + extinfo + ext

    with open(file_name, mode=mode) as opfh:
        for ele in my_list:
            if isinstance(ele, str):
                opfh.writelines(ele)
            elif isinstance(ele, (list, tuple)):
                opfh.writelines(sp.join(ele))
            else:
                raise ValueError("Unknown type to write into a file")


class BiasAdjustment:
    """Adjust reference bias

    When mapping reads to reference genome(e.g. GRCh37), there is a bias that
    reference alleles have more chance be mapped and consequently the expect 
    ratio of alternative-counts / reference-counts is less than 1;
    """

    def __init__(self, fn):
        """Initialization of BiasAdjustment
        """
        self.ipfn = fn
        self.ifh = self.read_file(fn)  # Input file handle

    def __str__(self):
        """__str__() method; return specific strings"""
        return "BiasAdjustment"

    def read_file(self, fn, mode="r"):
        """Load the target file"""
        if fn.endswith(".gz") or fn.endswith(".bgz"):
            return gzip.open(fn, mode=mode)
        else:
            return open(fn, mode=mode)

    def sort_tsv(self, file_name, file_type=None, with_patch=False,
                 chrom_col=None, pos_col=None):
        """Sort tab-separated file """
        if file_type in ["vcf", "bed"]:
            chrom_col, pos_col = 0, 1
        elif file_type in ["gff", "gtf", "gff3"]:
            chrom_col, pos_col = 0, 3
        elif None not in [file_type, chrom_col, pos_col]:
            pass
        else:
            errmsg = [
                "Please specify file_type,",
                "or specify chrom_col and pos_col without specifying file_type"
            ]
            raise ValueError(" ".join(errmsg))

        header, lines = [], []
        file_handler = self.read_file(file_name)

        line = next(file_handler, "EOF")
        assert line != "EOF", "The file cannot be empty."

        while line != "EOF":
            if line.startswith("#"):
                header.append(line)
            else:
                line_list = line.split("\t")
                if with_patch or line_list[chrom_col].lower() in chromsomes:
                    lines.append(line_list)

            line = next(file_handler, "EOF")

        lines = sorted(lines, key=lambda x: (x[chrom_col], int(x[pos_col])))
        header.extend(lines)

        return header

    @staticmethod
    def make_index(file_name):
        """Make index file for input file"""
        f_bs, f_ext = os.path.splitext(file_name)

        def indexed(fn, ext): return os.path.exists(fn + ext)

        def uptodate(fn, ext): return os.getmtime(fn) < os.getmtime(fn + ext)

        infomsg = "{} was indexed and is uptodate. Skipping".format(file_name)
        if f_ext == ".fa":
            if indexed(file_name, ".fai") and uptodate(file_name, ".fai"):
                print(infomsg)
            else:
                pysam.faidx(file_name)
        elif f_ext in [".bam", ".cram"]:
            if indexed(file_name, ".bai") and uptodate(file_name, ".bai"):
                print(infomsg)
            else:
                pysam.index(file_name)
        elif f_ext in [".gff", ".bed", ".vcf", ".sam"]:
            if indexed(file_name, ".gz.tbi") and uptodate(file_name, ".gz.tbi"):
                print(infomsg)
            else:
                pysam.tabix_index(file_name, preset=f_ext.replace(".", ""))

    def parse_fasta(self, fasta_fn):
        """Parse fasta file"""
        with pysam.Fastafile(fasta_fn) as fafh:
            print(fafh.fetch("22", 1, 10))

    def get_gene_region(self, genes=[]):
        """Get the region of target gene"""
        for gene in genes:
            yield dict(
                chrom=gene.g_chrom(),
                start=gene.get_start(),
                end=gene.get_end()
            )

    def overall_p(self):
        """Expectation of global probability of reads including alternative allele"""
        gene_list = []
        region = self.get_gene_retion(genes=gene_list)
        gene_region = "{chrom}:{start}-{end}".format(**region)

        alt_allele_reads = 0
        ref_allele_reads = 0
        if ref_allele_reads != 0:
            return float(alt_allele_reads) / ref_allele_reads
        return None

    def expected_adjusted_alt_counts(self, ):
        """The adjusted expectation of alternative allele counts"""


vcf = "../misc/clinvar_20180401.vcf"
adj = BiasAdjustment(vcf)
