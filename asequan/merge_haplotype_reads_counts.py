#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A script to merge [PREFIX].haplotype_counts.txt from phASER [https://github.com/secastel/phaser]

Header of .haplotype_counts.txt


contig - Contig haplotype is found on.
start - Start position of haplotype (1 based).
stop - Stop position of haplotype (1 based).
variants - Comma separated list of variant IDs phased in the haplotype.
variantCount - Number of variants contained in the haplotype.
variantsBlacklisted - Comma separated list of variant IDs whose counts were not used to do being blacklisted.
variantCountBlacklisted - Number of variants that were blacklisted
haplotypeA - Comma separated list of alleles found on haplotype A.
aCount - Number of unique reads mapping to haplotype A.
haplotypeB - Comma separated list of alleles found on haplotype B.
bCount - Number of unique reads mapping to haplotype B.
totalCount - Total number of unique reads covering this haplotype (aCount + bCount).
blockGWPhase - The genome wide phase of this haplotype block.
gwStat - phASER calculated genome wide phase statistic for this block.
max_haplo_maf - Maximum variant MAF of all variants phased in this haplotype block.
bam - Name of input BAM. If multiple input BAMs were used, then data will be separated for each BAM.
aReads - Haplotype read indices mapping to each variant on haplotype A.
bReads - Haplotype read indices mapping to each variant on haplotype B.


Check:
    1. --paired_end; Paired end library
    2. --bam; Duplications are masked in BAM files
    3. --mapq; Minimum mapping quality for reads is 255
    4. --baseq; Minimum base quality at the SNP required for reads is 10
    5. --gw_phase_method; Method to use for determinig genome wide phasing is 1 (MAF weighted phase anchoring) 
    6. --gw_phase_vcf; Replace GT field of output VCF using phASER genome wide phase: replace when gw_confidence >= --gw_phase_vec_min_confidence
    7. --blacklist; BED file containing genomic intervals to be excluded from phasing
    8. --haplo_count_blacklist; BED file conftainig genomic intervals to be excluded from haplotypic counts
    9. Population phased: yes

totalCount >= 10
Process all files with given pattern in given directory, while the depth could be controlled by a --search-depth option
"""


import os
import glob
from argparse import ArgumentParser

def get_args():
    """Get CLI arguments and construct a ArgumentParser"""
    parser = ArgumentParser()

    _group = parser.add_argument_group("Input")
    _group.add_argument("-i", "--input-dir", dest="input_dir", type=str, default=".", help="A direcotry including all input files")
    _group.add_argument("-p", "--iputt-file-pattern", dest="input_file_pattern", type=str, default="*.txt", help="A direcotry including all input files. One needs to translate wildcard chars using back-slash (e.g \\*.txt). Default: *.txt")

    _group = parser.add_argument_group("Config")
    _group.add_argument("--contigs", dest="contigs", default=[0], nargs="+", type=int, help="Contigs will be kept. When the value is 0 it will take all contigs. Default: 0")
    _group.add_argument("--min-maf", dest="min_maf", type=float, default=0.001, help="The minimum MAF(minor allele frequency) for chosen variants. Default: 0.001")
    _group.add_argument("--min-respec-counts", dest="min_respec_counts", type=int, default=5, help="The minimu number of allelic reads counts of chosen variants. Default: 5")
    _group.add_argument("--min-total-counts", dest="min_total_counts", type=int, default=10, help="The minimum number of total counts of chosen variants. Default: 10")
    _group.add_argument("--search-depth", dest="search_depth", type=int, default=0, help="The depth of the directory tree to search. Default: 0")
    _group.add_argument("--header-line", dest="header_line", default=1, type=int, help="Which line will be used as header line")

    _group = parser.add_argument_group("Output")
    _group.add_argument("-o", "--output-file", dest="output_file", type=str, default="output.vcf", help="The output file name. Default: output.tsv")
    _group.add_argument("--split-output", dest="split_output", action="store_true", help="Keep output file per input file. Default: off")

    return parser


def main():
    """Main entrance for the script"""
    parser = get_args()
    args = parser.parse_args()

    input_dir = args.input_dir
    input_file_pattern = args.input_file_pattern

    contigs = args.contigs
    min_maf = args.min_maf
    min_total_counts = args.min_total_counts
    min_respec_counts = args.min_respec_counts
    search_depth = args.search_depth

    output_file = args.output_file
    split_output = args.split_output

    _depth_shift = input_dir.rstrip("/").strip("/").count("/") + 1
    search_depth += _depth_shift

    input_file_pool = []
    _search_path = os.path.join(input_dir, input_file_pattern)
    for _input_file_path in glob.glob(_search_path, recursive=True):
        if os.path.isfile(_input_file_path):
            file_relative_depth = _input_file_path.rstrip("/").strip("/").count("/") - 2
            if file_relative_depth <= search_depth:
                input_file_pool.append(_input_file_path)

    kwargs = dict(contigs=contigs, min_maf=min_maf, min_total_counts=min_total_counts, min_respec_counts=min_respec_counts)
    all_record_string = set()
    for input_file in input_file_pool:
        with open(input_file) as ipfh:
            next(ipfh, "EOF")
            record_list = [
                expand_variants_to_vcf_record(x[3])
                for x in [y.split("\t") for y in ipfh.readlines()]
                if my_filter(x, **kwargs)
            ]

        record_string = "\n".join(record_list) + "\n"
        record_string = set(record_string.split("\n"))
        all_record_string = all_record_string.union(record_string)
        record_string = "\n".join(sorted(record_string, key=lambda x: x.split("\t")[0]))

        if split_output:
            output_file = input_file.replace(".txt", ".vcf")
            with open(output_file, "w") as opfh:
                opfh.write(record_string)

    if not split_output:
        all_record_string = "\n".join(sorted(all_record_string, key=lambda x: x.split("\t")[0]))
        with open(output_file, "a") as opfh:
            opfh.write(all_record_string)


def expand_variants_to_vcf_record(variants: str):
    """Expand a variants field into a vcf record"""

    def split_me_join(x):
        tmp = x.split("_")
        return "\t".join([tmp[0], tmp[1], ".", tmp[2], tmp[3], ".", ".", ".", "."])

    return "\n".join([split_me_join(x) for x in variants.split(",")])


def my_filter(line, **kwargs):
    """A simple filter function used to filter out low quality records"""
    # Possible kwargs: contig, start, stop, min_respec_counts, min_total_counts, max_haplo_maf
    contig, start, stop, a_counts, b_counts, total_counts, max_haplo_maf = int(line[0]), int(line[1]), int(line[2]), int(line[9]), int(line[10]), int(line[11]), float(line[14])

    # TODO: 1. contig could be chr1, but here only take integer in account.
    # TODO: 2. without consideration of moultiple variants for the position check.
    # because = lambda x: print("because: {}".format(x))
    if "contigs" in kwargs:
        if 0 not in kwargs["contigs"]:
            if contig not in kwargs["contigs"]:
                # because("contigs")
                return False

        if "start" in kwargs:
            # because("start")
            if start < kwargs["start"]:
                return False

        if "stop" in kwargs:
            # because("stop")
            if stop > kwargs["stop"]:
                return False

    if "min_respec_counts" in kwargs:
        min_respec_counts = kwargs["min_respec_counts"]
        if a_counts <= min_respec_counts or b_counts <= min_respec_counts:
            # because("min_respec_counts")
            return False

    if "min_total_counts" in kwargs:
        if total_counts < kwargs["min_total_counts"]:
            # because("min_total_counts")
            return False

    if "max_haplo_maf" in kwargs:
        if max_haplo_maf < kwargs["max_haplo_maf"]:
            # because("min_haplo_maf")
            return False

    return True


if __name__ == "__main__":
    main()
