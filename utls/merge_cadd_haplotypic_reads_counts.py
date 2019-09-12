#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Merge CADD annotation and heplotyplic reads counts (from phASER) per individuals

    Current input-dir: /home/umcg-zzhang/Documents/projects/ASEPrediction/training/inputs/haplotypicCountsPerChr

    NOTE:
        1. File name rules. A mixture of snake_case, chain.case and camlCase
            - chromosome
            - source
            - operations
            - extra field
            - extension
            [chromosome]_[source_0].[source_1]_[operation_0].[operation_1].[operation_n]_[extra_field_0].[extra_field_1].[extension]
            e.g. chr1_bios_haplotyplicReadsCounts.rmDup_Cov5v5.tsv
    TODO: 1. A CLI options to choose output directory
"""

import os
from argparse import ArgumentParser
from sys import stderr as STDE

def get_args():
    """Get CLI options"""
    parser = ArgumentParser()

    _group = parser.add_argument_group("Input")
    _group.add_argument("-a", "--annot-file", dest="annot_file", type=str, required=True, help="File of annotations form CADD. Required")
    _group.add_argument("-i", "--input-dir", dest="input_dir", type=str, default="./", help="Directory including all required input files. Default: ./")

    _group = parser.add_argument_group("Configuration")
    _group.add_argument("--file-name-delimiter", dest="file_name_delimiter", type=str, default="_", help="The puncuation will be used to split the file name. Default: _")
    _group.add_argument("--input-file-ext", dest="input_file_ext", type=str, default="\\*.tsv", help="The extension name of input file. Default: \\*.tsv")
    _group.add_argument("--skip-n-lines", dest="skip_n_lines", type=int, default=1, help="Whether skip n lines for each input file or not (0). Default: 0")

    _group = parser.add_argument_group("Output")
    _group.add_argument("--one-file-per-chr", dest="one_file_per_chr", action="store_true", help="Output the merged result one file per chromsome")
    _group.add_argument("-o", "--output-file", dest="output_file", type=str, default="haplotypicReadsCountsPerChrCov5.5_trainingSet.tsv", help="The output file. Default: haplotypicReadsCountsPerChrCov5.5_trainingSet.tsv")

    return parser


def merge(args):
    """Core implementation of current script"""
    annot_file = args.annot_file  # File of annotations from CADD
    input_dir = args.input_dir

    file_name_delimiter = args.file_name_delimiter
    input_file_ext = args.input_file_ext
    skip_n_lines = args.skip_n_lines

    one_file_per_chr = args.one_file_per_chr
    output_file = args.output_file

    with open(annot_file, "r") as flhd:
        header = next(flhd, "EOF").replace("#", "")
        new_header = "\t".join([
            header.strip("\n"), "chrBios", "posBios", "refAlleleBios",
            "altAlleleBios", "refCountsBios", "altCountsBios", "sampleBios"
        ])
        annot_dict = {tuple(line.split("\t")[:4]): line for line in flhd}  # TODO: A new implementation to handle duplications

    for idx, file_name in enumerate(os.listdir(input_dir)):
        lost_variants = 0
        input_file_path = os.path.join(input_dir, file_name)
        print("Working on:", input_file_path, file=STDE)

        if one_file_per_chr:
            write_mode = "a"  # Need to check whether the file exists to avoid messing up the output
            output_file = os.path.join(input_dir, file_name.replace(input_file_ext, "_output" + input_file_ext))
        else:
            write_mode = "w"
            output_file = os.path.join(input_dir, output_file)

        with open(input_file_path, "r") as ipfh, open(output_file, write_mode) as opfh:
            if one_file_per_chr or idx == 0:
                opfh.write(new_header + "\n")

            while skip_n_lines:
                next(ipfh, "EOF")  # Skip header, need a CLI option to handle it
                skip_n_lines -= 1

            for line in ipfh:
                line_list = line.split("\t")
                variants_id = line_list[3]
                a_count, b_count, sample_id = line_list[9], line_list[10], line_list[15].replace(".mdup.sorted.readGroupsAdded", "")  # Remove suffix for the smaple id. .mdup.sorted.readGroupsAdded
                for variant_id in variants_id.split(","):
                    variant_key = tuple(variant_id.split("_"))
                    annot_info = annot_dict.get(variant_key)
                    if annot_info is not None:
                        output_line = "\t".join([annot_info.strip("\n"), "\t".join(variant_key), a_count, b_count, sample_id])
                        opfh.write(output_line + "\n")
                    else:
                        lost_variants += 1
        print("  -- Nr. of lost variants:", lost_variants)


def main():
    """Main entry of the script"""
    parser = get_args()
    args = parser.parse_args()
    merge(args)


if __name__ == "__main__":
    main()
