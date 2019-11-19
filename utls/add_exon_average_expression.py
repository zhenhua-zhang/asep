#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Add average expression level of each exon to each variant

1. Exon expression level file:
    /groups/umcg-bios/prm02/projects/expression/combined_exon_count_run_1_passQC.TMM.txt.gz
2. Use the mean of all samples in the file.
3. No MT/sexual chromosome included
4. Only SNV
"""

from argparse import ArgumentParser

import pandas as pd


def main():
    """The docstring"""
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--expression-table", required=True, dest="expression_table",
        help="The file including expression level for each exon"
    )
    parser.add_argument(
        "-v", "--variant-matrix", required=True, dest="variant_matrix",
        help="The file including variants"
    )
    parser.add_argument(
        "-o", "--output-file", default="output_file.tsv", dest="output_file",
        help="The file preocessed dataframe will be dumpped to"
    )

    args = parser.parse_args()

    expression_table = args.expression_table
    variant_matrix = args.variant_matrix
    output_file = args.output_file

    expression_dtfm = pd.read_csv(expression_table, sep="\t", header=0)
    variant_dtfm = pd.read_csv(variant_matrix, sep="\t", header=0)

    expression_tobe_index = expression_dtfm["-"]
    expression_dtfm_new_idx = [
        tuple(23 if "MT" in y or "X" in y or "Y" in y else int(y) for y in x.rsplit("_")[1:])
        for x in expression_tobe_index
    ]

    del expression_dtfm["-"]

    variant_dtfm_new_idx = list(zip(variant_dtfm["Chrom"], variant_dtfm["Pos"]))

    expression_dtfm.index = ["_".join([str(y) for y in x]) for x in expression_dtfm_new_idx]
    variant_dtfm.index = ["_".join([str(y) for y in x]) for x in variant_dtfm_new_idx]

    position_pool = sorted(expression_dtfm_new_idx + variant_dtfm_new_idx, key=lambda x: x[:2])

    clipped_exon_position = (1, 0, 0)
    current_variant_position = (1, 0)
    exon_expression_for_variant = {}

    for position in position_pool:
        if len(position) == 3:
            clipped_exon_position = position
        elif len(position) == 2:
            current_variant_position = position
            variant_chr, variant_pos = current_variant_position
            exon_chr, exon_start, exon_end = clipped_exon_position
            if variant_chr == exon_chr:
                exon_average_expression = 0
                if exon_start <= variant_pos <= exon_end:
                    if current_variant_position in exon_expression_for_variant:
                        print("duplicated variant. This shouldn't happend")
                        continue
                    else:
                        exon_average_expression = expression_dtfm.loc["_".join([str(x) for x in clipped_exon_position]), :].mean()
                exon_expression_for_variant["_".join([str(x) for x in current_variant_position])] = exon_average_expression
            else:
                print("Comparing a variant {} and a exon {} from different chrom, this shouldn't happen".format(current_variant_position, clipped_exon_position))
        else:
            print("Unknown type of position. This shouldn't happen")

    variant_dtfm["exon_exp"] = pd.Series(exon_expression_for_variant)
    variant_dtfm.to_csv(output_file, sep="\t", header=True, index=False)


if __name__ == "__main__":
    main()
