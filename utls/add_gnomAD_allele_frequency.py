#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add allele frequency to training set

1. Preprocessing using bash:
optd=/groups/umcg-bios/tmp03/users/umcg-zzhang/projects/ASEPrediction/training/inputs/gnomAD
rgnf=${HOME}/Documents/projects/ASEPrediction/training/outputs/annotCadd/candidate.snps

count=0
for x in $(ls *vcf.gz); do
    bcftools query -H \
        -i 'TYPE="snp"' \
        -f "%CHROM\t%END\t%REF\t%ALT\t%INFO/AF_NFE\n" ${x} \
        -R ${rgnf} \
        > ${optd}/${x/.vcf.gz/_AF_NFE.tsv} &
    count=$[ ${count} + 1 ]
    [[ $[ ${count} % 7 ] -eq 0 ]] && wait
done

cd ${optd}
"""

# TODO: 1 Add argument parser

import sys

gnmd_af = sys.argv[1]  # gnomAD allele frequency file
tnst = sys.argv[2]  # original annotated file without allele frequency from gnomAD database
optf = sys.argv[3]  # Output file

with open(gnmd_af, 'r') as gnmd, open(tnst, 'r') as base, open(optf, "w") as opth:
    gnmd_dict = {tuple(lst[:2]): lst for lst in [x.strip().split("\t") for x in gnmd]}
    base_dict = {tuple(lst[:2]): lst for lst in [x.strip().split("\t") for x in base]}

    gnmd_keys, base_keys = gnmd_dict.keys(), base_dict.keys()

    for key in base_keys:
        allele_frequency = '0'
        if key in gnmd_keys:
            gnmd_rec = gnmd_dict[key]
            base_rec = base_dict[key]
            if gnmd_rec[2] in base_rec and gnmd_rec[3] in base_rec:
                allele_frequency = gnmd_rec[4]
        elif key == ("Chrom", "Pos"):
            allele_frequency = "gnomAD_AF"

        base_line = base_dict[key]
        base_line.insert(108, allele_frequency)
        opth.write("\t".join(base_line) + "\n")

