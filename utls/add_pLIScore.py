#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

pli_score_file = sys.argv[1]  # pLI score file from ExAC
training_set_file = sys.argv[2]
output_file = sys.argv[3]

with open(pli_score_file, 'r') as pli_score_file_handle, \
        open(training_set_file, 'r') as training_set_file_handle:
    pli_score_dict = {
        x.split("\t")[0].split(".")[0]: x.split("\t")[-3]
        for x in pli_score_file_handle if x.startswith("ENST")
    }
    pli_score_dict["FeatureID"] = "pLI_score"

    training_set_added_pLI_list = [
        s.strip().replace('#Chr', 'Chrom') + '\t' + pli_score_dict.get(s.split('\t')[19], "NA")
        for s in training_set_file_handle if not s.startswith('##')
    ]

    with open(output_file, 'w') as output_file_handler:
        for line in training_set_added_pLI_list:
            line += "\n"
            output_file_handler.write(line)
