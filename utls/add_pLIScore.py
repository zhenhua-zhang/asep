#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math

pli_score_file = sys.argv[1]  # pLI score file from ExAC
training_set_file = sys.argv[2]

with open(pli_score_file, 'r') as pli_score_file_handle, \
        open(training_set_file, 'r') as training_set_file_handle:
    pli_score_dict = {
        x.split("\t")[0].split(".")[0]: x.split("\t")[-3]
        for x in pli_score_file_handle if x.startswith("ENST")
    }
    pli_score_dict["FeatureID"] = "pLI_score"

    training_set_added_pLI_list = [
        (z.insert(107, pli_score_dict.get(z[19], "NA")), z)[-1]
        for z in [y.split("\t") for y in training_set_file_handle]
    ]

    with open("output.tsv", 'w') as output_file_handler:
        for line_list in (training_set_added_pLI_list):
            line = "\t".join(line_list)
            output_file_handler.writelines(line)

