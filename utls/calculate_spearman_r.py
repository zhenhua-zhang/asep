#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A modified Spearman's rank correlation coefficient calculator

@background
"""

import pickle


def read_msmt(pkl_fpt):
    """Read measurements"""
    with open(pkl_fpt, "rb") as pkl_rh:
        fi_dict = pickle.load(pkl_rh)

    return {key: [key, sum(val) / len(val)] for key, val in fi_dict.items()}


def rank(valst, key=-1):
    """Rank features by importances"""

    lstln = len(valst) - 1
    valst = sorted(valst, key=lambda x: x[key])

    _idx = 0
    while _idx <= lstln:
        if _idx == lstln:
            valst[_idx].append(_idx + 1.0)
            break

        _cidx = _idx

        while _idx + 1 <= lstln and valst[_idx][key] == valst[_idx+1][key]:
            _idx += 1

        _rank = (_idx + _cidx) / 2.0

        while _cidx <= _idx:
            valst[_cidx].append(_rank + 1)
            _cidx += 1

        _idx += 1

    return valst


def calc_spearman_r(a_dict, b_dict):
    """Calculate the Spearman's rank correlation coefficient for two lists"""
    a_keys = set(a_dict.keys())
    b_keys = set(b_dict.keys())

    all_keys = a_keys | b_keys

    dff_keys = a_keys ^ b_keys
    if dff_keys:
        print("inconsistent keys in two rank set. Non-shared keys will be ranked as -1")

    a_ranks = rank(a_dict.values())
    a_dict = {rec[0]: rec for rec in a_ranks}

    b_ranks = rank(b_dict.values())
    b_dict = {rec[0]: rec for rec in b_ranks}

    n_recs = len(all_keys)
    sumd2 = sum([(a_dict[key][-1] - b_dict[key][-1])**2 for key in all_keys])
    srcc = 1 - 6 * sumd2 / float(n_recs ** 3 - n_recs)
    return  srcc


def main():
    """Main entry of current script"""
    bios_pkl_path = "/home/umcg-zzhang/Documents/projects/ASEPrediction/training/outputs/predictor/allelicCounts/2019_Aug_30_07_30_51_bios_gbc_ini100_ic6_oc6_mngs5_nc_pLI_gAF_exn_fdr0.05/feature_importances.pkl"
    a_dict = read_msmt(bios_pkl_path)

    gtex_pkl_path = "/home/umcg-zzhang/Documents/projects/ASEPrediction/validating/outputs/predictor/2019_Aug_30_08_02_38_gtex_gbc_ini100_ic6_oc6_mngs5_pLI_gnAF_exn_fdr0.05/feature_importances.pkl"
    b_dict = read_msmt(gtex_pkl_path )

    srcc = calc_spearman_r(a_dict, b_dict)
    print("Spearman's rank correlation coefficient is:", srcc)


if __name__ == "__main__":
    main()

