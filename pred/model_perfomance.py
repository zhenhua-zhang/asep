#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Evaluate model performance by comparing quant and pred results.
'''

import argparse
import gzip

import numpy as np
import pandas as pd

import sklearn.metrics as smt


def getargs():
    '''Get CLI arguments.
    '''
    parser = argparse.ArgumentParser(description="Evaluate model performance by comparing quantification and prediction results.")
    parser.add_argument('-i', '--input-file', dest='input_file', required=True, help='The input files.')
    parser.add_argument('-o', '--output-pref', dest='output_pref', default='./model_compare', help='The output prefix. Default: %(default)s')

    return parser


def cmp_quan_pred():
    '''Compare quantification and prediction results.
    '''
    pass



class EvalMatrix:
    def __init__(self, para_dict):
        for key, val in para_dict.items():
            setattr(self, key, val)

        self.para_dict = para_dict

    def __str__(self):
        report_str = '''
{title}
>>>> Confusion matrix:
        {s:{width}}    <TRUE>
        {s:{width}}  {ps_lable:<{width}}  {ng_label:<{width}}
<PRED>  {ps_lable:>{width}}  {tpc:<{width}}  {fpc:<{width}}
        {ng_label:>{width}}  {fnc:<{width}}  {tnc:<{width}}

>>>> Labels:
             Positive label: {ps_lable}
             Negative label: {ng_label}

>>>> Statistics:
                   Accuracy: {acc:.{prec}f}
    Accuracy con. int (%95): [{acc_lower:.{prec}f}, {acc_upper:.{prec}f}]
         No infomation rate: {nif:.{prec}f}

         True positive rate: {tpr:.{prec}f}
         True negative rate: {tnr:.{prec}f}
  Positive predictive value: {ppv:.{prec}f}
  Negative predictive value: {npv:.{prec}f}

     Aera under curve (roc): {roc_auc:.{prec}f}
       False discovery rate: {fdr:.{prec}f}'''.format(s='', **self.para_dict)
        return report_str


def acc_ci(mean, err, size, const=1.96):
    itvl = const * np.sqrt((err * (1 - err)) / size)
    return mean - itvl, mean + itvl


def calc_eval_matrix(dtfm, cond, y_true, y_pred=None, y_score=None,
                     y_score_min=0.5, negative_label=0, title='Train evaluation',
                     class_label=('ASE', 'no-ASE'), prec=3):
    '''Calculate evaluation matrix.
    '''
    if isinstance(cond, str):
        chsn_dtfm = dtfm.query(cond)
    else:
        chsn_dtfm = dtfm.loc[cond, :]

    true_cond = chsn_dtfm[y_true] != negative_label
    pred_cond = chsn_dtfm[y_score] >= y_score_min \
            if y_pred is None else chsn_dtfm[y_pred] != negative_label

    tnc, fpc, fnc, tpc = smt.confusion_matrix(true_cond, pred_cond).ravel()

    nif = max(tpc, tnc) / (tnc + tpc + fpc + fnc)
    tpr = tpc / (tpc + fnc) # sensivity, recall, power
    tnr = tnc / (tnc + fpc) # specificity, selectivity

    ppv = tpc / (tpc + fpc) # precision, positive predictive value
    npv = tnc / (tnc + fnc) # negative predictive value
    fdr = fpc / (tpc + fpc) # false discovery rate

    acc = (tpc + tnc) / (tnc + fpc + fnc + tpc)
    acc_lower, acc_upper = acc_ci(acc, 1 - acc, tnc + fpc + fnc + tpc)

    roc_auc = smt.roc_auc_score(true_cond, chsn_dtfm[y_score])

    ps_lable, ng_lable = class_label
    width = max(len(ps_lable), len(ng_lable))
    conf_dict = {'title': title, 'ps_lable': ps_lable, 'ng_label': ng_lable,
                 'tnc': tnc, 'fpc': fpc, 'fnc': fnc, 'tpc': tpc, 'tpr': tpr,
                 'tnr': tnr, 'ppv': ppv, 'npv': npv, 'fdr': fdr, 'nif': nif,
                 'acc': acc, 'acc_lower': acc_lower, 'acc_upper': acc_upper,
                 'roc_auc': roc_auc, 'prec': prec, 'width': width}

    return EvalMatrix(conf_dict)


def main():
    '''The main entry of the script.
    '''
    args = getargs().parse_args()
    input_file = args.input_file
    output_pref = args.output_pref

    dtfm = pd.read_csv(input_file)
    report_path = '{}-performance_report.txt'.format(output_pref)
    with open(report_path, 'w') as opfh:
        # Crosstab of SNVs that are shared by both cohort, apply BIOS model on GTEx cohort
        snvs_both = dtfm['bb_ASE_gtex'].notna() & dtfm['bb_ASE_bios'].notna()
        em = calc_eval_matrix(dtfm, snvs_both, y_true='bb_ASE_gtex',
                              y_score='prob1_mean_gtex',
                              title='Report 1: Shared SNVs, BIOS model on GTEx dataset')
        opfh.write(str(em))

        em = calc_eval_matrix(dtfm, snvs_both, y_true='bb_ASE_bios',
                              y_score='prob1_mean_bios',
                              title='Report 2: Shared SNVs, GTEx model on BIOS dataset')
        opfh.write(str(em))

        # Crosstab of SNVs that only exists in GTEx cohort, apply BIOS model on GTEx cohort.
        snvs_gtex = dtfm['bb_ASE_gtex'].notna() & dtfm['bb_ASE_bios'].isna()
        em = calc_eval_matrix(dtfm, snvs_gtex, y_true='bb_ASE_gtex',
                              y_score='prob1_mean_gtex',
                              title='Report 3: GTEx only SNVs, BIOS model on GTEx dataset')
        opfh.write(str(em))

        # Crosstab of SNVs that only exists in BIOS cohort, apply GTEx model on BIOS cohort.
        snvs_bios = dtfm['bb_ASE_gtex'].isna() & dtfm['bb_ASE_bios'].notna()
        em = calc_eval_matrix(dtfm, snvs_bios, y_true='bb_ASE_bios',
                              y_score='prob1_mean_bios',
                              title='Report 4: BIOS only SNVs, GTEx model on BIOS dataset')
        opfh.write(str(em))

    gtex_only_opt = '{}-gtex_only_snps.csv'.format(output_pref)
    dtfm.loc[snvs_gtex, :].to_csv(gtex_only_opt, index=False)

    bios_only_opt = '{}-bios_only_snps.csv'.format(output_pref)
    dtfm.loc[snvs_bios, :].to_csv(bios_only_opt, index=False)

if __name__ == '__main__':
    main()
