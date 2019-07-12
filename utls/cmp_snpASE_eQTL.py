#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I'm module docstring"""

import os
import math
import logging
import argparse

import pandas as pd
import matplotlib.pyplot as plt


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.INFO)
FMT = logging.Formatter(
    "|{levelname: ^10}| {asctime} | {funcName: <12}| {message}", style="{",
    datefmt="%Y-%m-%d %H:%M:%S"
)
STREAM_HANDLER.setFormatter(FMT)
LOGGER.addHandler(STREAM_HANDLER)


def fetch_arg(log=LOGGER):
    """Parse arguments"""
    log.info("Parsing arguments...")

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("Inputs")
    group.add_argument(
        "-e", "--eqtl-file", dest="eqtl", required=True, type=str,
        help="File including eQTLs"
    )
    group.add_argument(
        "-s", "--snp-ase-file", dest="sase", required=True, type=str,
        help="File including SNP level ASE"
    )
    group.add_argument(
        "--eqtl-col", dest="eqtlcol", default="2,3,5,0,1,4", type=str,
        help="Columns in eQTL files will be used for the intersection." +
        " Index should indicate [chrom, position, alleles, pvalue, snpid, gene]"
    )
    group.add_argument(
        "--sase-col", dest="sasecol", default="0,1,2,3,112", type=str,
        help="Columns in snp-ASE files will be used for the intersection." +
        " Index should indicate [chromosome, position, ref, alt, pvalue]"
    )
    group.add_argument(
        "-o", "--opt-dir", dest="opt_dir", default="output", type=str,
        help="Ouput directory"
    )

    return parser.parse_args()

def intersect(eqtlf, sasef, eqtlcol, sasecol, eqtldel='\t', sasedel='\t',
              optfn="output.csv", fdr=0.05, log=LOGGER):
    """A function to check the intesection between snp-ASE and eQTL"""

    if len(eqtlcol) != 6:
        log.error("Wrong length of eqltcol, should be 5 ...")
    e_chrom, e_pos, e_alleles, e_pvalue, e_snp, e_gene = eqtlcol

    if len(sasecol) != 5:
        log.error("Wrong length of sasecol, should be 5 ...")
    s_chrom, s_pos, s_ref, s_alt, s_pvalue = sasecol

    with open(eqtlf, 'r') as eqtlfh, open(sasef, 'r') as sasefh:
        eqtlhm = {
            (rec[e_chrom], rec[e_pos]): [
                rec[e_chrom], rec[e_pos], rec[e_alleles], rec[e_pvalue],
                rec[e_snp], rec[e_gene]
            ]
            for rec in [x.strip().split(eqtldel) for x in eqtlfh]
        }

        sasehm = {
            (rec[s_chrom], rec[s_pos]): [
                rec[s_chrom], rec[s_pos], rec[s_ref], rec[s_alt], rec[s_pvalue]
            ]
            for rec in [x.strip().split(sasedel) for x in sasefh]
            if rec[s_pvalue] not in ["NA", "bb_p_adj"] \
                and float(rec[s_pvalue]) <= fdr
        }

    eqtlrecs = eqtlhm.keys()
    saserecs = sasehm.keys()

    shared_loci = list(set(eqtlrecs) & set(saserecs))
    log.info("Shared loci: {}".format(str(len(shared_loci))))

    with open(optfn, "w") as optfh:
        optfh.write("snp\tchr\tpos\tref\talt\teqtl_pv\tsase_pv\teqtl_gnsb\n")

        for shared_locus in shared_loci:
            e_chrom, e_pos, e_alleles, e_pvalue, e_snp, e_gene \
                = eqtlhm[shared_locus]

            s_chrom, s_pos, s_ref, s_alt, s_pvalue = sasehm[shared_locus]

            is_identical = (
                (s_ref in e_alleles) and (s_alt in e_alleles)
                and (e_chrom == s_chrom) and (e_pos == s_pos)
            )

            if is_identical:
                optfh.write(
                    "\t".join(
                        [
                            e_snp, e_chrom, e_pos, s_ref, s_alt, e_pvalue,
                            s_pvalue, e_gene, "\n"
                        ]
                    )
                )
            else:
                log.warning("Not consist at: " + e_chrom + ":" +  e_pos)


def pval_scatter_plot(shared_loci_file, optfn="output.pdf", log=LOGGER):
    """Scatter plot of shared loci between eQTL and sASE"""

    log.info("pval_scatter_plot start ...")

    dtfm = pd.read_csv(shared_loci_file, sep="\t")
    dtfm = dtfm.loc[dtfm["sase_pval"] <= 0.05]

    fig = plt.figure()
    axe = fig.add_subplot(111)
    dtfm["eqtl_pval"] = dtfm["eqtl_pval"].apply(
        lambda x: 20 if x <= 5e-20 else -math.log10(x)
    )
    dtfm["sase_pval"] = dtfm["sase_pval"].apply(
        lambda x: 20 if x <= 5e-20 else -math.log10(x)
    )
    axe.scatter(dtfm["eqtl_pval"], dtfm["sase_pval"], s=0.25, c="red")
    axe.set_xlabel("p-value of eQTL in BIOS")
    axe.set_ylabel("p-value of sASE in BIOS")
    axe.set_title("eQTL p-value vs sASE p-value")
    fig.savefig(optfn, figsize=(10, 10))

    log.info("pval_scatter_plot end ...")


def main(log=LOGGER):
    """Main function"""

    log.info("Main function start ...")

    arguments = fetch_arg()
    eqtl_file = arguments.eqtl
    sase_file = arguments.sase
    opt_dir = arguments.opt_dir
    eqtlcol = [int(x) for x in arguments.eqtlcol.split(",")]
    sasecol = [int(x) for x in arguments.sasecol.split(",")]
    os.makedirs(opt_dir, exist_ok=True)

    # eqtlcol = [chromosome, position, alleles, pvalue, snpid, gene]
    # eqtlcol = [2, 3, 7, 0, 1, 4]
    # eqtlcol = [2, 3, 5, 0, 1, 4]

    # sasecol = [chromosome, position, ref, alt, pvalue]
    # sasecol = [0, 1, 2, 3, 112]
    optfn = os.path.join(opt_dir, "output.tsv")
    intersect(eqtl_file, sase_file, eqtlcol, sasecol, optfn=optfn)

    iptfn = optfn
    optfn = os.path.join(opt_dir, "eQTL_vs_sASE_pval_scatter.pdf")
    pval_scatter_plot(iptfn, optfn)

    log.info("Main function end ...")

if __name__ == "__main__":
    main()
