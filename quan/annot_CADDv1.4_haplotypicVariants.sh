#!/bin/bash
#
# File Name  : annot_CADDv1.4.sh
# Author     : zhzhang
# E-mail     : zhzhang2015@sina.com
# Created on : Wed 10 Apr 2019 02:18:03 PM CEST
# Version    : v0.0.1
# License    : MIT
#
#SBATCH --time=23:59:0
#SBATCH --output=%j-%u-annot_CADDv1.4.log
#SBATCH --job-name=annot_CADDv1.4
#SBATCH --ntasks=1
#SBATCH --cpus=1
#SBATCH --mem=5G

module purge
module load CADD/v1.4

CADD.sh -a -g GRCh37 -o /home/umcg-zzhang/Documents/projects/ASEPrediction/training/outputs/annotCadd/haplotypicReadsCounts/haplotypicCountsPerChrCov5.5.tsv.gz -t tmp /home/umcg-zzhang/Documents/projects/ASEPrediction/training/outputs/haplotypicCountsPerChr/haplotypicCountsPerChrCov5.5.vcf

[ $? -eq 0 ] && echo "Success" || echo "Failed"
