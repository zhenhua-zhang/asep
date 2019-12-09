#!/bin/bash
#SBATCH --time=0:5:0

set -o errexit
set -o errtrace

module load BEDTools
module list

cd /home/umcg-zzhang/Documents/projects/ASEPrediction/training/outputs/annotCadd/allelicReadsCounts

bedtools intersect \
    -wa -wb \
    -a <(awk '{if ($0 !~ "chr"){print $1"\t"$2-1"\t"$2"\t"$3}}' variants_exon_withASEeffects_more5Carriers_dbSNPs.tsv) \
    -b <(grep -w protein_coding /apps/data/ftp.ensembl.org/pub/release-71/gtf/homo_sapiens/Homo_sapiens.GRCh37.71.gtf | grep -w exon) \
    | sed -n 's/^.*gene_id "\(.*\)"; transcript_id.*gene_name "\(.*\)\"; gene_biotype.*$/\1\t\2/p' \
    | sort | uniq > GRCh37_71_geneId_geneName_withASEVariant.tsv
