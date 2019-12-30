#!/bin/bash
#SBATCH --time=0:19:0
#SBATCH --mem=15G
#SBATCH --cpus=5
#SBATCH --output=%j-%u-pca_mfa.log
#SBATCH --job-name=pca_mfa

set -o errexit
set -o errtrace

source /apps/modules/modules.bashrc
module load R/3.5.1-foss-2015b-bare
module list

pjd="/home/umcg-zzhang/Documents/projects/ASEPrediction"
Rscript $pjd/scripts/utls/pca_mfa.R \
	-i $pjd/training/outputs/annotCadd/allelicReadsCounts/trainingset_withpLIScore_withGnomADAF_withExonExpVal_exon_FDR0.05.tsv.gz \
    --random-seed 1415
