#!/bin/bash
# File Name  : transform_expression_abundance_into_binary_sbatch.sh
# Author     : zhzhang
# E-mail     : zhzhang2015@sina.com
# Created on : Thu 28 Feb 2019 02:39:10 PM CET
# Version    : v0.0.1
# License    : MIT

#SBATCH --time=9:59:0
#SBATCH --output=%j-%u-transform_expression_abundance_into_binary_sbatch_v0.0.2.log
#SBATCH --job-name=teaibs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G

module load R/3.3.3-foss-2015b

Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr2_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr3_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr4_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr5_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr6_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr7_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr8_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr9_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr10_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr11_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv chr1.tsv
Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv chr1.tsv
