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
#SBATCH --cpus-per-task=5
#SBATCH --mem=55G

module load R/3.3.3-foss-2015b

srun -n 1 -c 1 --mem=30G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv ../annotCadd/chr1.tsv &
sleep 5m
srun -n 1 -c 1 --mem=15G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr2_training_set.tsv ../annotCadd/chr2.tsv &
wait

srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr3_training_set.tsv ../annotCadd/chr3.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr4_training_set.tsv ../annotCadd/chr4.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr5_training_set.tsv ../annotCadd/chr5.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr6_training_set.tsv ../annotCadd/chr6.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr7_training_set.tsv ../annotCadd/chr7.tsv &
wait

srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr8_training_set.tsv ../annotCadd/chr8.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr9_training_set.tsv ../annotCadd/chr9.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr10_training_set.tsv ../annotCadd/chr10.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr11_training_set.tsv ../annotCadd/chr11.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr12_training_set.tsv ../annotCadd/chr12.tsv &
wait

srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr13_training_set.tsv ../annotCadd/chr13.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr14_training_set.tsv ../annotCadd/chr14.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr15_training_set.tsv ../annotCadd/chr15.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr16_training_set.tsv ../annotCadd/chr16.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr17_training_set.tsv ../annotCadd/chr17.tsv &
wait

srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr18_training_set.tsv ../annotCadd/chr18.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr19_training_set.tsv ../annotCadd/chr19.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr20_training_set.tsv ../annotCadd/chr20.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr21_training_set.tsv ../annotCadd/chr21.tsv &
sleep 5m
srun -n 1 -c 1 --mem=10G Rscript transform_expression_abundance_into_binary_v0.0.2.R chr22_training_set.tsv ../annotCadd/chr22.tsv &
wait
