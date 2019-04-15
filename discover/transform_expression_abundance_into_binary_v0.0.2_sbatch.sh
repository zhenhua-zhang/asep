#!/bin/bash
# File Name  : transform_expression_abundance_into_binary_sbatch.sh
# Author     : zhzhang
# E-mail     : zhzhang2015@sina.com
# Created on : Thu 28 Feb 2019 02:39:10 PM CET
# Version    : v0.0.1
# License    : MIT

#SBATCH --time=19:59:0
#SBATCH --output=%j-%u-transform_expression_abundance_into_binary_sbatch_v0.0.2.log
#SBATCH --job-name=teaibs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=55G

module load R/3.3.3-foss-2015b

srun -n 1 -c 1 --mem=30G \
	--error chr1.err \
	--output chr1.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv ../annotCadd/chr1.tsv &
sleep 1m
srun -n 1 -c 1 --mem=15G \
	--error chr2.err \
	--output chr2.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr2_training_set.tsv ../annotCadd/chr2.tsv &
wait

srun -n 1 -c 1 --mem=10G \
	--error chr3.err \
	--output chr3.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr3_training_set.tsv ../annotCadd/chr3.tsv &
sleep 1m
srun -n 1 -c 1 --mem=8G \
	--error chr4.err \
	--output chr4.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr4_training_set.tsv ../annotCadd/chr4.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error chr5.err \
	--output chr5.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr5_training_set.tsv ../annotCadd/chr5.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error chr6.err \
	--output chr6.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr6_training_set.tsv ../annotCadd/chr6.tsv &
sleep 1m
srun -n 1 -c 1 --mem=12G \
	--error chr7.err \
	--output chr7.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr7_training_set.tsv ../annotCadd/chr7.tsv &
wait

srun -n 1 -c 1 --mem=10G \
	--error chr8.err \
	--output chr8.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr8_training_set.tsv ../annotCadd/chr8.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error chr9.err \
	--output chr9.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr9_training_set.tsv ../annotCadd/chr9.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error chr10.err \
	--output chr10.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr10_training_set.tsv ../annotCadd/chr10.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error chr11.err \
	--output chr11.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr11_training_set.tsv ../annotCadd/chr11.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error chr12.err \
	--output chr12.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr12_training_set.tsv ../annotCadd/chr12.tsv &
wait

srun -n 1 -c 1 --mem=3G \
	--error chr13.err \
	--output chr13.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr13_training_set.tsv ../annotCadd/chr13.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error chr14.err \
	--output chr14.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr14_training_set.tsv ../annotCadd/chr14.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error chr15.err \
	--output chr15.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr15_training_set.tsv ../annotCadd/chr15.tsv &
sleep 1m
srun -n 1 -c 1 --mem=12G \
	--error chr16.err \
	--output chr16.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr16_training_set.tsv ../annotCadd/chr16.tsv &
sleep 1m
srun -n 1 -c 1 --mem=15G \
	--error chr17.err \
	--output chr17.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr17_training_set.tsv ../annotCadd/chr17.tsv &
wait

srun -n 1 -c 1 --mem=10G \
	--error chr18.err \
	--output chr18.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr18_training_set.tsv ../annotCadd/chr18.tsv &
sleep 1m
srun -n 1 -c 1 --mem=15G \
	--error chr19.err \
	--output chr19.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr19_training_set.tsv ../annotCadd/chr19.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error chr20.err \
	--output chr20.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr20_training_set.tsv ../annotCadd/chr20.tsv &
sleep 1m
srun -n 1 -c 1 --mem=5G \
	--error chr21.err \
	--output chr21.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr21_training_set.tsv ../annotCadd/chr21.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error chr22.err \
	--output chr22.out \
	Rscript transform_expression_abundance_into_binary_v0.0.2.R chr22_training_set.tsv ../annotCadd/chr22.tsv &
wait


head -1 chr1.tsv > trainig_set.tsv
for fl in chr{1..22}.tsv; do
	sed -n '2,$p' ${fl} >> training_set.tsv
done
