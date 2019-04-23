#!/bin/bash
# File Name  : transform_expression_abundance_into_binary_sbatch.sh
# Author     : zhzhang
# E-mail     : zhzhang2015@sina.com
# Created on : Thu 28 Feb 2019 02:39:10 PM CET
# Version    : v0.0.1
# License    : MIT

#SBATCH --time=5:59:0
#SBATCH --output=%j-%u-transform_expression_abundance_into_binary_v0.0.2.log
#SBATCH --job-name=teaibs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=55G

module load R/3.3.3-foss-2015b

check_dir(){
    [ -d $1 ] && echo "Find $1" || exit 1
}

WKHM=/home/umcg-zzhang/Documents/projects/ASEPrediction/training
exeDir=${WKHM}/scripts
dstDir=${WKHM}/outputs/annotCadd
scpName=transform_expression_abundance_into_binary_v0.0.2.R

errDir=${dstDir}/errs
outDir=${dstDir}/outs
mkdir ${outDir} ${errDir}

check_dir ${exeDir}
check_dir ${dstDir}
check_dir ${errDir}
check_dir ${outDir}

srun -n 1 -c 1 --mem=30G \
	--error ${errDir}/chr1.err \
	--output ${outDir}/chr1.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr1_training_set.tsv ${dstDir}/chr1.tsv &
sleep 1m
srun -n 1 -c 1 --mem=15G \
	--error ${errDir}/chr2.err \
	--output ${outDir}/chr2.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr2_training_set.tsv ${dstDir}/chr2.tsv &
wait

srun -n 1 -c 1 --mem=10G \
	--error ${errDir}/chr3.err \
	--output ${outDir}/chr3.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr3_training_set.tsv ${dstDir}/chr3.tsv &
sleep 1m
srun -n 1 -c 1 --mem=8G \
	--error ${errDir}/chr4.err \
	--output ${outDir}/chr4.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr4_training_set.tsv ${dstDir}/chr4.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error ${errDir}/chr5.err \
	--output ${outDir}/chr5.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr5_training_set.tsv ${dstDir}/chr5.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error ${errDir}/chr6.err \
	--output ${outDir}/chr6.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr6_training_set.tsv ${dstDir}/chr6.tsv &
sleep 1m
srun -n 1 -c 1 --mem=12G \
	--error ${errDir}/chr7.err \
	--output ${outDir}/chr7.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr7_training_set.tsv ${dstDir}/chr7.tsv &
wait

srun -n 1 -c 1 --mem=10G \
	--error ${errDir}/chr8.err \
	--output ${outDir}/chr8.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr8_training_set.tsv ${dstDir}/chr8.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error ${errDir}/chr9.err \
	--output ${outDir}/chr9.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr9_training_set.tsv ${dstDir}/chr9.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error ${errDir}/chr10.err \
	--output ${outDir}/chr10.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr10_training_set.tsv ${dstDir}/chr10.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error ${errDir}/chr11.err \
	--output ${outDir}/chr11.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr11_training_set.tsv ${dstDir}/chr11.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error ${errDir}/chr12.err \
	--output ${outDir}/chr12.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr12_training_set.tsv ${dstDir}/chr12.tsv &
wait

srun -n 1 -c 1 --mem=3G \
	--error ${errDir}/chr13.err \
	--output ${outDir}/chr13.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr13_training_set.tsv ${dstDir}/chr13.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error ${errDir}/chr14.err \
	--output ${outDir}/chr14.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr14_training_set.tsv ${dstDir}/chr14.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error ${errDir}/chr15.err \
	--output ${outDir}/chr15.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr15_training_set.tsv ${dstDir}/chr15.tsv &
sleep 1m
srun -n 1 -c 1 --mem=12G \
	--error ${errDir}/chr16.err \
	--output ${outDir}/chr16.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr16_training_set.tsv ${dstDir}/chr16.tsv &
sleep 1m
srun -n 1 -c 1 --mem=15G \
	--error ${errDir}/chr17.err \
	--output ${outDir}/chr17.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr17_training_set.tsv ${dstDir}/chr17.tsv &
wait

srun -n 1 -c 1 --mem=10G \
	--error ${errDir}/chr18.err \
	--output ${outDir}/chr18.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr18_training_set.tsv ${dstDir}/chr18.tsv &
sleep 1m
srun -n 1 -c 1 --mem=15G \
	--error ${errDir}/chr19.err \
	--output ${outDir}/chr19.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr19_training_set.tsv ${dstDir}/chr19.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error ${errDir}/chr20.err \
	--output ${outDir}/chr20.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr20_training_set.tsv ${dstDir}/chr20.tsv &
sleep 1m
srun -n 1 -c 1 --mem=5G \
	--error ${errDir}/chr21.err \
	--output ${outDir}/chr21.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr21_training_set.tsv ${dstDir}/chr21.tsv &
sleep 1m
srun -n 1 -c 1 --mem=10G \
	--error ${errDir}/chr22.err \
	--output ${outDir}/chr22.out \
	Rscript ${exeDir}/${scpName} ${dstDir}/chr22_training_set.tsv ${dstDir}/chr22.tsv &
wait


head -1 ${dstDir}/chr1.tsv > ${dstDir}/trainig_set.tsv
for fl in ${dstDir}/chr{1..22}.tsv; do
	sed -n '2,$p' ${fl} >> ${dstDir}/trainig_set.tsv
done
