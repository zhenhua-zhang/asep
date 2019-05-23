#!/bin/bash
# File Name  : transform_expression_abundance_into_binary_sbatch.sh
# Author     : zhzhang
# E-mail     : zhzhang2015@sina.com
# Created on : Thu 28 Feb 2019 02:39:10 PM CET
# Version    : v0.0.1
# License    : MIT

#SBATCH --time=2:59:0
#SBATCH --output=%j-%u-transform_expression_abundance_into_binary_v0.0.2.log
#SBATCH --job-name=transform_expression_abundance_into_binary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=25G

set -o errexit
set -o errtrace

DEBUG=0

OPEN_LOG() {
	exec $1>$2
}

CLOSE_LOG() {
	exec $1>&-
}

ERROR() {
	echo -e "ERROR: $1" >&2 && exit -1
}

INFOR() {
	echo -e "INFOR: $1" >&2
}

check_dir() {
	[ -d $1 ] && INFOR "Found dir ${1}" || ERROR "NOT found dir ${1}"
}

check_file() {
	[ -f $1 ] && INFOR "Found file ${1}" || ERROR "NOT found file ${1}"
}

check_success() {
	cmd=${2:-Last command}
	[ $1 -eq 0 ] && INFOR "${cmd} success!" || ERROR "${cmd} non-zero exit ..."
}

mkdir_p() {
	mkdir -p $@
	check_success $? "mkdir -p $@"
}


module purge
module load R/3.3.3-foss-2015b-bare
module list

WKHM=${HOME}/Documents/projects/ASEPrediction/validating
exeDir=${WKHM}/scripts
dstDir=${WKHM}/outputs/cadd_annotation
scpName=transform_expression_abundance_into_binary_v0.0.2.R

errDir=${dstDir}/errs
outDir=${dstDir}/outs
mkdir_p ${outDir} ${errDir}

check_dir ${exeDir}
check_dir ${dstDir}
check_dir ${errDir}
check_dir ${outDir}
check_file ${exeDir}/${scpName}

start_chr=22
stop_chr=22

for chr in {22..22}; do
	input_file=${dstDir}/chr${chr}_set.tsv
	output_file=${dstDir}/chr${chr}_training_set.tsv
	INFOR "Processing ${input_file} ${output_file}"
	srun -n 1 -c 1 --mem=5G \
		--error ${errDir}/chr${chr}.err \
		--output ${outDir}/chr${chr}.out \
		Rscript ${exeDir}/${scpName} ${input_file} ${output_file} &

	INFOR "Created slave worker for chr${chr}..."
	sleep 1m
	[ $[ ${chr} % 4 ] -eq 0 ] && INFOR "Waiting..." && wait

	[ ${DEBUG} -eq 1 ] && break
done

INFOR "Waiting for the rest workers ..." && wait
check_success $? "for loop"

head -1 ${dstDir}/chr1_training_set.tsv | tr "." "-" > ${dstDir}/training_set.tsv
for fl in ${dstDir}/chr{1..22}_training_set.tsv; do
	sed -n '2,$p' ${fl} >> ${dstDir}/training_set.tsv
  [ ${DEBUG} -eq 1 ] && break
done
check_success $?

rm ${dstDir}/chr{1..22}_set.tsv ${dstDir}/chr{1..22}_training_set.tsv -fr
check_success $?
