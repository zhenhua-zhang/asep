#!/bin/sh
set -o errexit
set -o errtrace

# Utils
ERRO() {
	echo -e "[ERRO]: $1" >&2 && exit -1
}

WARN() {
	echo -e "[WARN]: $1" >&2 && exit -1
}

INFO() {
	echo -e "[INFO]: $1" >&2
}

check_dir() {
	[ -d $1 ] && INFO "Found dir ${1}" || ERRO "NOT found dir ${1}"
}

check_file() {
	[ -f $1 ] && INFO "Found file ${1}" || ERRO "NOT found file ${1}"
}

echo_help() {
	usage
	echo -e "Help: "
	exit 0
}

module load BCFtools


# trainingset=
# cut -f1,2 ${trainingset} > candidate.snps

vcfpath=/apps/data/gnomAD/release-170228/vcf/genomes/r2.0.2
check_dir ${vcfpath}

cd ${vcfpath}

rgnf=${HOME}/Documents/projects/ASEPrediction/validating/outputs/annotCadd/candidate.snps
check_file ${rgnf}

optd=/groups/umcg-bios/tmp03/users/umcg-zzhang/projects/ASEPrediction/validating/inputs/gnomAD
check_dir ${optd}

count=0
for x in $(ls *vcf.gz); do
    bcftools query -H \
        -i 'TYPE="snp"' \
        -f "%CHROM\t%END\t%REF\t%ALT\t%INFO/AF_AMR\n" ${x} \
        -R ${rgnf} \
        > ${optd}/${x/.vcf.gz/_AF_NFE.tsv} &
    count=$[ ${count} + 1 ]
    [[ $[ ${count} % 7 ] -eq 0 ]] && wait
done

cd ${optd}

# (head -1 $(ls *|head -1); grep -v ^\# $(ls)) > output.tsv
