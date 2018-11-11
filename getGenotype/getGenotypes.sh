#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=getGenotypes.sh
#SBATCH --output=%u-%j.log
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhenhua.zhang0217@gmail.com


set -o errexit
set -o errtrace

# PATH
project_dir="/groups/umcg-bios/tmp03/users/umcg-zzhang/projects/ASEpredictor"
genotype_dir=${project_dir}"/genotypes"
genotypes_candidates_dir=${project_dir}"/genotypes_candidates"
candidates_file=${project_dir}"/candidates_for_training.txt"
tmp_dir=${project_dir}/tmp
basic_columns="1-9"

# PATH
imputation_dir=/groups/umcg-bios/prm02/projects/HRC_imputations
CODAM=${imputaion_dir}/CODAM/results/unzipped
LLS_660Q=${imputaion_dir}/LLS/660Q/results/unzipped
LLS_OmniExpr=${imputation_dir}/LLS/OmniExpr/results/unzipped
NTR_Affy6=${imputation_dir}/NTR/Affy6/results/unzipped
NTR_GoNL=${imputation_dir}/NTR/GoNL/results/unzipped
PAN=${imputation_dir}/PAN/results/unzipped
RS=${imputation_dir}/RS/results/unzipped

# Load and check modules
loaded=$(module list tabix 2>&1 | grep -c 'None found')
if [ ${loaded} -eq 1 ];then module load tabix; fi
module list


for subDir in $(ls ${genotype_dir}); do
    echo  "========= ${subDir} ========="
    cd ${genotype_dir}"/"${subDir}
    pwd
    declare -A sampleName2Index
    vcf=chr1.dose.vcf.gz
    counter=10
    [ -h ${vcf} ] || (echo "ERR: ${vcf} doesn't exist." >&2 && exit -1)
    original_vcf=$(readlink ${vcf})
    all_sample_names=$(tabix -H ${original_vcf}|sed -n '/#CHROM/p'|cut -d$'\t' -f10-)

    for ID in ${all_sample_names[@]}; do
        sampleName2Index[${ID}]=${counter}
        counter=$[ ${counter} + 1 ]
    done
    counter=10

    candidate_columns=${basic_columns}
    for sample_name in $(awk '{print $1}' ${candidates_file}); do
        candidate_column=${sampleName2Index[${sample_name}]}
        if [ ! -z ${candidate_column} ]; then
            candidate_columns=${candidate_columns}","${candidate_column}
        fi
    done

    # if [ -e ${tmp_dir}/candidates_tmp.txt ]; then rm ${tmp_dir}/tmp; fi
    # echo ${candidate_columns} >> ${tmp_dir}/candidates_tmp.txt
    # if [ -e ${tmp_dir}/tmp ]; then rm ${tmp_dir}/tmp; fi

    for vcf_file in $(ls *gz); do
        echo  "********* ${vcf_file} *********"
        output_vcf_file=${vcf_file/.dose.vcf.gz/_candidates.dose.vcf.gz}
        original_vcf_file=$(readlink ${vcf_file})
        gzip -dc ${original_vcf_file} \
        | cut -d$'\t' -f${candidate_columns} \
        | gzip - \
        > ${genotypes_candidates_dir}/${subDir}/${output_vcf_file}
        # > ${tmp_dir}/tmp
    done

    cd - 2>&1 > /dev/null
    echo $(pwd)

    unset sampleName2Index
done
