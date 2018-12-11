#!/bin/bash
# rsync before doing subtraction
# PATH

set -o errexit
set -o errtrace

module load cluster-utils

project_dir=/groups/umcg-bios/tmp03/users/umcg-zzhang/projects/ASEpredictor
# scripts_dir=${project_dir}/scripts
genotype_dir=${project_dir}/genotypes
candidates_file=${project_dir}/candidates_for_training.txt
genotypes_candidates_dir=${project_dir}/genotypes_candidates

imputation_dir=/groups/umcg-bios/prm02/projects/HRC_imputation
RS=RS
PAN=PAN
CODAM=CODAM
LLS_660Q=LLS/660Q
NTR_GoNL=NTR/GoNL
NTR_Affy6=NTR/Affy6
LLS_OmniExpr=LLS/OmniExpr

sub_dirs="${CODAM} ${LLS_660Q} ${LLS_OmniExpr} ${NTR_Affy6} ${NTR_GoNL} ${PAN} ${RS}"

extra_path=results/unzipped
basic_columns="1-9"

for subDir in ${sub_dirs}; do 
    mkdir -p ${genotype_dir}/${subDir}

    for chr in $(seq 1 22); do
        vcf_file=chr${chr}.dose.vcf.gz
        if [ ! -e ${genotype_dir}/${subDir}/${vcf_file} ]; then
            echo "rsync ${vcf_file} from ${imputation_dir}/${subDir}/${extra_path}"
            rsync ${imputation_dir}/${subDir}/${extra_path}/${vcf_file}* ${genotype_dir}/${subDir}
        fi

        echo  "========= ${genotype_dir}/${subDir} ========="
        echo -e ""
        
        duration="10:00:00"
        job_name=${subDir//\//_}_chr${chr}
        output=${subDir//\//_}_chr${chr}-%j.log
        task_no=1
        cpus_per_task=1
        mem_per_cpu=3G

        if [ ! -e ${subDir//\//_}_chr${chr}_getGenotype.sh ]; then 
            echo \
"#!/bin/bash
#SBATCH --time=${duration}
#SBATCH --job-name=${job_name}
#SBATCH --output=${output}
#SBATCH --tasks=${task_no}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --mem-per-cpu=${mem_per_cpu}

set -o errexit
set -o errtrace

# Load and check modules
# loaded=\$(module list tabix 2>&1 | grep -c 'None found')
# if [ \${loaded} -eq 1 ];then module load tabix; fi
module load tabix
module list

cd ${genotype_dir}/${subDir} && pwd

if [ ! -e ${vcf_file} ];then
    echo \"ERR: ${vcf_file} doesn't exist.\" >&2 && exit -1
fi

all_sample_names=\$(tabix -H ${vcf_file} | sed -n '/#CHROM/p' | cut -d$'\\t' -f10- )
declare -A sampleName2Index
counter=10
for ID in \${all_sample_names[@]}; do
    sampleName2Index[\${ID}]=\${counter}
    counter=\$[ \${counter} + 1 ]
done
counter=10

candidate_columns=${basic_columns}
for sample_name in \$(awk '{print \$1}' ${candidates_file}); do
    candidate_column=\${sampleName2Index[\${sample_name}]}
    if [ ! -z \${candidate_column} ]; then
        candidate_columns=\${candidate_columns}","\${candidate_column}
    fi
done

echo  \"********* ${vcf_file} *********\"
output_vcf_file=${vcf_file/.dose.vcf.gz/_candidates.dose.vcf.gz}
mkdir -p ${genotypes_candidates_dir}/${subDir}
output_file=${genotypes_candidates_dir}/${subDir}/\${output_vcf_file}
tabix -h -p vcf ${vcf_file} ${chr}: | cut -d$'\\t' -f\${candidate_columns} | bgzip > \${output_file}
tabix \${output_file}
unset sampleName2Index 
echo \"Job was DONE \" " > ${subDir//\//_}_chr${chr}_getGenotype.sh

            sbatch --exclude=umcg-node002 ${subDir//\//_}_chr${chr}_getGenotype.sh
            while [ $(cqueue | grep zzhang | awk '{print $6}' | wc -l) -ge 5 ]; do 
                sleep 1m 
            done
        fi
    done

    while [ $(cqueue | grep zzhang | awk '{print $6}' | wc -l) -gt 0 ]; do 
        sleep 1m 
    done
    rm ${genotype_dir}/${subDir}/*
done

echo "# Failed jobs" > job_failed.log
for log in $(ls *.log); do
    counter=$(grep -c "Job was DONE" ${log})
    if [ ${counter} -ne 1 ]; then
        echo ${log} >> job_failed.log
    fi
done
