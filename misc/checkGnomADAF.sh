#!/bin/bash
#SBATCH --time=0:10:0
#SBATCH --ntasks=1
#SBATCH --output=%j-%u.log
#SBATCH --job-name=updateGAVINGnomadColumn
#SBATCH --mail-user=zhenhua.zhang217@gmail.com
#SBATCH --mail-type=all
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=10

# Set up timer
SECONDES=0

# Set up error handler rules
set -o errexit
set -o errtrace

# TODO create a log function
echo -e "=== Start ==="

# Job info
echo -e "Current directory: $(pwd)"

# Load modules
module list
module load BCFtools
module list

# Arrange woring dirs
## proecessed GAVIN file
pjDir=${HOME}/projects/ASEpredictor
pjIpDir=${pjDir}/outputs/biosGavinOverlapCov10
pjOpDir=${pjDir}/outputs/biosGavinOverlapCov10
pjIpFile=${pjIpDir}/tmp.bed
pjOpFile=${pjOpDir}/tmp.vcf


## GnomAD files
gDir=/apps/data/gnomAD
gIpDir=${gDir}/release-170228/vcf/genomes/r2.0.2/

bcftools query \
    -R ${pjIpFile} \
    -f '%CHROM\t%POS\t%REF\t%ALT\tAF_raw%INFO/AF_raw;AF=%INFO/AF\n' \
    ${gIpDir}/gnomad.genomes.r2.0.2.sites.chr{1..22}.normalized.vcf.gz \
    > ${pjOpFile}
