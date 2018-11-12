
#!/bin/bash
################################################################################
#   File Name  : addClinVarGroupToCADD.sh
##  Author     : zhzhang
### E-mail     : zhzhang2015@sina.com
### Created on : Sun 11 Nov 2018 06:33:42 PM CET
##  Version    : unknown
#   License    : MIT
################################################################################
#SBATCH --time          = 1:0:0
#SBATCH --output        = %j-%u-addClinVarGroupToCADD.sh.log
#SBATCH --job-name      = addClinVarGroupToCADD.sh
#SBATCH --ntasks        = 1
#SBATCH --cpus_per_task = 1
#SBATCH --mem_per_cpu   = 1G

# Load modules
module list
module load BCFtools
module list

# Arrange working directories
pjDir='/home/umcg-zzhang/Documents/projects/ASEpredictor'
pjIpDir=${pjDir}/inputs
pjOpDir=${pjDir}/outputs

caddDir=${pjOpDir}/biosCADD
caddIpDir=${caddDir}
caddIpFile=${caddDir}/biosCDD.tvs

cvDir='/apps/data/ClinVar'
cvIpDir=${cvIpDir}
cvIpFile=${cvIpDir}/clinvar_20180401.vcf.gz

uOpDir=${caddDir}
uOpFile=${uOpFile}/biosCaddClinvar.tvs.tmp

opFm='%CHROM\t%POS\t%ID\t%REF\t%ALT\t%INFO/AF_EXAC\t%INFO/CLNSIG\n'   # Output format
incEx='TYPE="snp"'  # Including sites

bcftools query \
    -f ${opFm} \
    -i ${incEx} \
    -T ${caddIpFile} \
    -o ${uOpFile}
