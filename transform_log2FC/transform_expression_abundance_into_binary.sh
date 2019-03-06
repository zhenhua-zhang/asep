#!/bin/bash
#
# File Name  : transform_expression_abundance_into_binary.sh
# Author     : zhzhang
# E-mail     : zhzhang2015@sina.com
# Created on : Thu 24 Jan 2019 03:04:00 PM CET
# Version    : v0.0.1
# License    : MIT
#
#SBATCH --time=9:0:0
#SBATCH --output=%j-%u-transform_expression_abundance_into_binary.log
#SBATCH --job-name=transform_expression_abundance_into_binary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

pwd
module load R/3.3.3-foss-2015b
Rscript transform_expression_abundance_into_binary.R

if [ $? -eq 0 ]; then
	echo "Job was DONE!"
else
	echo "Return non-zero!!!"
fi
