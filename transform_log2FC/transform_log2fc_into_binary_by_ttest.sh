#!/bin/bash
#
# File Name  : transform_log2fc_into_binary_by_ttest.sh
# Author     : zhzhang
# E-mail     : zhzhang2015@sina.com
# Created on : Thu 24 Jan 2019 03:04:00 PM CET
# Version    : v0.0.1
# License    : MIT
#
#SBATCH --time=9:0:0
#SBATCH --output=%j-%u-transform_log2fc_into_binary_by_ttest.log
#SBATCH --job-name=transform_log2fc_into_binary_by_ttest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

pwd
module load R/3.3.3-foss-2015b
Rscript transform_log2fc_into_binary_by_ttest.R

if [ $? -eq 0 ]; then
	echo "Job was DONE!"
else
	echo "Return non-zero!!!"
fi
