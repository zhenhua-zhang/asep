#!/bin/bash
#
# File Name  : predictor_of_ase.sh
# Author     : zhzhang
# E-mail     : zhzhang2015@sina.com
# Created on : Tue 15 Jan 2019 04:13:01 PM CET
# Version    : v0.0.1
# License    : MIT
#
#SBATCH --time=3:0:0
#SBATCH --output=%j-%u-predictor_of_ase.log
#SBATCH --job-name=predictor_of_ase
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=5G

module list 
pwd

ASED=Documents/projects/ASEpredictor
source ${HOME}/${ASED}/_asep_env/bin/activate
python ${HOME}/${ASED}/outputs/predictor/predictor_of_ase.py

if [ $? -eq 0 ]; then
	echo "Job was done!"
else
	echo "Job returned without 0"
fi
