#!/bin/bash

TRAINDIR=${HOME}/Documents/projects/ASEPrediction/training
INPUT=${TRAINDIR}/inputs/GAVIN/test.tsv
MODEL=${TRAINDIR}/scripts/2019_Apr_05_09_31_52_brfc_ic5_ini50_oc5_mgs5/2019_Apr_05_09_31_52_brfc_ic5_ini50_oc5_mgs5_object.pkl

source "${HOME}"/Documents/git/asep/_asep_env/bin/activate
python "${HOME}"/Documents/git/asep/predictor/asep.py \
	predict \
	-i "${INPUT}" \
	-m "${MODEL}" \
	-o "./"
