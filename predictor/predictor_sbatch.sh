#!/bin/bash
#
# File Name  : predictor_sbatch.sh
# Author     : zhzhang
# E-mail     : zhzhang2015@sina.com
# Created on : Fri 08 Mar 2019 01:38:10 PM CET
# Version    : v0.0.1
# License    : MIT
#
#SBATCH --time=7:59:0
#SBATCH --output=%j-%u-predictor_sbatch.log
#SBATCH --job-name=predictor_sbatch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=10G

#usage: asep.py [-h] [-V] [--run-flag RUN_FLAG] {train,validate,predict} ...
#
#positional arguments:
#  {train,validate,predict}
#    train               Train a model
#    validate            Validate the model.
#    predict             Predict new dataset by the trained model
#
#optional arguments:
#  -h, --help            show this help message and exit
#
#Global:
#  -V                    Verbose level
#  --run-flag RUN_FLAG   Flags for current run

######## train ########
#usage: asep.py train [-h] -i INPUT_FILE [-f FIRST_K_ROWS] [-m MASK_AS]
#                     [-M MASK_OUT] [--min-group-size MIN_GROUP_SIZE]
#                     [--max-group-size MAX_GROUP_SIZE]
#                     [--drop-cols DROP_COLS [DROP_COLS ...]]
#                     [--response-col REPONSE_COL] [--test-size TEST_SIZE]
#                     [-c CONFIG_FILE] [--classifier {abc,gbc,rfc,brfc}]
#                     [--resampling] [--nested-cv] [--inner-cvs INNER_CVS]
#                     [--inner-n-jobs INNER_N_JOBS]
#                     [--inner-n-iters INNER_N_ITERS] [--outer-cvs OUTER_CVS]
#                     [--outer-n-jobs OUTER_N_JOBS]
#                     [--learning-curve-cvs LC_CVS]
#                     [--learning-curve-n-jobs LC_N_JOBS]
#                     [--learning-curve-space-size LC_SPACE_SIZE]
#                     [-o OUTPUT_DIR] [--with-learning-curve]
#
#optional arguments:
#  -h, --help            show this help message and exit
#
#Input:
#  -i INPUT_FILE, --input-file INPUT_FILE
#                        The path to file of training dataset. Default: None
#
#Filter:
#  -f FIRST_K_ROWS, --first-k-rows FIRST_K_ROWS
#                        Only read first k rows as input from input file.
#                        Default: None
#  -m MASK_AS, --mask-as MASK_AS
#                        Pattern will be kept. Default: None
#  -M MASK_OUT, --mask-out MASK_OUT
#                        Pattern will be masked. Default: None
#  --min-group-size MIN_GROUP_SIZE
#                        The minimum of individuals bearing the same variant
#                        (>= 2). Default: 2
#  --max-group-size MAX_GROUP_SIZE
#                        The maximum number of individuals bearing the same
#                        variant (<= 10,000). Default: None
#  --drop-cols DROP_COLS [DROP_COLS ...]
#                        The columns will be dropped. Seperated by semi-colon
#                        and quote them by ','. if there are more than one
#                        columns. Default: None
#  --response-col REPONSE_COL
#                        The column name of response variable or target
#                        variable. Default: bb_ASE
#
#Configuration:
#  --test-size TEST_SIZE
#                        The proportion of dataset for testing. Default: None
#  -c CONFIG_FILE, --config-file CONFIG_FILE
#                        The path to configuration file, all configuration will
#                        be get from it, and overwrite values from command line
#                        except -i. Default: None
#  --classifier {abc,gbc,rfc,brfc}
#                        Algorithm. Choices: [abc, gbc, rfc, brfc]. Default:
#                        rfc
#  --resampling          Use resampling method or not. Default: False
#  --nested-cv           Use nested cross validation or not. Default: False
#  --inner-cvs INNER_CVS
#                        Fold of cross-validations for RandomizedSearchCV.
#                        Default: 6
#  --inner-n-jobs INNER_N_JOBS
#                        Number of jobs for RandomizedSearchCV, Default: 5
#  --inner-n-iters INNER_N_ITERS
#                        Number of iters for RandomizedSearchCV. Default: 50
#  --outer-cvs OUTER_CVS
#                        Fold of cross-validation for outer_validation
#  --outer-n-jobs OUTER_N_JOBS
#                        Number of jobs for outer_validation
#  --learning-curve-cvs LC_CVS
#                        Number of folds to draw learning curve
#  --learning-curve-n-jobs LC_N_JOBS
#                        Number of jobs to draw learning curves
#  --learning-curve-space-size LC_SPACE_SIZE
#                        Number of splits will be create in learning curve
#
#Output:
#  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
#                        The directory including output files. Default: ./
#  --with-learning-curve
#                        Whether to draw a learning curve. Default: False

######## predict ########
#usage: asep.py predict [-h] -i PREDICT_INPUT_FILE -m MODEL_FILE -o
#                       PREDICT_OUTPUT_DIR
#
#optional arguments:
#  -h, --help            show this help message and exit
#
#Input:
#  -i PREDICT_INPUT_FILE, --predict-input-file PREDICT_INPUT_FILE
#                        New files including case to be predicted
#  -m MODEL_FILE, --model-file MODEL_FILE
#                        Model to be used
#
#Output:
#  -o PREDICT_OUTPUT_DIR, --predict-output-dir PREDICT_OUTPUT_DIR
#                        Output directory for predict input file

######## validate ########
#usage: asep.py validate [-h] -v VALIDATION_FILE
#
#optional arguments:
#  -h, --help            show this help message and exit
#
#Input:
#  -v VALIDATION_FILE, --validation-file VALIDATION_FILE
#                        The path to file of validation dataset

module load Python/3.5.1-foss-2015b
module list
echo "Current directory: $(pwd)"

source /groups/umcg-bios/tmp03/users/umcg-zzhang/projects/ASEPrediction/_asep_env/bin/activate
python /groups/umcg-gcc/tmp03/umcg-zzhang/git/asep/predictor/asep.py \
 	--run-flag rfc_ic6_ini50_oc6_mings5 \
	train \
	--first-k-rows 50000 \
 	--train-input-file /home/umcg-zzhang/Documents/projects/ASEPrediction/training/outputs/annotCadd/training_set.tsv
	--inner-cvs 4 \
 	--inner-n-jobs 5 \
	--inner-n-iters 50 \
	--outer-cvs 6 \
 	--outer-n-jobs 4 \
 	--min-group-size 5 \
 	--classifier rfc

	# --with-learning-curve \
	# --learning-curve-cvs 10 \
	# --learning-curve-n-jobs 5 \

[ $? -eq 0 ] && echo "Job was done" || echo "Exited with non-zero"
