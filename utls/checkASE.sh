#!/bin/bash -f

# set -o xtrace
set -o errexit
set -o errtrace
set -o noclobber

ERRO() {
	echo -e "[ERRO]: $1" >&2 && exit -1
}

WARN() {
	echo -e "[WARN]: $1" >&2
}

INFO() {
	echo -e "[INFO]: $1"
}

check_file() {
	[ -f $1 -a $1 ] && INFO "Found file ${1}" || ERRO "NOT found file ${1}"
}

echo_version() {
	cat << EOF

$(basename $0), Version ${SCRIPT_VERSOIN:=UNKNOWN}
EOF
}

echo_usage() {
	cat <<EOF

Usage: ./$(basename $0) -i/--input INPUT-PATH -o/--output OUTPUT-PATH [options]
EOF
}

echo_help() {
	echo_version
	echo_usage
	cat <<EOF

Help:
  -i, --input    Required. Action: store_value
    The path to the input-file.
  -r, --realAseCol    Optional. Action: store_value
    Column indicating real ASE effects. Default: 88
  -p, --predictedAseCol    Optional. Action: store_value
    Column indicating predicted ASE effects. Default: 102
  -h, --help    Optional. Action: print_info
    Print this help context and exit.

More information please contact Zhenhua Zhang <zhenhua.zhang217@gmail.com>
EOF
	exit 0
}

opt=$(getopt -o "-i:r::p::h" -l "input:,realAseCol::,predictedAseCol::,help" -- $@)
eval set -- ${opt}

while true; do
	case $1 in
		-i|--input) shift && tgsv=$1 ;;
		-r|--realAseCol) shift && realAseCol=$1 ;;
		-p|--predictedAseCol) shift && predictedAseCol=$1 ;;
		-h|--help) echo_help && exit 0 ;;
		--) shift && break;;
	esac
	shift
done

check_file ${tgsv}

awk -v realAseCol=${realAseCol:=88} \
	-v predictedAseCol=${predictedAseCol:=102} \
	'BEGIN { ASE=0; ASEWrong=0; ASECorrect=0; NonASE=0; NonASEWrong=0; NonASECorrect=0; } FNR>1 { if($realAseCol==1) { ASE=ASE+1; if($predictedAseCol>=0.5) { ASECorrect=ASECorrect+1; } else { ASEWrong=ASEWrong+1; } } else { NonASE=NonASE+1; if($predictedAseCol>=0.5) { NonASEWrong=NonASEWrong+1; } else { NonASECorrect=NonASECorrect+1; } } } END { print "ASE:", ASE; print "ASEWrong:", ASEWrong; print "ASECorrect:", ASECorrect; print "NonASE:", NonASE; print "NonASEWrong:", NonASEWrong; print "NonASECorrect:", NonASECorrect; }' ${tgsv}

# validation_trainingset_withpLIScore_withGnomADAF_exon_FDR0.05_pred.tsv
