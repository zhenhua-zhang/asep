#!/bin/bash
#SBATCH --time=2:59:0
#SBATCH --output=%j-%u-merge_cadd_allelicCounts.log
#SBATCH --job-name=merge_cadd_allelicCounts
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=5G

#10002723	1	C	T	SNV	0	Intergenic	UPSTREAM	1	upstream	0.456953642384	0.0666666666667	NA	NA	NA	NA	NA	NA	ENSG00000173614	ENST00000377205	NMNAT1	CCDS108.1	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	104	7310	NA	NA	NA	NA	0.026	0.000	0.000	-0.180	0.106	0.222	575	NA	NA	NA	NA	0.598	0.378	0.000	0.000	0.000	0.000	0.016	0.000	0.000	0.008	0.000	0.000	0.000	0.000	0.000	NA	NA	4.36	1.02	8	10	82.9798	NA	NA	GS	94.8	19.0	234.84	360.562	7.5	3.0	14.91	13.5	4.95	16.0	16.0	5.68	0.3717	0.0382	0.6545	1.6418	0.1649	NA	60	0	0	4	0	9	68	11	55	593	NA	NA	0.528742	9.645	1	10002722	1_10002723_C_T_10_13_AD2CJPACXX-1-6

check_dir() {
    [ -d $1 ] && echo "Find $1" || exit 1
}

WKHM=/home/umcg-zzhang/Documents/projects/ASEPrediction/training/outputs
srcDir=${WKHM}/allelicCountsPerChrCov10
dstDir=${WKHM}/annotCadd
errDir=${dstDir}/errs
mkdir ${errDir}

check_dir ${srcDir}
check_dir ${dstDir}
check_dir ${errDir}

for x in {1..22}; do
  srun --mem=5G --ntasks=1 --cpus-per-task=1 --error=${errDir}/chr${x}_training_set.err \
    sed -n "/^${x}\t/p" ${srcDir}/allelicCountsPerChrCov10_noheader.tsv | sort -k2,2n > ${dstDir}/chr${x}_file1.tsv \
      && sed -n "/^${x}\t/p" ${srcDir}/allelicCountsPerChrCov10.bed | sort -k2,2n > ${dstDir}/chr${x}_file2.tsv \
      && echo -e "Chrom\tPos\tRef\tAlt\tType\tLength\tAnnoType\tConsequence\tConsScore\tConsDetail\tGC\tCpG\tmotifECount\tmotifEName\tmotifEHIPos\tmotifEScoreChng\toAA\tnAA\tGeneID\tFeatureID\tGeneName\tCCDS\tIntron\tExon\tcDNApos\trelcDNApos\tCDSpos\trelCDSpos\tprotPos\trelProtPos\tDomain\tDst2Splice\tDst2SplType\tminDistTSS\tminDistTSE\tSIFTcat\tSIFTval\tPolyPhenCat\tPolyPhenVal\tpriPhCons\tmamPhCons\tverPhCons\tpriPhyloP\tmamPhyloP\tverPhyloP\tbStatistic\ttargetScan\tmirSVR-Score\tmirSVR-E\tmirSVR-Aln\tcHmmTssA\tcHmmTssAFlnk\tcHmmTxFlnk\tcHmmTx\tcHmmTxWk\tcHmmEnhG\tcHmmEnh\tcHmmZnfRpts\tcHmmHet\tcHmmTssBiv\tcHmmBivFlnk\tcHmmEnhBiv\tcHmmReprPC\tcHmmReprPCWk\tcHmmQuies\tGerpRS\tGerpRSpval\tGerpN\tGerpS\tTFBS\tTFBSPeaks\tTFBSPeaksMax\ttOverlapMotifs\tmotifDist\tSegway\tEncH3K27Ac\tEncH3K4Me1\tEncH3K4Me3\tEncExp\tEncNucleo\tEncOCC\tEncOCCombPVal\tEncOCDNasePVal\tEncOCFairePVal\tEncOCpolIIPVal\tEncOCctcfPVal\tEncOCmycPVal\tEncOCDNaseSig\tEncOCFaireSig\tEncOCpolIISig\tEncOCctcfSig\tEncOCmycSig\tGrantham\tDist2Mutation\tFreq100bp\tRare100bp\tSngl100bp\tFreq1000bp\tRare1000bp\tSngl1000bp\tFreq10000bp\tRare10000bp\tSngl10000bp\tdbscSNV-ada_score\tdbscSNV-rf_score\tRawScore\tPHRED\tchrBios\tposBios\trefAlleleBios\taltAlleleBios\trefCountsBios\taltCountsBios\tsampleBios" > ${dstDir}/chr${x}_training_set.tsv \
      && join -1 2 -2 3 -t $'\t' ${dstDir}/chr${x}_file1.tsv ${dstDir}/chr${x}_file2.tsv | awk -F "\t" '{nf=split($NF, arr, "_"); if(arr[1]==$2 && arr[3]==$3 && arr[4]==$4 && arr[5]!=arr[6]){printf $2"\t"$1"\t"; for(i=3; i<NF-2;i++){printf $i"\t"}; for(j=1;j<=6;j++){printf arr[j]"\t"}; printf arr[7]; for(k=8;k<=nf;k++){printf "_"arr[k]}; printf "\n"}}' >> ${dstDir}/chr${x}_training_set.tsv \
      && rm ${dstDir}/chr${x}_file1.tsv ${dstDir}/chr${x}_file2.tsv -fr &
  echo "Create slave for chr${x}..."
  [ $[ ${x} % 5 ] -eq 0 ] && echo "Waiting..." && wait
done

echo "Waiting..." && wait

[ $? -eq 0 ] && echo "Success" || echo "Non-zero exit..."
