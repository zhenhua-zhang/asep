#!/bin/bash
#SBATCH --time=2:0:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --job-name=biosGavinOverlap.v0.0.1
#SBATCH --output=%j-%u.log

# Directories already existed
pjDir=/home/umcg-zzhang/umcg-zzhang/projects/ASEpredictor
pjIpDir=${pjDir}/inputs
pjOpDir=${pjDir}/outputs

# Directories to be created


# Variables to control scripts
fCov=10  # The NO. of total counts is greater than ${fCov}


# echo "Step 1..."
echo "Step 1 was done ..."
# Step 1 Merge all recordes by chrommosome

# ipDir=/groups/umcg-bios/tmp03/projects
# ipDir=${ipDir}/genotypes_BIOS_LLDeep_Diagnostics_merged_phasing_noRnaEditing
# ipDir=${ipDir}/results/phasing/readbackedPhasing/allelic_counts_mergedPerSample
# Example of input file
# contig	position	variantID	      refAllele  altAllele	refCount	altCount	totalCount
# 1         101704532	1_101704532_G_T	  G	         T	        13	        10	        23	      

# opDir=${pjIpDir}/allelicCountsPerChr
# mkdir -p ${opDir}
# Example of output file
# contig	position	variantID	      refAllele  altAllele	refCount	altCount	totalCount	sample
# 1         101704532	1_101704532_G_T	  G	         T	        13	        10	        23	        AC1C40ACXX-1-18_BD2D5MACXX-6-18
# header="contig\tposition\tvariantID\trefAllele\taltAllele"
# header=${header}"\trefCount\taltCount\ttotalCount\tsample"
# for x in chr{1..22}; do
# 	opFile=${x}_BiosAllelicCounts.tsv
# 	echo -e ${header} > ${opDir}/${opFile}
# 	awk 'FNR > 1 { split(FILENAME, ARR, "."); print $0"\t"ARR[2] }' \
# 		${ipDir}/${x}/* >> ${opDir}/${opFile} 2>/dev/null
# 	echo -e "${opFile} was done... "
# done




# echo "Step 2.1..."
echo "Step 2.1 was done..."
# Step 2.1 Filtering allelic count records by cov and convert into BED format
# XXX: the position of BED records are 0-based


ipDir=${opDir}
# ipFile=
# Example of input file
# contig	position	variantID	      refAllele  altAllele	refCount	altCount	totalCount	sample
# 1         101704532	1_101704532_G_T	  G	         T	        13	        10	        23	        AC1C40ACXX-1-18_BD2D5MACXX-6-18

opDir=${pjOpDir}/allelicCountsPerChrCov${fCov}
opFile=allelicCountsPerChrCov${fCov}.bed
# Example of output file

# mkdir -p ${opDir}

# for ipFile in $(ls ${ipDir}); do 
# 	awk -v fCov=${fCov} ' FNR > 1 { if($8 >= fCov) {print $1"\t"$2-1"\t"$2"\t"$3"_"$6"_"$7"_"$9} }'\
# 		${ipDir}/${ipFile}
# done > ${opDir}/${opFile}

biosIpDir=${opDir}
biosIpFile=${opFile}



echo "Step 2.2..."
# echo "Step 2.2 was done"
# Step 2.2 Filtering GAVIN records 

ipDir=/home/umcg-zzhang/umcg-zzhang/projects/ASEpredictor/inputs/GAVIN
ipFile=gavin_r0.5_calibvars.plusbenign.cadd_v1.4annot.gnomad.tsv
# Example of input records
# gene   chr pos      ref alt group      effect               impact cadd CaddChrom CaddPos  CaddRef CaddAlt Type Length AnnoType   Consequence      ConsScore ConsDetail   GC             CpG             motifECount motifEName motifEHIPos motifEScoreChng oAA nAA GeneID FeatureID GeneName CCDS Intron Exon cDNApos relcDNApos CDSpos relCDSpos protPos relProtPos Domain Dst2Splice Dst2SplType minDistTSS minDistTSE SIFTcat SIFTval PolyPhenCat PolyPhenVal priPhCons mamPhCons verPhCons priPhyloP mamPhyloP verPhyloP bStatistic targetScan mirSVR-Score mirSVR-E mirSVR-Aln cHmmTssA cHmmTssAFlnk cHmmTxFlnk cHmmTx cHmmTxWk cHmmEnhG cHmmEnh cHmmZnfRpts cHmmHet cHmmTssBiv cHmmBivFlnk cHmmEnhBiv cHmmReprPC cHmmReprPCWk cHmmQuies GerpRS GerpRSpval GerpN GerpS TFBS TFBSPeaks TFBSPeaksMax tOverlapMotifs motifDist Segway EncH3K27Ac EncH3K4Me1 EncH3K4Me3 EncExp EncNucleo EncOCC EncOCCombPVal EncOCDNasePVal EncOCFairePVal EncOCpolIIPVal EncOCctcfPVal EncOCmycPVal EncOCDNaseSig EncOCFaireSig EncOCpolIISig EncOCctcfSig EncOCmycSig Grantham Dist2Mutation Freq100bp Rare100bp Sngl100bp Freq1000bp Rare1000bp Sngl1000bp Freq10000bp Rare10000bp Sngl10000bp dbscSNV-ada_score dbscSNV-rf_score RawScore PHRED gnomad_AF
# NUP107 12  69107589 G   A   PATHOGENIC splice_donor_variant HIGH   29.3 12        69107589 G       A       SNV  0      Transcript CANONICAL_SPLICE 6         splice_donor 0.370860927152 0.0666666666667 NA          NA NA NA NA NA ENSG00000111581 ENST00000229179 NUP107 CCDS8985.1 11/27 NA NA NA NA NA NA NA NA -1 DONOR 4586 570 NA NA NA NA 0.676 1.000 1.000 0.451 2.431 3.514 492 NA NA NA NA 0.000 0.000 0.000 0.457 0.512 0.000 0.008 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.024 293 1.31483e-43 4.9 4.9 NA NA NA NA NA GE1 3.0 2.0 3.52 33.6431 0.6 NA NA NA NA NA NA NA NA NA NA NA NA NA 72 0 0 7 0 4 55 13 52 569 0.999988247793601 0.934 4.121872 29.3 0.0 

# Example of output records
opDir=${ipDir}
opFile=${ipFile/.tsv/.bed}

awk ' FNR > 1 { printf $2"\t"$3-1"\t"$3"\t"; line=$1; for(i=2; i<=NF; i++){line=line"|"$i;} printf line"\n"; }' \
	${ipDir}/${ipFile} > ${opDir}/${opFile}





echo "Step 3..."
# echo "Step 3 was done..."
# Step 3 Intersection by BEDTools
gavinIpDir=${opDir}
gavinIpFile=${opFile}
opDir=${pjOpDir}/biosGavinOverlapCov${fCov}
opFile=biosGavinOverlapCov${fCov}.bed

mkdir -p ${opDir}

module list
module load BEDTools
module list
bedtools intersect \
	-wa -wb \
	-a ${biosIpDir}/${biosIpFile} \
	-b ${gavinIpDir}/${gavinIpFile} \
	> ${opDir}/${opFile}




echo "Step 4..."
# Step 4 Make annotation based on 
ipDir=${opDir}
ipFile=${opFile}
opDir=${opDir}
opFile=biosGavinOverlapCov${fCov}Anno.tsv


# chr start stop variantID chr start stop variantInfo
awk ' { n=split($4, arrA, "_"); lineA=arrA[1]"\t"arrA[2]"\t"arrA[3]"\t"arrA[4]"\t"arrA[5]"\t"arrA[6]"\t"arrA[7]; for(i=8; i<=n; i++){lineA=lineA"_"arrA[i]} m=split($8, arrB, "|"); lineB=arrB[1]; for (i=2; i<=m; i++){ lineB=lineB"\t"arrB[i]; } printf lineB"\t"lineA"\n"; }' \
${ipDir}/${ipFile} > ${opDir}/${opFile}

echo "Done $(date)"

echo "Step 5 ..."
# Step draw pictures
ipDir=
opDir=


