#!/bin/bash
# set -o errexit
# XXX (zhzhang2015@sina.com): Create a file of flanks of gene
#
#python gtf2bed.py > Homo_sapiens.GRCh37.75_autosome.bed
#
loaded=$(module list 2>&1 | grep -ic bedtools)
if [ ${loaded} -eq 0 ]; then module load BEDTools; fi
#
#awk '/_gene\t/ {if (($3-$2>1500) &&($1==1)) {print $0}}' \
#        Homo_sapiens.GRCh37.75_autosome.bed \
#        > Homo_sapiens.GRCh37.75_chr1.bed
#
#bedtools sort  -i Homo_sapiens.GRCh37.75_chr1.bed \
#        > Homo_sapiens.GRCh37.75_chr1_sorted.bed
#
bedtools flank \
        -g genome -b 300 -s \
        -i Homo_sapiens.GRCh37.75_chr1_sorted.bed \
        > Homo_sapiens.GRCh37.75_chr1_flank_b500.bed

loaded=$(module list 2>&1 | grep -ic tabix)
if [ ${loaded} -eq 0 ]; then module load tabix; fi

fileName=Homo_sapiens.GRCh37.75_chr1_flank_b500.bed 
vcfFile=chr1_candidates.dose.vcf.gz
counter=1
for x in $( awk '{print $1" "$2" "$3" "$4}' ${fileName} );do
        if [ ${counter} -eq 1 ]; then 
                read -r chr <<< ${x}
        elif [ ${counter} -eq 2 ]; then 
                read -r start <<<  ${x}
        elif [ ${counter} -eq 3 ]; then
                read -r stop <<<  ${x}
        elif [ ${counter} -eq 4 ]; then 
                read -r name <<< ${x}
                n=$(tabix ${vcfFile} ${chr}:${start}-${stop} | wc -l)
                [ ${n} == 0 ] && echo ${chr}:${start}-${stop} ${name}
                counter=1
                continue
        fi
        counter=$[ ${counter} + 1 ]
done
