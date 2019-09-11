#/bin/bash

# FIXME: Look out when copy-paste the context into terminal directly, tabs-will
#		be translated into request of `completion`

# /home/umcg-zzhang/Documents/projects/ASEPrediction/training/inputs/haplotypicCountsPerChr
# Merge 把按小区域分开的单个文件合并到同一个按染色体分开文件中
count=1; 
for x in $(ls); do 
	(head -1 ${x}/$(ls ${x} | head -1); cat ${x}/* | grep -v contig ) > ${x}_haplotypicReadsCounts.tsv &
	[ $[ ${count} % 5 ] -eq 0 ] && wait
	count=$[ ${count} + 1 ]
done
wait

# Substract 提取部分列内容, 未考虑coverage
# contig	start	stop	variants_aCounts_bCounts_bam
count=1
for x in $(ls *.tsv.gz); do
zcat ${x} | awk -F "\t" '{n=split($4, arr, ","); for (i in arr){split(arr[i], arr1, "_"); print arr1[1]"\t"arr1[2]-1"\t"arr1[2]"\t"arr[i]"_"$10"_"$11"_"$16}}' > ${x/.tsv.gz/.bed} &
[ $[ ${count} % 5 ] -eq 0 ] && wait
count=$[ ${count} + 1 ]
done
wait

# Merge all .bed into one
(echo -e "contig\tstart\tstop\tchr_pos_ha_hb_ac_bc_indv"; cat * | grep -v contig) | gzip > /home/umcg-zzhang/Documents/projects/ASEPrediction/training/outputs/haplotypicCountsPerChr/haplotypicCountsPerChr.bed.gz

# trasform .bed to .vcf
awk 'NR>2 { split($4, arr, "_"); if(arr[6] >= 5 && arr[7] >=5) { print arr[1]"\t"arr[2]"\t.\t"arr[3]"\t"arr[4]"\t.\t.\t.\t." } }' haplotypicCountsPerChr.bed > haplotypicCountsPerChr5_5.vcf
sort -n -k1,1 -k2,2 haplotypicCountsPerChr.vcf | uniq > haplotypicCountsPerChrCov5v5_sort_uniq.vcf

