#!/bin/bash

module load BCFtools
count=1

for x in $(ls /apps/data/gnomAD/release-170228/vcf/genomes/r2.0.2/*gz); do
	flnm=${x##*/}

	echo "Start new job NO. ${count}"
	bcftools query \
		-R snpPos_genic.tsv \
		-i 'TYPE="snp"' \
		-f "%CHROM\t%POS\t%REF\t%ALT[0]\t%INFO/AF\n" ${x} \
		> ../../inputs/gnomAD/${flnm/.vcf.gz/.snp.AF.tsv} &

	[ $[ ${count} % 8 ] -eq 0 ] && wait || echo "Start another round ..."
	count=$[ ${count} + 1 ]
done

wait
