set -o errexit
set -o errtrace

# bcftools isec -p check_gavin_to_clinvar -n-1 GAVIN/gavin_simple.sorted.vcf.gz ClinVar/clinvar_20180401.vcf.gz
# bcftools view -i 'INFO/CLNSIG !~ "^affects/i" && INFO/CLNSIG !~ "^association/i" && INFO/CLNSIG !~ "^benign/i" && INFO/CLNSIG !~ "^drug_response/i" && INFO/CLNSIG !~ "^likely_benign/i" && INFO/CLNSIG !~ "^likely_pathogenic/i" && INFO/CLNSIG !~ "^not_provided/i" && INFO/CLNSIG !~ "^other/i" && INFO/CLNSIG !~ "^pathogenic/i" && INFO/CLNSIG !~ "^protective/i" && INFO/CLNSIG !~ "^risk_factor/i" && INFO/CLNSIG !~ "^uncertain_significance/i"' clinvar_overlap.vcf.gz | wc

module load tabix
module load BCFtools

# clinvar_header.vcf
# clinvar_overlap.vcf
# clinvar_specific.vcf
# gavin_header.vcf
# gavin_overlap.vcf
# gavin_specific.vcf

# Types of clincal significance in ClinVar
# CSV_clinvar={"Benign" "Likely benign" "Uncertain significance" "Likely pathogenic" "Pathogenic"}
# Benign
# Likely benign
# Uncertain significance
# Likely pathogenic
# Pathogenic
# drug response
# association
# risk factor
# protective
# conflicting data from submitters
# other
# not provided

# Types of clinical significance in GAVIN
# CSV_gavin={ BENIGN PATHOGENIC POPULATION }
# PATHOGENIC
# POPULATION
# BENIGN

skip=1
if [[ ${skip} -eq 0 ]]; then
	cd ~/Documents/projects/ASEpredictor/inputs && pwd
	bcftools isec -p check_gavin_to_clinvar -n=2 \
		GAVIN/gavin_simple.sorted.vcf.gz ClinVar/clinvar_20180401.vcf.gz
	cd check_gavin_to_clinvar && pwd
	mv 0000.vcf gavin_overlap.vcf
	mv 0001.vcf clinvar_overlap.vcf

	cd -
	bcftools isec -p check_gavin_to_clinvar -n-1 \
		GAVIN/gavin_simple.sorted.vcf.gz ClinVar/clinvar_20180401.vcf.gz
	cd check_gavin_to_clinvar && pwd
	mv 0000.vcf gavin_specific.vcf
	mv 0001.vcf clinvar_specific.vcf
fi


skip=1
if [[ ${skip} -eq 0 ]]; then
	cd ~/Documents/projects/ASEpredictor/inputs/check_gavin_to_clinvar && pwd
	for x in clinvar gavin; do
		for y in overlap specific; do
			cat  ${x}_${y}.vcf | bgzip > ${x}_${y}.vcf.gz
			tabix -f ${x}_${y}.vcf.gz

			if [[ ${x} =~ "gavin" ]]; then
				for z in pathogenic benign population; do
						bcftools view \
							-i 'INFO/CLNSIG = "'${z^^}'"' ${x}_${y}.vcf.gz \
							| bgzip > ${x}_${y}_${z}.vcf.gz
					tabix -f ${x}_${y}_${z}.vcf.gz
				done
			else
				for z in pathogenic benign likely_benign likely_pathogenic uncertain_significance not_provided drug_response other protective risk_factor affects association conflicting; do
					bcftools view \
						-i 'INFO/CLNSIG ~ "^'${z}'/i"' ${x}_${y}.vcf.gz \
						| bgzip > ${x}_${y}_${z}.vcf.gz
					tabix -f ${x}_${y}_${z}.vcf.gz
				done
				z="coma"
				bcftools view \
					-i 'INFO/CLNSIG = "."' ${x}_${y}.vcf.gz \
					| bgzip > ${x}_${y}_${z}.vcf.gz
				tabix -f ${x}_${y}_${z}.vcf.gz
			fi

			done
		done
fi


skip=1
if [[ ${skip} -eq 0 ]]; then
	cd ~/Documents/projects/ASEpredictor/inputs/check_gavin_to_clinvar && pwd
	for x in $(ls *vcf.gz); do
		amount=$(bcftools view -H ${x} | wc -l)
		echo "# of records in ${x}: ${amount}"
	done
fi

# clinvar_overlap_affects.vcf.gz: 1
# clinvar_overlap_association.vcf.gz: 9
# clinvar_overlap_benign.vcf.gz: 36616
# clinvar_overlap_coma.vcf.gz: 198
# clinvar_overlap_conflicting.vcf.gz: 4075
# clinvar_overlap_drug_response.vcf.gz: 27
# clinvar_overlap_likely_benign.vcf.gz: 58587
# clinvar_overlap_likely_pathogenic.vcf.gz: 12487
# clinvar_overlap_not_provided.vcf.gz: 1024
# clinvar_overlap_other.vcf.gz: 78
# clinvar_overlap_pathogenic.vcf.gz: 34797
# clinvar_overlap_protective.vcf.gz: 2
# clinvar_overlap_risk_factor.vcf.gz: 25
# clinvar_overlap_uncertain_significance.vcf.gz: 7382
# clinvar_overlap.vcf.gz: 155308

# clinvar_specific_affects.vcf.gz: 104
# clinvar_specific_association.vcf.gz: 138
# clinvar_specific_benign.vcf.gz: 2823
# clinvar_specific_coma.vcf.gz: 1190
# clinvar_specific_conflicting.vcf.gz: 12485
# clinvar_specific_drug_response.vcf.gz: 268
# clinvar_specific_likely_benign.vcf.gz: 3300
# clinvar_specific_likely_pathogenic.vcf.gz: 4733
# clinvar_specific_not_provided.vcf.gz: 9262
# clinvar_specific_other.vcf.gz: 1697
# clinvar_specific_pathogenic.vcf.gz: 20733
# clinvar_specific_protective.vcf.gz: 34
# clinvar_specific_risk_factor.vcf.gz: 390
# clinvar_specific_uncertain_significance.vcf.gz: 142006
# clinvar_specific.vcf.gz: 199163

# gavin_overlap_benign.vcf.gz: 101518
# gavin_overlap_pathogenic.vcf.gz: 48741
# gavin_overlap_population.vcf.gz: 5049
# gavin_overlap_uncertain_significance.vcf.gz: 5049
# gavin_overlap.vcf.gz: 155308

# gavin_specific_benign.vcf.gz: 46672
# gavin_specific_pathogenic.vcf.gz: 10835
# gavin_specific_population.vcf.gz: 262674
# gavin_specific_uncertain_significance.vcf.gz: 262674
# gavin_specific.vcf.gz: 320181


# Merge pathogenic and likely_pathogenic, benign and likely_benign
skip=1
if [[ ${skip} -eq 0 ]]; then
	for x in overlap specific; do
		for y in benign pathogenic; do
			A=clinvar_${x}_${y}.vcf.gz
			B=clinvar_${x}_likely_${y}.vcf.gz
			C=clinvar_${x}_${y^^}.vcf.gz
			bcftools concat ${A} ${B} -a -O z -o ${C}
			tabix -f ${C}

			a_count=$(bcftools view -H ${A} | wc -l) 
			b_count=$(bcftools view -H ${B} | wc -l)
			c_count=$(bcftools view -H ${C} | wc -l)

			if [[ $[ ${a_count} + ${b_count} ] -eq ${c_count} ]]; then
				rm ${A}* ${B}* -fr
			else
				echo ${a_count} ${b_count} ${c_count}
			fi
		done

		D=clinvar_${x}_not_provided.vcf.gz 
		E=clinvar_${x}_drug_response.vcf.gz 
		F=clinvar_${x}_other.vcf.gz 
		G=clinvar_${x}_protective.vcf.gz 
		H=clinvar_${x}_risk_factor.vcf.gz 
		I=clinvar_${x}_association.vcf.gz 
		J=clinvar_${x}_affects.vcf.gz
		K=clinvar_${x}_conflicting.vcf.gz
		M=clinvar_${x}_coma.vcf.gz
		L=clinvar_${x}_MISC.vcf.gz

		D_count=$(bcftools view -H ${D} | wc -l)
		E_count=$(bcftools view -H ${E} | wc -l)
		F_count=$(bcftools view -H ${F} | wc -l)
		G_count=$(bcftools view -H ${G} | wc -l)
		H_count=$(bcftools view -H ${H} | wc -l)
		I_count=$(bcftools view -H ${I} | wc -l)
		J_count=$(bcftools view -H ${J} | wc -l)
		K_count=$(bcftools view -H ${K} | wc -l)
		M_count=$(bcftools view -H ${M} | wc -l)



		bcftools concat ${D} ${E} ${F} ${G} ${H} ${I} ${J} ${K} ${M} -a -O z -o ${L}
		tabix -f ${L}

		DEFGHIJL_count=$[ ${D_count} + ${E_count} + ${F_count} + ${G_count} + ${H_count} + ${I_count} + ${J_count} + ${K_count} + ${M_count} ]
		L_count=$(bcftools view -H ${L} | wc -l)

		if [[ ${DEFGHIJL_count} -eq ${L_count} ]]; then
			rm ${D}* ${E}* ${F}* ${G}* ${H}* ${I}* ${J}* ${K}* ${M}* -fr
		else
			echo ${DEFGHIJL_count} ${L_count}
		fi
	done
fi


skip=1
if [[ ${skip} -eq 0 ]]; then
	for x in benign pathogenic; do
		echo "clinvar_overlap_${x^^}.vcf.gz" $(bcftools view -H clinvar_overlap_${x^^}.vcf.gz | wc -l)
		echo "gavin_overlap_${x}.vcf.gz" $(bcftools view -H gavin_overlap_${x}.vcf.gz | wc -l)


		bcftools isec -p . -n=2 clinvar_overlap_${x^^}.vcf.gz gavin_overlap_${x}.vcf.gz

		cat 0000.vcf | bgzip > clinvar_overlap_${x^^}_consist.vcf.gz
		tabix -f clinvar_overlap_${x^^}_consist.vcf.gz
		echo "clinvar_overlap_${x^^}_consist.vcf.gz" $(bcftools view -H clinvar_overlap_${x^^}_consist.vcf.gz | wc -l)

		cat 0001.vcf | bgzip > gavin_overlap_${x}_consist.vcf.gz
		tabix -f gavin_overlap_${x}_consist.vcf.gz
		echo "gavin_overlap_${x}_consist.vcf.gz" $(bcftools view -H gavin_overlap_${x}_consist.vcf.gz | wc -l)


		bcftools isec -p . -n-1 clinvar_overlap_${x^^}.vcf.gz gavin_overlap_${x}.vcf.gz

		cat 0000.vcf | bgzip > clinvar_overlap_${x^^}_unconsist.vcf.gz
		tabix -f clinvar_overlap_${x^^}_unconsist.vcf.gz
		echo "clinvar_overlap_${x^^}_unconsist.vcf.gz" $(bcftools view -H clinvar_overlap_${x^^}_unconsist.vcf.gz | wc -l)

		cat 0001.vcf | bgzip > gavin_overlap_${x}_unconsist.vcf.gz
		tabix -f gavin_overlap_${x}_unconsist.vcf.gz
		echo "gavin_overlap_${x}_unconsist.vcf.gz" $(bcftools view -H gavin_overlap_${x}_unconsist.vcf.gz | wc -l)
	done

	echo "Clean up..."
	rm 000* sites.txt README.txt -fr

# clinvar_overlap_BENIGN.vcf.gz 95203
# gavin_overlap_benign.vcf.gz 101518
# clinvar_overlap_BENIGN_consist.vcf.gz 94618
# gavin_overlap_benign_consist.vcf.gz 94618
# clinvar_overlap_BENIGN_unconsist.vcf.gz 585
# gavin_overlap_benign_unconsist.vcf.gz 6900

# clinvar_overlap_PATHOGENIC.vcf.gz 47284
# gavin_overlap_pathogenic.vcf.gz 48741
# clinvar_overlap_PATHOGENIC_consist.vcf.gz 47245
# gavin_overlap_pathogenic_consist.vcf.gz 47245
# clinvar_overlap_PATHOGENIC_unconsist.vcf.gz 39
# gavin_overlap_pathogenic_unconsist.vcf.gz 1496
fi
