#!/usr/bin/env Rscript
rm(list = ls())

library("dplyr")
library("foreach")

ppaste <- function(...){ paste(..., sep = "/") }

bnm.tst <- function(succ_vec, fail_vec){
	foreach(s=succ_vec, f=fail_vec) %dopar% {
		binom.test(c(s, f), p = 0.05)$p.value
	}
}

trans_into_bin <- function(rtb, cr, pv = 0.01){
    gp <- rtb %>%
		filter(refCountsBios >= 5 & altCountsBios >= 5 & refCountsBios != altCountsBios) %>%
        mutate(
			binom_p = bnm.tst(refCountsBios, altCountsBios),
			log2FC = log2(altCountsBios / refCountsBios)
        ) %>%
        group_by(chr, pos, ref, alt) %>%
        mutate(
			overallMean = sum(altCountsBios) / sum(refCountsBios),
			FDRPerVariant = p.adjust(binom_p),
			varInsideChi2Pval = chisq.test(refCountsBios, altCountsBios)
        ) %>%
        mutate(
            var = ifelse(length(log2FC) <= 1, 0, var(log2FC)),
            mean = mean(log2FC),
            gp_size = length(log2FC),
            p_value = ifelse(
                length(log2FC) <= 1,
                ifelse(FDRPerVariant <= 0.01 || abs(log2FC) >= 1, 0, 1),
                ifelse(
                    max(log2FC) == min(log2FC),
                    ifelse(abs(max(log2FC)) > 1, 0, 1),
                    t.test(log2FC, mu = overallMean)$p.value
                )
            )
        ) %>%
        ungroup() %>%
        mutate(ASE = ifelse(p_value <= pv, ifelse(mean < 0, -1, 1), 0)) %>%
        select(rn) %>%
        arrange(chr, pos, ref, alt) %>%
        distinct() %>%
        as.data.frame()

    return(gp)
}

hmd <- path.expand("~")
pjd <- ppaste(hmd, "Documents", "projects", "ASEpredictor")

ipd <- ppaste(pjd, "outputs", "biosGavinOverlapCov10")
ipf <- ppaste(ipd, "biosGavinOverlapCov10Anno.tsv")
rtb <- read.csv(ipf, header = TRUE, sep = "\t")
rn <- c(colnames(rtb)[1:118], "var", "mean", "p_value", "gp_size", "ASE")
odf <- trans_into_bin(rtb, rn)

opd <- ppaste(pjd, "outputs", "biosGavinOverlapCov10")
opf <- ppaste(opd, "biosGavinOlCv10AntUfltCstLog2FCBin_.tsv")
write.table(odf, file = opf, quote = FALSE, sep = "\t", row.names = FALSE)
