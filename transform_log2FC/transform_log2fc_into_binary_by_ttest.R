#!/usr/bin/env Rscript

rm(list=ls())

library('dplyr')

ppaste <- function(...){
    paste(..., sep='/')
}

hmd <- '/home/umcg-zzhang'
pjd <- ppaste(hmd, 'Documents', 'projects', 'ASEpredictor')

ipd <- ppaste(pjd, 'outputs', 'biosGavinOverlapCov10')
ipf <- ppaste(ipd, 'biosGavinOlCv10AntUfltCst.tsv')

opd <- ppaste(pjd, 'outputs', 'biosGavinOverlapCov10')
opf <- ppaste(opd, 'biosGavinOlCv10AntUfltCstLog2FCBin.tsv')

add_binom_log2fc_chist <- function(rtb){
                                        # :prams: rtb (data.frame): input data.frame
    
    df <- rtb %>%
        mutate(
            binom_p=binom.test(c(refAlleleBios, altAlleleBios), p=0.5),
            log2FC=-log2(altAlleleBios/refAlleleBios)
        ) %>%
        group_by(chr, pos, ref, alt) %>%
        mutate(
            FDRPerVariant=p.adjust(binom_p),
            varInsideChi2Pval=chisq.test(refAlleleBios, alrAlleleBios)
        ) %>%
        ungroup() %>%
        as.data.frame()
    
    return(df)
}

trans_into_bin <- function(rtb, cr, pv=0.01){
                                        # :prams: rtb (data.frame): input data.frame
                                        # :prams: cr (vector): slelected rows

    gp <- rtb %>%
        group_by(chr, pos, ref, alt) %>%
        mutate(
            var=ifelse(length(log2FC)<=1, 0, var(log2FC)),
            mean=mean(log2FC), 
            gp_size=length(log2FC),
            p_value=ifelse(
                length(log2FC)<=1, 
                ifelse(FDRPerVariant<=0.01 || abs(log2FC) >= 1, 0, 1), 
                ifelse(
                    max(log2FC)==min(log2FC),
                    ifelse(abs(max(log2FC))>1, 0, 1),
                    t.test(log2FC, mu=0)$p.value
                )
            )
        ) %>%
        ungroup() %>%
        mutate(ASE=ifelse(p_value<=pv, ifelse(mean<0, -1, 1), 0)) %>%
        select(rn) %>%
        arrange(chr, pos, ref, alt) %>%
        distinct() %>%
        as.data.frame()

    return(gp)
}

rtb <- read.csv(ipf, header = TRUE, sep='\t')
rn <- colnames(rtb)[1:117]
rn <- c(rn, 'var', 'mean', 'p_value', 'gp_size', 'ASE')

odf <- trans_into_bin(rtb, rn)
rm(rtb)
write.table(odf, file=opf, quote=FALSE, sep="\t", row.names=FALSE)
