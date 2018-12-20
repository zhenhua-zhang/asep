rm(list=ls())

library('dplyr')
wd <- getwd()

ppaste <- function(...){
  paste(..., sep='/')
}

hmd <- '/home/umcg-zzhang'
pjd <- ppaste(hmd, 'Documents', 'projects', 'ASEpredictor')

ipd <- ppaste(pjd, 'outputs', 'biosGavinOverlapCov10')
ipf <- ppaste(ipd, 'biosGavinOlCv10AntUfltCst.tsv')

opd <- ppaste(pjd, 'outputs', 'biosGavinOverlapCov10')
opf <- ppaste(opd, 'biosGavinOlCv10AntUfltCstLog2FCBs.tsv')

transform <- function(rtb, cr){
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
        t.test(log2FC, mu=0)[[3]]
      ),
      FDRPerVariant=ifelse(length(log2FC)<=1, FDRPerVariant, NA)
    ) %>%
    mutate(ASE=ifelse(p_value<=0.01, 1, 0)) %>%
    ungroup() %>%
    select(rn) %>%
    arrange(chr, pos, ref, alt) %>%
    distinct() %>%
    as.data.frame()

  return(gp)
}

rtb <- read.csv(ipf, header = TRUE, sep='\t')
rn <- colnames(rtb)[1:117]
rn <- c(rn, 'var', 'mean', 'p_value', 'gp_size', 'FDRPerVariant', 'ASE')

odf <- transform(rtb, rn)
write.table(odf, file=opf, quote=FALSE, sep="\t", row.names=FALSE)
