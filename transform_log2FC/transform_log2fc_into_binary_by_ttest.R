#!/usr/bin/env Rscript
#
# File Name  : transform_log2fc_into_binary_by_ttest.R
# Author     : zhzhang
# E-mail     : zhzhang2015@sina.com
# Created on : Fri 25 Jan 2019 10:51:30 AM CET
# Version    : v0.0.1
# License    : MIT
#

rm(list = ls())

library(dplyr)
# library(ggplot)

ppaste <- function(...){ paste(..., sep = "/") }

trans_into_bin <- function(rtb, cr, pv = 0.01){
  cat("Filter: BY refCountsBios and altCountsBios ...\n")
  gp <- rtb %>% 
    filter(
      refCountsBios > 4 & altCountsBios > 4 & refCountsBios != altCountsBios
    )

  cat("Mutate: ADD binom_p, FC, and log2FC ...\n")
  gp <- gp %>%
    rowwise() %>%
    mutate(
      binom_p = binom.test(c(refCountsBios, altCountsBios))$p.value,
      FC = altCountsBios / refCountsBios,
      log2FC = log2(altCountsBios / refCountsBios),
    ) %>%
    ungroup()

  cat("Group: BY (chr, pos, ref, and alt) ...\n")
  gp <- gp %>% group_by(chr, pos, ref, alt) 

  cat("Mutate: ADD FDRPerVariant and etc. ...\n")
  gp <- gp %>% 
    mutate(
      FDRPerVariant = p.adjust(binom_p),
      # varInsideChi2Pval = ifelse(
         # length(refCountsBios) <= 2,  
         # 1,
         # TODO: Is there a bug for chisq.test? it claimed at least 
         # TODO: two levels, but here I used at least there levels
         # chisq.test(refCountsBios, altCountsBios)$p.value
      # ),
      group_size = length(log2FC),

      # Using log2(altCountsBios/refCountsBios)
      meta_log2FC_mean = log2(sum(altCountsBios) / sum(refCountsBios)),
      log2FC_var = ifelse(length(log2FC) <= 1, 0, var(log2FC)),
      log2FC_mean = mean(log2FC),
      pval_st_log2FC = ifelse(
        length(log2FC) > 2,
        ifelse(max(log2FC) == min(log2FC), 0, shapiro.test(log2FC)$p.value),
        0
      ),
      pval_st_log2FC_adj = p.adjust(pval_st_log2FC),
      pval_tt_log2FC = ifelse(
        length(log2FC) <= 2,
        ifelse(FDRPerVariant <= 0.01 || abs(log2FC) >= 1, 0, 1),
        ifelse(
          max(log2FC) == min(log2FC),
          ifelse(abs(max(log2FC)) > 1, 0, 1),
          t.test(log2FC, mu = meta_log2FC_mean[[1]])$p.value
        )
      ),
      pval_tt_log2FC_adj = p.adjust(pval_tt_log2FC),

      # Using altCountsBios / refCountsBios
      meta_FC_Mean = sum(altCountsBios) / sum(refCountsBios),
      FC_var = ifelse(length(FC) <= 1, 0, var(FC)),
      FC_mean = mean(FC),
      pval_st_FC = ifelse(
        length(FC) > 2,
        ifelse( max(log2FC) == min(FC), 0, shapiro.test(FC)$p.value),
        0
      ),
      pval_st_FC_adj = p.adjust(pval_st_FC),
      pval_tt_FC = ifelse(
        length(FC) <= 2,
        ifelse(FDRPerVariant <= 0.01 || abs(log2FC) >= 1, 0, 1),
        ifelse(
          max(FC) == min(FC),
          ifelse(abs(max(FC)) > 1, 0, 1),
          t.test(FC, mu = meta_FC_Mean[[1]])$p.value
        )
      ),
      pval_tt_FC_adj = p.adjust(pval_tt_FC),
    )

  cat("Mutate: ADD ASE ...\n")
  gp <- gp %>%
    ungroup() %>%
    mutate(
       ASE = ifelse(pval_tt_FC_adj <= pv, ifelse(log2FC_mean < 0, -1, 1), 0)
    )

  cat("SELECT, ARRANGE, DISTINCT, and Make dataframe ...\n")
  gp <- gp %>%
    select(rn) %>%
    arrange(chr, pos, ref, alt) %>%
    distinct() %>%
    as.data.frame()

    return(gp)
}

hmd <- path.expand("~")
pjd <- ppaste(hmd, "Documents", "projects", "ASEpredictor")
opd <- ppaste(pjd, "outputs", "biosGavinOverlapCov10")
ipd <- ppaste(pjd, "outputs", "biosGavinOverlapCov10")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0){
  ipf <- ppaste(ipd, "biosGavinOverlapCov10Anno.tsv")
  opf <- ppaste(opd, "biosGavinOlCv10AntUfltCstLog2FCBin.tsv")
} else if(length(args) == 1){
  ipf <- args[1]
  opf <- ppaste(opd, "biosGavinOlCv10AntUfltCstLog2FCBin.tsv")
} else if(length(args) == 2){
  ipf <- args[1]
  opf <- args[2]
}

# Deal with `tar` compressed files
if (endsWith(ipf, '.tar.gz') || endsWith(ipf, '.tgz')){ ipf <- untar(ipf) }
rtb <- read.csv(ipf, header = TRUE, sep = "\t")
rn <- c(
  colnames(rtb)[1:117], "meta_log2FC_mean", "log2FC_var", "log2FC_mean", 
  "pval_tt_log2FC", "pval_tt_log2FC_adj","pval_st_log2FC", 
  "pval_st_log2FC_adj", "meta_FC_Mean", "FC_var", "FC_mean", "pval_tt_FC", 
  "pval_tt_FC_adj", "pval_st_FC", "pval_st_FC_adj", "group_size", "ASE"
 )
odf <- trans_into_bin(rtb, rn)

write.table(odf, file = opf, quote = FALSE, sep = "\t", row.names = FALSE)
