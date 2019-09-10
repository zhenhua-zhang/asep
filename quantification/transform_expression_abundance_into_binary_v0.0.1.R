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
  cat("FILTER: BY refCountsBios and altCountsBios ...\n")
  gp <- rtb %>% 
    filter(
      refCountsBios > 4 & altCountsBios > 4 & refCountsBios != altCountsBios
    )

  cat("MUTATE: ADD binom_p, group_size, and binom_p_adj ...\n")
  gp <- gp %>%
    group_by(chr, pos, ref, alt) %>%
    mutate(
	  log2FC = log2(sum(refCountsBios) / sum(altCountsBios)),
      binom_p = binom.test(c(sum(refCountsBios), sum(altCountsBios)))$p.value,
      group_size = length(log2FC)
    )

  cat("MUTATE: ADD binom_p_adj, ASE ...\n")
  gp <- gp %>%
    ungroup() %>%
    mutate(
      binom_p_adj = p.adjust(binom_p),
      ASE = ifelse(binom_p_adj < 0.01, ifelse(log2FC < 0, -1, 1), 0)
    ) %>%
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
  opf <- ppaste(opd, "biosGavinOlCv10AntUfltCstBin")
} else if(length(args) == 2){
  ipf <- args[1]
  opf <- args[2]
}

# Deal with `tar` compressed files
if (endsWith(ipf, '.tar.gz') || endsWith(ipf, '.tgz')){ ipf <- untar(ipf) }
rtb <- read.csv(ipf, header = TRUE, sep = "\t")
rn <- c(
  colnames(rtb)[1:117], "log2FC", "binom_p", "binom_p_adj", "group_size",
  "ASE"
)
odf <- trans_into_bin(rtb, rn)

write.table(odf, file = opf, quote = FALSE, sep = "\t", row.names = FALSE)
