library(dplyr)
rm(list = ls())
setwd(dir = "/home/umcg-zzhang/Documents/git/asep/bios_eqtlgen_overlaps")

eqtlfn <- "eqtlgen_chr22.tsv"
eqtldf <- read.table(eqtlfn, sep = "\t", header = TRUE)
eqtlgroups <- eqtldf %>%
    group_by(SNPChr, SNPPos, AssessedAllele, OtherAllele)


biosfn <- "bios_chr22.tsv"
biosdf <- read.table(biosfn, sep = "\t", header = TRUE)
biosgroups <- biosdf %>%
    group_by(chr, pos, ref, alt)


intersect(eqtldf, biosdf)
