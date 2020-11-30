#/usr/bin/env Rscript
library(data.table)
library(ggplot2)
library(dplyr)
library(stringr)
library(svglite)

setwd("~/Documents/projects/ws_ase_pred/")

# class_col <- "bb_ASE"
# feature_col <- 'GerpN'
# sumtbl <- bios.dtfm %>%
#   group_by(get(class_col)) %>%
#   summarise(feature_mean = mean(get(feature_col)), stdev = sd(get(feature_col))) %>%
#   rename(ASE = `get(class_col)`) %>%
#   mutate(ASE = ifelse(.$ASE == 0, "non-ASE", "ASE")) %>%
#   as.data.frame()
# 
# print(sumtbl)

# A function to compare feautres between ASE and non-ASE.
cmpf <- function(dtfm, feature_col, class_col="bb_ASE", dw_bp=TRUE,
                 pref=NULL, bp_title=NULL, bp_x_lab=NULL,
                 bp_y_lab=NULL, log10_fc=FALSE, fmt="svg") {
  # Compare
  if (is.null(bp_title))
    bp_title = paste(feature_col, "difference between ASE and non-ASE")
  
  if (is.null(bp_x_lab))
    bp_x_lab = class_col
  
  if (is.null(bp_y_lab))
    bp_y_lab = feature_col
  
  if (is.null(pref))
    pref = paste0("./", feature_col, "-per_ASE_SNP.", fmt)
  
  if (log10_fc)
    feature_col_pl <- str_glue("log10({feature_col})")
  else
    feature_col_pl <- feature_col
  
  # Box plot to compare feautres between ASE and non-ASE
  if (dw_bp) {
    g <- ggplot(data = dtfm) + theme_bw()
    g <- g + geom_jitter(aes_string(x = class_col, y = feature_col_pl), size = 0.1)
    g <- g + geom_violin(aes_string(x = class_col, y = feature_col_pl), alpha = 0.5)
    g <- g + ggtitle(bp_title)
    g <- g + xlab(bp_x_lab) + ylab(bp_y_lab)
    g <- g + scale_x_discrete(limits = as.factor(c(0, 1)), labels = c("Non-ASE", "ASE"))
    
    if (is.character(fmt))
      fmt <- c(fmt)
    
    for (.fmt in fmt)
      ggsave(paste0(pref, ".", .fmt))
  }
  
  sumdtfm <- mean_per_class <- dtfm %>%
    group_by(get(class_col)) %>%
    summarise(mean = base::mean(base::get(feature_col), na.rm = TRUE),
              stdv = stats::sd(base::get(feature_col), na.rm = TRUE)) %>%
    rename(ASE=`get(class_col)`) %>%
    mutate(ASE = ifelse(.$ASE == 0, "non-ASE", "ASE")) %>%
    as.data.frame()
  
  return(sumdtfm)
}


fmt_sumdtfm <- function(dtfm, ftnm) {
  ase_list <-  list(non_ase_mean = dtfm$mean[1], non_ase_stdv = dtfm$stdv[1], ase_mean = dtfm$mean[2], ase_stdv = dtfm$stdv[2])
  return(ase_list)
}



# BIOS
bios_dtfm <- fread('./training/outputs/cadd_annot/allelicReadsCounts/bios.trainingset.pli_af.exon.fdr0.05.tsv.gz', header = TRUE, data.table = FALSE)
bios_dtfm <- bios_dtfm[((! is.na(bios_dtfm$bb_ASE)) & (bios_dtfm$group_size >= 5)),]
bios_dtfm$bb_ASE <- as.factor(bios_dtfm$bb_ASE)
bios_dtfm$gnomAD_AF <- as.numeric(bios_dtfm$gnomAD_AF)

cohort <- "bios"
class_col <- "bb_ASE"

ftnm <- "gnomAD_AF"
pref = str_glue("./integration/allele_frequency_in_bios_gtex/{ftnm}-per_ASE_SNP-{cohort}")
smdtfm <- cmpf(bios_dtfm, feature_col = ftnm, pref = pref,
               class_col = "bb_ASE", bp_x_lab = "ASE",
               bp_y_lab = "Reference allele frequency", log10_fc = TRUE,
               fmt = c('png', 'pdf', 'svg'))

ftpl <- c("GerpN", "bStatistic", "Dist2Mutation", "cDNApos", "cHmmReprPCWk",
          "cHmmQuies", "relcDNApos", "cHmmTx", "minDistTSE", "CDSpos")

bios_ftdtfm <- list()
for (ftnm in ftpl) {
  pref = str_glue("./integration/feature_means_in_bios_gtex/{ftnm}-per_ASE_SNP-{cohort}")
  smdtfm <- cmpf(bios_dtfm, feature_col = ftnm, pref = pref, class_col = class_col, bp_x_lab = "ASE")
  sumdtfm <- fmt_sumdtfm(smdtfm, ftnm)
  bios_ftdtfm[[ftnm]] <- sumdtfm
}
bios_ftdtfm <- as.data.frame(rbindlist(bios_ftdtfm, idcol = TRUE))




# GTEx
dtfm_path = './validating//outputs/cadd_annot/allelicReadsCounts/gtex.trainingset.pli_af.exon.fdr0.05.tsv.gz'
gtex_dtfm <- fread(dtfm_path, header = TRUE, data.table = FALSE)
gtex_dtfm <- gtex_dtfm[((! is.na(gtex_dtfm$bb_ASE)) & (gtex_dtfm$group_size >= 5)),]
gtex_dtfm$bb_ASE <- as.factor(abs(gtex_dtfm$bb_ASE))
gtex_dtfm$gnomAD_AF <- as.numeric(gtex_dtfm$gnomAD_AF)

cohort <- "gtex"
ftnm <- "gnomAD_AF"
pref <- str_glue("./integration/allele_frequency_in_bios_gtex/{ftnm}-per_ASE_SNP-{cohort}")
smdtfm <- cmpf(gtex_dtfm, feature_col = ftnm, pref = pref,
               class_col = "bb_ASE", bp_x_lab = "ASE", bp_y_lab = "Reference allele frequency",
               log10_fc = TRUE, fmt = c('png', 'pdf', 'svg'))

ftpl <- c("GerpN", "bStatistic", "cHmmReprPCWk", "GerpRS", "cHmmTxWk", 
          "minDistTSS", "cHmmTx", "cDNApos", "minDistTSE", "cHmmQuies" )
gtex_ftdtfm <- list()
for (ftnm in ftpl) {
  pref = str_glue("./integration/feature_means_in_bios_gtex/{ftnm}-per_ASE_SNP-{cohort}")
  smdtfm <- cmpf(gtex_dtfm, feature_col = ftnm, pref = pref, class_col = class_col, bp_x_lab = "ASE")
  sumdtfm <- fmt_sumdtfm(smdtfm, ftnm)
  gtex_ftdtfm[[ftnm]] <- sumdtfm
}
gtex_ftdtfm <- as.data.frame(rbindlist(gtex_ftdtfm, idcol = TRUE))

bios_gtex_ftdtfm <- full_join(bios_ftdtfm, gtex_ftdtfm, suffix = c('-bios', '-gtex'), by = '.id')
fwrite(bios_gtex_ftdtfm, "./integration/feature_means_in_bios_gtex/bios_gtex_features_mean_stdev.csv")

