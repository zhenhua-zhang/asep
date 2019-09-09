#!/usr/bin/env Rscript
library(data.table)

tst <- function(gtex_df, bios_df, msmnts = "group_size") {
  gtex_ase <- which(gtex_df$bb_ASE == 1)
  gtex_nase <- which(gtex_df$bb_ASE == 0)
  bios_ase <- which(bios_df$bb_ASE == 1)
  bios_nase <- which(bios_df$bb_ASE == 0)

  tt <- list()
  ct <- list()
  msmnts <- c(msmnts)
  for (msmnt in msmnts) {
    print(msmnt)
    # 1. t.test
    ## group_size. To check the local allele frequency influent the statistic
    ## gnomAD_AF. To check the ethic allele frequency influent the statistic
    bios_tt <- t.test(bios_df[bios_ase, msmnt], bios_df[bios_nase, msmnt])
    gtex_tt <- t.test(gtex_df[gtex_ase, msmnt], gtex_df[gtex_nase, msmnt])
    tt[[msmnt]] <- list(biostt = bios_tt, gtextt = gtex_tt)

    # 2. correlation analysis. Spearman's rank-order correlation
    bios_bbp <- which((bios_df$bb_p_adj != NA) & (bios_df[msmnt] != NA))
    gtex_bbp <- which((gtex_df$bb_p_adj != NA) & (gtex_df[msmnt] != NA))
    bios_ct <- cor.test(-log10(bios_bbp["bb_p_adj"]), bios_bbp[msmnt], method="spearman")
    gtex_ct <- cor.test(-log10(gtex_bbp["bb_p_adj"]), gtex_bbp[msmnt], method="spearman")
    ct[[msmnt]] <- list(biosct=bios_ct, gtexct=gtex_ct)
  }
  return(list(tt = tt, ct = ct))
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  gtex_fn <- if (file.exists(args[1])) args[1] else stop("file not found")
  bios_fn <- if (file.exists(args[2])) args[2] else stop("file not found")
  msmnts <- args[3] # delimited by comma

  gtex_df <- fread(gtex_fn, showProgress = FALSE, data.table = FALSE)
  bios_df <- fread(bios_fn, showProgress = FALSE, data.table = FALSE)
  msmnts <- strsplit(msmnts, ",", fixed = TRUE)[[1]]

  print(tst(gtex_df, bios_df, msmnts))
}

main()