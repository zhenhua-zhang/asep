require(data.table)
require(ggplot2)

log_or_0 <- function(num, inf = 100) {
  if (num < 0) {
    return(NA)
  } else if (num == 0) {
    return(inf)
  } else {
    return(log10(num))
  }
}

t_test <- function(gtex_df, bios_df, msmnt = "group_size") {
  bios_ase <- which(bios_df$bb_ASE == 1)
  bios_ase_msmnt <- bios_df[bios_ase, msmnt]
  cat("Nr of selected variants WITH ASE effect in BIOS:", length(bios_ase_msmnt), "\n")

  bios_nase <- which(bios_df$bb_ASE == 0)
  bios_nase_msmnt <- bios_df[bios_nase, msmnt]
  cat("Nr of selected variants WITHOUT ASE effect in BIOS:", length(bios_nase_msmnt), "\n")

  gtex_ase <- which(gtex_df$bb_ASE == 1)
  gtex_ase_msmnt <- gtex_df[gtex_ase, msmnt]
  cat("Nr of selected variants WITH ASE effects in GTEx:", length(gtex_ase_msmnt), "\n")

  gtex_nase <- which(gtex_df$bb_ASE == 0)
  gtex_nase_msmnt <- gtex_df[gtex_nase, msmnt]
  cat("Nr of selected variants WITHOUT ASE effects in GTEx:", length(gtex_nase_msmnt), "\n")

  bios_tt <- t.test(bios_ase_msmnt, bios_nase_msmnt)
  gtex_tt <- t.test(gtex_ase_msmnt, gtex_nase_msmnt)

  tt <- list()
  tt[[msmnt]] <- list(biostt = bios_tt, gtextt = gtex_tt)

  return(tt)
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  gtex_fn <- if (file.exists(args[1])) args[1] else stop("File (gtex_fn) not found")
  bios_fn <- if (file.exists(args[2])) args[2] else stop("File (bios_fn) not found")
  msmnts <- args[3]

  mingps <- 5
  if (length(args) == 4) {
    mingps <- as.numeric(args[4])
  }

  maxgps <- 5000
  if (length(args) == 5) {
    maxgps <- as.numeric(args[5])
  }

  bios_df <- fread(bios_fn, showProgress = FALSE, data.table = FALSE)
  cat("Nr of variants in BIOS:", dim(bios_df)[[1]], "\n")

  bios_df <- bios_df[(bios_df$group_size >= mingps) & (bios_df$group_size <= maxgps), ]
  cat("Nr of SELECTED variants in BIOS:", dim(bios_df)[[1]])
  cat(" min group_size >=", mingps)
  cat(", max group_size <=", maxgps, "\n")

  gtex_df <- fread(gtex_fn, showProgress = FALSE, data.table = FALSE)
  cat("Nr of variants in GTEx:", dim(gtex_df)[[1]], "\n")

  gtex_df <- gtex_df[(gtex_df$group_size >= mingps) & (gtex_df$group_size <= maxgps), ]
  cat("Nr of SELECTED variants in GTEx:", dim(gtex_df)[[1]], "min group_size >=", mingps, ", max group_size <=", maxgps, "\n")

  msmnts <- strsplit(msmnts, ",", fixed = TRUE)[[1]]
  for (msmnt in msmnts) {
    bios_df[msmnt] <- as.numeric(bios_df[[msmnt]])
    gtex_df[msmnt] <- as.numeric(gtex_df[[msmnt]])
    ttr <- t_test(gtex_df, bios_df, msmnt)

    print(ttr)
  }

  # print(summary(glm(bb_ASE ~ gnomAD_AF + EncExp + cHmmTx + cHmmTxWk, data = gtex_df)))
  # print(summary(glm(bb_ASE ~ gnomAD_AF + EncExp + cHmmTx + cHmmTxWk, data = bios_df)))
}


main()