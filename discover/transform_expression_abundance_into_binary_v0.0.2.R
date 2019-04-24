#!/usr/bin/env Rscript
#
# File Name  : transform_log2fc_into_binary_by_ttest.R
# Author     : zhzhang
# E-mail     : zhzhang2015@sina.com
# Created on : Fri 25 Jan 2019 10:51:30 AM CET
# Version    : v0.0.2
# License    : MIT
#

#SBATCH --time=0:3:0
#SBATCH --output=%j-%u-transform_expression_abundance_into_binary_sbatch_v0.0.2.log
#SBATCH --job-name=teaibs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
# module load R/3.3.3-foss-2015b
# Rscript  transform_expression_abundance_into_binary_v0.0.2.R chr1_training_set.tsv chr1.tsv

rm(list = ls())

library(dplyr)

# Customized paste function
ppaste <- function(...){ paste(..., sep = "/") }

# Log-likelihood function under Binomial distritbution
bn_llik <- function(p, alts, refs) {
  totals <- alts + refs
  r <- -sum(lchoose(totals, alts) + alts * log(p) + refs * log(1 - p))

  if (r == -Inf){
    return(-.Machine$integer.max)
  } else if (r == Inf){
    return(.Machine$integer.max)
  } else {
    return(r)
  }
}

# Likelihood ratio test under binomial likelihood
bn_lrt <- function(alt_counts, ref_counts) {

  nul_p <- 0.5
  nul_llik <- bn_llik(nul_p, alt_counts, ref_counts)  # Filter out alt == ref

  alt_p <- exp(mean(log(alt_counts / ref_counts)))
  alt_llik_opt <- optim(
    par = alt_p, fn = bn_llik, NULL,
      method = "L-BFGS-B", lower = c(1e-12), upper = c(1),
      control = list(maxit = 10000),
      alts = alt_counts, refs = ref_counts
  )

  alt_p <- alt_llik_opt$par
  alt_llik <- alt_llik_opt$value

  dof <- length(alt_counts) - 1

  if (dof == 0) {
    chisq <- NA
    chisq_p <- NA
  } else {
    chisq <- - 2 * (alt_llik - nul_llik)
    chisq_p <- 1 - pchisq(chisq, dof)
  }

  r <- list(nul_bn_llik=nul_llik, alt_bn_llik=alt_llik, p_value = chisq_p)
  return(r)
}

# Log-likelihood function under Beta-Binomial distribution
bb_llik <- function(p_od, alts, refs){
  alts_len <- length(alts)
  refs_len <- length(refs)
  if(alts_len != refs_len) {
    stop("alts and refs should have identical length...")
  }

  if (length(p_od) == 1){
    p <- 0.5
    od <- p_od[[1]]
  } else if (length(p_od) == 2){
    p <- p_od[[1]]
    od <- p_od[[2]]
  } else {
    stop("p_od should be no more than 2 elements")
  }

  if (p > 1 || p < 0){
    stop("The first element of p_od should be a decimal between 0 to 1")
  }

  if (od <= 0) { 
	stop("The second element of p_od should be positive, but get ", od) 
  }

  a <- od * p
  b <- od * (1 - p)
  r <- -sum(lbeta(alts + a, refs + b) - lbeta(a, b) + lchoose(alts + refs, alts))

  if (r == -Inf){
    return(-.Machine$integer.max)
  } else if (r == Inf){
    return(.Machine$integer.max)
  } else {
    return(r)
  }
}

# Likelihood ratio test under Beta-Binomial distribution
bb_lrt <- function(alt_counts, ref_counts){
  alts_len <- length(alt_counts)
  refs_len <- length(ref_counts)

  if(alts_len != refs_len) {
    stop("alt_counts and ref_counts should have identical length...")
  }

  alt_od_init <- sum(alt_counts + ref_counts)  # Using total reads counts
  alt_p_init <- exp(mean(log(alt_counts / ref_counts)))
  alt_opt <- optim(
    par = c(alt_p_init, alt_od_init), fn = bb_llik, NULL,
    method = "L-BFGS-B", lower = c(1e-12, 1e-10), upper = c(1, 1e10),
    control = list(maxit = 10000),
    alts = alt_counts, refs = ref_counts
  )

  alt_par <- alt_opt$par
  alt_llik <- alt_opt$val

  # Using optim to find the over-dispersion parameter under null model
  # nul_opt <- optim(
  #   par = c(10), fn = bb_llik, NULL,
  #     method = "L-BFGS-B", lower = c(1e-12), upper = c(1e12),
  #     control = list(maxit = 10000),
  #     alts = alt_counts, refs = ref_counts
  # )
  # nul_par <- nul_opt$par
  # nul_llik <- nul_opt$val

  # nul_par <- alt_par[[2]]  # Using the over-dispersion parameter derived from samples 
  nul_par <- sum(alt_counts + ref_counts)  # Using average total reads counts
  nul_llik <- bb_llik(c(0.5, nul_par), alt_counts, ref_counts)
  nul_llik <- nul_llik

  dof <- alts_len - 1
  if (dof == 0) {
    chisq <- NA
    chisq_p <- NA
  } else {
    chisq <- - 2 * (alt_llik - nul_llik)
    chisq_p <- 1 - pchisq(chisq, dof)
  }

  r <- list(
    nul_p=0.5, nul_od=nul_par, nul_llik_bb=nul_llik,
    alt_p=alt_par[[1]], alt_od=alt_par[[2]], alt_llik_bb=alt_llik,
    p_value = chisq_p
  )

  return(r)
}


trans_into_bin <- function(rtb, cr, pv = 0.05, min_dep = 10, min_dep_per = 3){
  cat("FILTER: BY refCountsBios and altCountsBios ...\n")
  gp <- rtb %>% 
    filter(
    (refCountsBios >= min_dep_per)
    & (altCountsBios >= min_dep_per)
    & (refCountsBios != altCountsBios)
    & (refCountsBios + altCountsBios >= min_dep)
    & (!is.na(GeneID))
    & (!is.na(FeatureID))
    & (!is.na(GeneName))
  ) 

  cat("MUTATE: ADD bn_p, bb_p, group_size, and log2FC ...\n")
  gp <- gp %>%
    group_by(Chrom, Pos, Ref, Alt, GeneID) %>%
    mutate(
      log2FC = log2(sum(altCountsBios) / sum(refCountsBios)),
	  bn_p = bn_lrt(altCountsBios, refCountsBios)$p_value, 
	  bb_p = bb_lrt(altCountsBios, refCountsBios)$p_value,
	  group_size = n_distinct(sampleBios)
    )

  cat("MUTATE: ADD bn_p_adj, bn_ASE, bb_p_adj, bb_ASE ...\n")
  gp <- gp %>%
    ungroup() %>%
    mutate(
      bn_p_adj = p.adjust(bn_p, method = "fdr"),
      bb_p_adj = p.adjust(bb_p, method = "fdr"),
      bn_ASE = ifelse(bn_p_adj < pv, ifelse(log2FC < 0, -1, 1), 0),
      bb_ASE = ifelse(bb_p_adj < pv, ifelse(log2FC < 0, -1, 1), 0)
    ) %>%
    select(rn)

  cat("ARRANGE: Filter ...\n")
  gp <- gp %>%
    arrange(Chrom, Pos, Ref, Alt) %>%
    distinct() %>%
    as.data.frame()

  return(gp)
}

# Deal with input and output
args <- commandArgs(trailingOnly = TRUE)
if(length(args) == 2){
  ipf <- args[1]
  opf <- args[2]
} else {
  stop("Wrong number of arguments. Require 2...\n")
}

# Deal with `tar` compressed files
if (endsWith(ipf, '.tar.gz') || endsWith(ipf, '.tgz')){ ipf <- untar(ipf) }
rtb <- read.csv(ipf, header = TRUE, sep = "\t")
rn <- c(
  colnames(rtb)[1:107], "log2FC", "bn_p", "bn_p_adj", "bb_p", "bb_p_adj",
  "group_size", "bn_ASE", "bb_ASE"
)

# Calculate
odf <- trans_into_bin(rtb, rn)
# cat(dim(odf[which(abs(odf$bn_ASE)==1), ]), " in bn_ASE(1, -1)\n")
# cat(dim(odf[which(abs(odf$bb_ASE)==1), ]), " in bb_ASE(1, -1)\n")

# Dump out results
write.table(odf, file = opf, quote = FALSE, sep = "\t", row.names = FALSE)
