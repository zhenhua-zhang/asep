# dbinom() is the PMF, while pbinom() id CDF

rm(list=ls())
library(ggplot2)

# beta-binomial probability mass function
bb_pmf <-function(success, trials, alpha=2, beta=3){
  if(success>trials){ return(NULL) }
  numerator <- sum(c(lgamma(trials+1), lgamma(success+alpha), lgamma(trials-success+beta), lgamma(alpha+beta)))
  denominator <- sum(c(lgamma(success+1), lgamma(trials-success+1), lgamma(trials+alpha+beta), lgamma(alpha), lgamma(beta)))
  return(exp(numerator - denominator))
}

# beta-binomial cumulative distribution function
bb_cdf <- function(success, trials, alpha=2, beta=3, init=0){
  return(sum(sapply(init:success, bb_pmf, trials, alpha, beta)))
}

estimate_alpha_beta <- function(p_vec){
  # The input is a vector of estimated p
  e <- mean(p_vec)
  v <- var(p_vec)
  alpha <- ((1-e)/v - 1/e) * (e^2)
  beta <- alpha*(1/e-1)
  return(c(alpha, beta))
}

bb_test <- function(successes, trials, target=0.5){
  p_vec <- c(successes / trials)
  ab <- estimate_alpha_beta(p_vec)
  alpha <- ab[[1]]
  beta <- ab[[2]]
  
  cdf <- bb_cdf(successes, trials, alpha, beta)
}

min_size <- function(p_val, trials, upper, lower=0, alpha=2, beta=3, prec=0.01){
  if(abs(upper-lower) <= 1) {return((upper+lower)/2)}
  area <- bb_cdf((upper+lower)/2, trials, alpha, beta)
  if(abs(area-p_val) <= prec) {
    return((upper+lower)/2)
  } else if(area<p_val){
    lower <- round((upper+lower)/2)
    min_size(p_val, trials, upper, lower, alpha, beta)
  } else {
    upper <- round((upper+lower)/2)
    min_size(p_val, trials, upper, lower, alpha, beta)
  }
}

draw_bb_pmf <- function(trials, p_vec=c(), alpha=2, beta=3){
  bars_df <- data.frame(k=0:trials, p=sapply(0:trials, bb_pmf, trials, alpha, beta))
  p <- ggplot(data=bars_df, aes(x=k, y=p)) + theme_bw()
  for(p_value in p_vec){
    p <- p + geom_vline(xintercept=min_size(p_value, trials, trials, 0, alpha, beta))
  }
  p <- p + geom_line(stat="identity", color='red')
  p
}

draw_binom_pmf <- function(trials){
  bars_df <- data.frame(k=0:trials, p=sapply(0:trials, dbinom, trials, 0.5))
  p <- ggplot(data=bars_df, aes(x=k, y=p)) + theme_bw()
  p <- p + geom_line(stat="identity", color='red')
  p
}


input_file <- "/home/umcg-zzhang/Documents/projects/ASEpredictor/outputs/biosGavinOverlapCov10/NOC2L"
df <- read.csv(input_file, sep="\t")
df$p <- df$altCountsBios / (df$refCountsBios + df$altCountsBios)

ab <- estimate_alpha_beta(df$p)
(alpha <- ab[[1]])
(beta <- ab[[2]])
(succ <- sum(df$altCountsBios))
(tria <- sum(df$refCountsBios + df$altCountsBios))
(estp <- bb_cdf(tria/2, tria, alpha, beta))
draw_bb_pmf(tria, p_vec=c(), alpha=alpha, beta=beta)

pbinom(succ, tria, 0.5)
draw_binom_pmf(tria)

dbinom(10, 20, p=0.5)
binom.test(succ+2000, tria, alternative = "less")

# It indeed a beta-binomial distribution. The joint distribution of multiple 
# binomial random variable is still a binomial random variable, when the 
# variables are iid and the p is identical. While if the p is various, then 
# asuuming p is repspect to Beta distribution. Then the joint distribution is 
# Beta-binomial distribution, while the n is the sum of all trials and successes
# is the sum of all successful trials, but HOW to know the shape parameters
# (alpha and beta for Beta distribution)