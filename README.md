# Allele specific expression analysis on 3475 blood samples

## Backgrounds

### Definition
Allele-Specific Expression(ASE, a.k.a. Allelic Imbalance or Allelic Expression)
refers the phenomenon that various abundance of transcripts from maternal and
paternal alleles at heterozygous loci in diploid organisms.

### Causation
Genetic variants (e.g. `eQTL`), epigenetic markers (e.g. `eQTM`), non-sense 
mediated decay(i.e. `NMD`) and alternative splicing etc.

### Measurements  

- Allele-specific PCR
- Fluorescence in situ hybridization (`FISH`)
- Array-based method
- RNA sequencing 

### Applications

- Cancer
- Brain development
- Heart diseases


## Measurement of ASE Effects
In previous studies, there many different method to estimate the ASE effects at
different level. 

### Binomial model
We assume the outcomes of each individual for each locus are independent and 
identical distributed(i.i.d.).
Generally, for the observed locus, a simple distribution of allele-specific
expression result is respected to Binomial distribution.
$$
f(k|n,\theta) \sim Binom(n, \theta)
$$
where `k` is the amount of success and means the abundance of alternative 
alleles in our case. We use a maximum likelihood estimation (`MLE`) under a 
Binomial distribution to estimate the parameter $\theta$ at first. Then a 
log-likelihood ratio test (`LRT`) was done to determine the goodness of fit to
the balanced model in which the expression of two are identical or gently
different(i.e. $\theta \approx 0.5$).

```{R}
#
# Log-likelihood function under Binomial distritbution
#
bn_llik <- function(p, alts, refs) {
  totals <- alts + refs
  r <- -sum(dbinom(alts, totals, p, log=TRUE))

  if (r == -Inf){
    return(-.Machine$integer.max)
  } else if (r == Inf){
    return(.Machine$integer.max)
  } else {
    return(r)
  }
}
```

Likelihood ratio test by maximum likelihood estimation

```{R}
#
# Likelihood ratio test under binomial likelihood
#
bn_lrt <- function(alt_counts, ref_counts) {
  total_counts <- alt_counts + ref_counts

  nul_p <- 0.5
  nul_llik <- bn_llik(nul_p, alt_counts, ref_counts)

  alt_p <- sum(alt_counts) / sum(ref_counts)
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
    chisq <- 2 * (alt_llik - nul_llik)
    chisq_p <- 1 - pchisq(chisq, dof)
  }

  r <- list(nul_bn_llik=nul_llik, alt_bn_llik=alt_llik, p_value = chisq_p)
  return(r)
}
```

### Beta-Binomial model

However, the $\theta$ could be various. And we assume $\theta$ is respect to a
Beta distribution.
$$
\theta \sim Beta(\alpha, \beta)
$$
Then the joint distribution is a Beta-Binomial distribution.

In Beta-Binomial distribution, let $\pi=\frac{\alpha}{\alpha+\beta}$the mean is
$$
\mu=\frac{n\alpha}{\alpha + \beta}=n\pi
$$
while let $\rho=\frac{1}{\alpha+\beta+1}$, then the variance is
$$
\sigma^2 =\frac{n\alpha\beta(\alpha+\beta+n)}{(\alpha+\beta)^2(\alpha+\beta+1)} \\
  =n\pi(1-\pi)[1+(n-1)\rho]
$$
The $\pi$ here is the mean probability of success in Bernoulli trials, while the
$\rho$ is the over-dispersion of the distribution.

In our problem, we need to estimate $\pi$ and $\rho$ for null and alternative
hypothesis.

First, we need to know the relation among $\pi, \rho, \alpha, \beta$.
Let $\gamma=\frac{(1-\rho)}{\rho}$
$$
\alpha =\frac{\pi(1-\rho)}{\rho} \\
  = \gamma\pi \\
\beta =\frac{(1-\pi)(1-\rho)}{\rho} \\
  =\gamma(1-\pi)
$$
Correspondingly,
$$
\pi =\frac{\alpha}{\alpha+\beta} \\
\gamma = \alpha+\beta
$$

Then, we use $\pi, \gamma$ as the variables of our likelihood function, where
$\pi$ is the probability of success, and $\gamma$ is the over-dispersion parameter.

Here is our null model: $\pi=0.5, \gamma=100$ which means the probability of 
success is 0.5 and less over-dispersion.

```{R}
library(rmutil, warn.conflicts=FALSE)
library(ggplot2)
sample_space <- data.frame(
  x = 0:2000, p_values_equal = dbetabinom(0:2000, 2000, 0.5, 2000),
  p_values_bias = dbetabinom(0:2000, 2000, 0.3, 50)
)

p <- ggplot(data=sample_space) + theme_bw()
p <- p + geom_line(aes(x = x, y = p_values_equal), color = "red")
p <- p + geom_line(aes(x = x, y = p_values_bias), color = "green")
p

```

Log-likelihood function under Beta-Binomial distribution

```{R}
#
# Log-likelihood function under Beta-Binomial distribution
#
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
  if (od <= 0) { stop("The second element of p_od should be positive") }

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
```

Likelihood ratio test under Beta-Binomial distribution

```{R}
#
# Likelihood ratio test under Beta-Binomial distribution
#
bb_lrt <- function(alt_counts, ref_counts){
  alts_len <- length(alt_counts)
  refs_len <- length(ref_counts)

  if(alts_len != refs_len) {
    stop("alt_counts and ref_counts should have identical length...")
  }

  # nul_opt <- optim(
  #   par = c(10), fn = bb_llik, NULL,
  #     method = "L-BFGS-B", lower = c(1e-12), upper = c(1e12),
  #     control = list(maxit = 10000),
  #     alts = alt_counts, refs = ref_counts
  # )

  # nul_par <- nul_opt$par
  # nul_llik <- nul_opt$val

  nul_llik <- bb_llik(c(0.5, 1), alt_counts, ref_counts)
  nul_par <- 1.0
  nul_llik <- nul_llik


  alt_opt <- optim(
    par = c(0.5, 10), fn = bb_llik, NULL,
      method = "L-BFGS-B", lower = c(1e-12, 1e-12), upper = c(1, 1e12),
      control = list(maxit = 10000),
      alts = alt_counts, refs = ref_counts
  )

  alt_par <- alt_opt$par
  alt_llik <- alt_opt$val

  dof <- alts_len - 1
  if (dof == 0) {
    chisq <- NA
    chisq_p <- NA
  } else {
    chisq <- 2 * (alt_llik - nul_llik)
    chisq_p <- 1 - pchisq(chisq, dof)
  }

  r <- list(
    nul_p=0.5, nul_od=nul_par, nul_llik_bb=nul_llik,
    alt_p=alt_par[[1]], alt_od=alt_par[[2]], alt_llik_bb=alt_llik,
    p_value = chisq_p
  )

  return(r)
}
```

## Predicting ASE effects via Machine Learning Model

One of goals in this studies is to connect the DNA annotations and ASE effects
by machine learning.

## Are predicting ASE or othe thing?
1. 如果把所有的跟表达量相关的数据去掉会有什么结果？
   Features related to expression: cHmm*, EncExp
   To remove the influence of EncExp which representes the expression level for each allele, I plan
   to compare the diference of expression between loci with ASE effects and loci without ASE effects.
   However, theoretically, there a batch effect between the EncExp which is "Maximum ENCODE
   expression value" from ENCODE and exon-level expression data from BIOS.

2. 比较ASE位点的平均表达量和非ASE位点的平均表达量？
   我们的数据显示ASE跟表达量和AF相关，这是合理的。因为ASE是应该受到选择。
   Useless features: Chrom, Pos, Type, Length, oAA, nAA, GeneID, FeatureID, GeneName, CCDS, Intron, Exon,
<<<<<<< HEAD


## How to improve the model?
### Using Multiple factor analysis
  Using mulitple factor analysis [MFA](https://en.wikipedia.org/wiki/Multiple_factor_analysis) analysis before the training. There's an implementation of MAF in scikit-learn. The implementation is called `FactorAnalysis` in `decomposition` package.
  1. Why using MAF?
  Principal componet analysis (PCA) when variables are quantitative.
  Multiple correspondence analysis (MCA) when variables are qualitative.
  Multiple factor analysis (MAF) when variables are the mixture of the quantitative and the qualitative.
=======
>>>>>>> ebb62347c8acc13fb1f5d4800cc78729c8073262
