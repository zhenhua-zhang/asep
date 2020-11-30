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

### Measurements

- Allele-specific PCR
- Fluorescence in situ hybridization (`FISH`)
- Array-based method
- RNA sequencing

## Measurement of ASE Effects
In previous studies, there many different method to estimate the ASE effects at different level.

### Binomial model
We assume the outcomes of each individual for each locus are independent and identical distributed(i.i.d.).
Generally, for the observed locus, a simple distribution of allele-specific expression result is respected to Binomial distribution.
$$
f(k|n,\theta) \sim Binom(n, \theta)
$$
where `k` is the amount of success and means the abundance of alternative alleles in our case. We use a maximum likelihood estimation (`MLE`) under a
Binomial distribution to estimate the parameter $\theta$ at first. Then a log-likelihood ratio test (`LRT`) was done to determine the goodness of fit to the balanced model in which the expression of two are identical or gently different(i.e. $\theta \approx 0.5$).

### Beta-Binomial model

However, the $\theta$ could be various. And we assume $\theta$ is respect to a Beta distribution.
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
The $\pi$ here is the mean probability of success in Bernoulli trials, while the $\rho$ is the over-dispersion of the distribution.

In our problem, we need to estimate $\pi$ and $\rho$ for null and alternative hypothesis.

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

Then, we use $\pi, \gamma$ as the variables of our likelihood function, where $\pi$ is the probability of success, and $\gamma$ is the over-dispersion parameter. Here is our null model: $\pi=0.5, \gamma=100$ which means the probability of success is 0.5 and less over-dispersion.

## Predicting ASE effects via Machine Learning Model
One of goals in this studies is to connect the DNA annotations and ASE effects
by machine learning.

## Problems
1. If the `--nested-cv` is not given, then the scripts picks up a model fitted by `RandomizedSearchCV` as a estimator for the cross validation. In this case, the validation is called outer validation, which enables the script to draw a ROC-AUC plot. However, th training report will be useless as it only records the report of the first fitting. (The code should be modified to fix this potential problem)
