### Background
           1      2      3  4                           5  6       7 
DNA:  ---▣▣▣▣▣---▣▣▣---▤▤▤▤▤┿▨▨▨▨▨---▨▨▨▨---▨▨▨▨▨---▨▨▨▨▶▤▤▤▤▤---▣▣▣▣▣---

RNA:                     ▤▤▤▤▤┿▨▨▨▨▨---▨▨▨▨---▨▨▨▨▨---▨▨▨▨▶▤▤▤▤▤

Pro:                        ◁▤▤▤▤▤┿▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▶▤▤▤▤▤AAAAAAAAA

1: Enhancer or silencer
2: Promoter
3: 5'UTR
4: start codon
5: stop codon
6: 3'UTR
7: Enhancer or silencer


### Project path
/groups/umcg-bios/tmp03/users/umcg-zzhang/projects/ASEpredictor


### Sample information
- Discarding LLDeep data & only considering BIOS data
- Discarding duplications
- Samples discarded by Niek (without CODAM and failed QC)


### Genotype results
- /groups/umcg-bios/prm02/projects/HRC\_imputations/


### Reference genome and annotations file
- ASE analysis (Niek):
  + Reference genome: 
    * path: /apps/data/ftp.broadinstitute.org/bundle/2.8/b37/human_g1k_v37.fasta 
    * md5: 0ce84c872fc0072a885926823dcd0338
  + Annotation file:
    * path: /apps/data/ftp.ensemble.org/pub/release-75/gtf/homo_sapiens/Homo_sapiens.GRCh37.75.gtf
    * md5: 4b16a5c536f675b7c5d4aa272ce6536b

- Imputation (Freerk):
  + Reference genome:
    * path:
    * md5:


### What we have?
- Genotypes are imputed by IMPUTE2 against GoNL reference panel
- Genotypes including regulatory regions
- RNA-seq raw data
- Allele-specific expression data


### Goal
1. Prediction of ASE -- a ASE score 
2. Pathogenicity of predicted ASE -- a pathogenic score (How?)
3. Polygenic risk score (??)


### Methods
- Baseline by normal individuals
- 

### Rare variations 
- Shall we consider Bayesian methods?

imblanced allele:
- ORF vs non-ORF
- inronic vs coding
- 


**Features to pick up**
    - Each SNPs 
    - gene upstream up to 1M bp(enhancer with length ranging from 50-1500bp)
    - gene upstream up to 20-2k bp(silencer with length ranging from )
    - Annotation information(?)
    - Alternative splicing site mutations (??)
    - Expression QTLs

gene-wised: 
    - all
    - gene panel for specific disease

co-expression network-wised


**X-chromosome inactivation (XCI)**
    Genes on X chromosome are monoallelic expressed. Then one can fetch the 
    baseline of expression


**None-sense mediated decay**
Mutation in one allele results in a premature stop codon, as a consequence, 
transcripts from this allele are degrated by nonsense-mediated decay mechanism.

### Fetch sequence by bedtools(compile it into a shared library?)
基因上游调控区的序列
基因下游调控区的序列
外显子序列
内含子序列
序列的生物学意义：lncRNA/coding/pseudogene/

需要考虑所有transcripts，还是考虑最长的一个？



### Implementation (for more details, go to README.md in the project dir)
1. Input a sequence
2. Transform the sequence into a HMM model or a gene model
3. 


allele_specific_expression:
   |-- ababoost.py
   |-- main.py
   |-- predictor.py
   |-- preprocessor.py
   |-- scheduler.py
   |-- trainer.py
   |-- utls.py
   |-- README.md


