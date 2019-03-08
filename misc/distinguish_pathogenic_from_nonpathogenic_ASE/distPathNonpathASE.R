
#!/usr/bin/env Rscript
#
##
### author: zhzhang
### e-mail: zhzhang2015@sina.com / zhenghua.zhang217@gmail.com
### data  : 2018.10.31
##
#
################################################################################

# Logging function
lg.info <- function(m, ln='INFO'){ 
    timeStamp <- strftime(Sys.time(), "%Y-%m-%d,%H:%M:%S")
    reporterName <- commandArgs()
    levelName <- ln
    message(timeStamp, ' <', reporterName[4], '> ', levelName, ': ', m)
}

lg.info('=== Start ===')

# Loading necessay library
lg.info('Loading necessay library...')

library(dplyr)
library(tidyr)

# Arrange working dirs
lg.info('Arranging working dirs...')

hmDir <- '/home/umcg-zzhang'
pjDir <- paste0(hmDir, '/projects', '/ASEpredictor')
pjIpDir <- paste0(pjDir, '/inputs')
pjOpDir <- paste0(pjDir, '/outputs')
pjScDir <- paste0(pjDir, '/scripts')
pjMsDir <- paste0(pjDir, '/miscellanies')
pjTpDir <- paste0(pjDir, '/tmp')

# Arrange input and output files
lg.info('Arrange input and output files...')

af <- 0.001  # threshold of allele frequency for filtering

#
##
### PLEASE CHANGE the VALUE of debug WHEN 
##
#

debug <- FALSE  
ext <- if (debug) '_debug' else ''

pjIpFile  <- paste0(pjOpDir, '/biosGavinOverlapCov10', '/biosGavinOverlapCov10Anno.tsv')
pjOpUfFile <- paste0(pjOpDir, '/biosGavinOverlapCov10', '/biosGavinOverlapCov10AnnoUnFiltered.tsv', ext)
pjOpFcFile <- paste0(pjOpDir, '/biosGavinOverlapCov10', '/biosGavinOverlapCov10AnnoFilteredByLog2FC.tsv', ext)
pjOpAfFile <- paste0(pjOpDir, '/biosGavinOverlapCov10', '/biosGavinOverlapCov10AnnoFilteredByAf', as.character(af), '.tsv', ext)

# Loading file
lg.info('Reading file...')
df <- read.csv(pjIpFile, header=1, sep='\t')

# Filtering data
lg.info('Discard records with low coverage...')
covSum <- 10
fil <- which((df$refCountsBios + df$altCountsBios>=covSum) & (df$refCountsBios>0) & (df$altCountsBios>0))
df <- df[fil, ]

# Do exact binomial test
##
### Take the amount of reads supporting reference allele as 'success'
##
#
lg.info('Applying Binomial test...')
df$pVal <- lapply(
    mapply(
        binom.test, df$refCountsBios, 
        df$refCountsBios+df$altCountsBios, 
        SIMPLIFY=FALSE
    ), function(x) return(x$p.value)
)

# Two stratgies for correction. The first is for all p values; second for each site
## Str 1. Considering all p values
lg.info('Overall multiple tests adjust...')
df$FDROverall <- p.adjust(df$pVal, method='fdr')

## Str 2. Considering per variant
#
##
### There is a trick. <<- and ->> operators can cause a search through 
### parent enviroments for an existing definition of the variable being
### assigned
##
lg.info('Variant-wised multiple tests adjust...')
papr <- function(x) {df[rownames(x), 'FDRPerVariant'] <<- p.adjust(x$pVal)}
invisible( by(data=df, INDICES=list(df$chr, df$pos, df$ref, df$alt), 
              FUN=papr, simplify=TRUE)
)

# Add column of log2 fold change
lg.info('Calculating log 2 fold change(altCounts / refCounts)... ')
df$log2FC <- log2(df$altCountsBios / df$refCountsBios)

# Write unfiltered df into drive
lg.info('Write unfiltered file to the drive...')
write.table(x=df, file=pjOpUfFile, sep='\t', row.names=FALSE, quote=FALSE)

# Filtering
fltn <- 'FDRPerVariant'
lg.info(paste0('Filtering by FDR( ', fltn, ')...'))
flt <- which(df$FDRPerVariant <= 0.05)
df <- df[flt, ]

## Filter by AF
fltn <- 'allele frequency(AF)'
lg.info(paste0('Filtering by ', fltn, '...'))
flt <- which(df$gnomad_AF <= af)
dfFltAf <- df[flt, ]

## Write to drive 
lg.info('Write DF filtered by AF to the drive...')
write.table(x=dfFltAf, file=pjOpAfFile, sep='\t', row.names=FALSE, quote=FALSE)

## Filter by log2FC
fltn <- 'log2 of fold change(log2FC, FC = altCounts/refCounts)'
lg.info(paste0('Filtering by ', fltn, '...'))
flt <- which((df$log2FC >= 1) | (df$log2FC <= -1))
dfFltFC <- df[flt, ]

## Write to drive 
lg.info('Write DF filtered by log2FC to the drive...')
write.table(x=dfFltFC, file=pjOpFcFile, sep='\t', row.names=FALSE, quote=FALSE)

# set.seed(1234)
# df <- data.frame(
#     ref=rbinom(10, 100, 0.5), alt=rbinom(10, 100, 0.9), 
#     group=sample(1:2,20, rep=T), gender=sample(1:2, 20, rep=T)
# )

# df$pVal <- lapply(mapply(binom.test, df$ref, df$ref+df$alt, SIMPLIFY=FALSE), function(x) x$p.value)

# invisible( by(df, list(df$group, df$gender), 
#          function(x) df[rownames(x), 'pAdj']<<-p.adjust(x$pVal), 
#          simplify=TRUE)
# )

# df
