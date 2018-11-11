
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
library(ggplot2)
# library(GGally)
# library(gridExtra)
# library(grid)
# library(ggpubr)

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
lg.info('Arranging input and ouput files...')

## Input files
pjIpFile <- paste0(pjOpDir, '/biosGavinOverlapCov10', '/biosGavinOverlapCov10AnnoUnFiltered.tsv')

## Output files
pjOpFile <- paste0(pjOpDir, '/biosGavinOverlapCov10', '/allRefAltCounts.png')

df <- read.csv(pjIpFile, header=1, sep='\t')
df$chr <- as.factor(df$chr)

# Statistical analysis and plots + theme_bw()
p <- ggplot(data=df, aes(x=altCountsBios, y=refCountsBios, color=group)) + theme_bw()
p <- p + geom_point(alpha=0.5)
p <- p + geom_abline(slope=1, intercept=0, linetype='dotted')
p <- p + geom_abline(slope=0.5, intercept=0, linetype='dotted')
p <- p + geom_abline(slope=2, intercept=0, linetype='dotted')
p <- p + labs( title='TOTAL(Filtered)', y='Number of reads of reference allele', x='Number of reads of alternative allele')
p <- p + lims(x=c(1, 6000), y=c(1, 6000))
p <- p + theme(legend.position=c(0.9, 0.9))
ggsave(pjOpFile, device='png', width='840', height='840', units='px')
