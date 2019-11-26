#!/usr/bin/env Rscript
library(ggplot2)
library(optparse)
library(data.table)
library(PCAmixdata)

parser <- OptionParser(description = "Do mulitple factor analysis (MAF) on training data set")
parser <- add_option(parser, c("-i", "--input-file"), action = "store", type = "character", dest = "input_file", help = "The input file")
parser <- add_option(parser, c("-o", "--output-file"), action = "store", type = "character", dest = "output_file", help = "The output file")

parsed_args <- parse_args2(parser)
args <- parsed_args$args
opts <- parsed_args$options

input_file <- opts$input_file
output_file <- opts$output_file

# variant_by_feature_data_frame
vbfdf <- fread(input_file, data.table = FALSE, verbose = FALSE, stringsAsFactors = TRUE)

# removed_features
rmd_features <- c("Chrom", "Pos", "Annotype", "motifEName", "FeatureID", "GeneID", "GeneName", "CCDS", "Intron", "Exon", "ConsScore", "ConsDetail")

# qualitative_feature
quali_features <- c("Ref", "Alt", "Type", "Consequence", "MotifEHIPos", "oAA", "nAA", "Domain", "Dst2SplType", "SIFTcat", "PolyPhenCat", "Segway")
vbfdf_quali <- vbfdf[, quali_features]  # variant_by_feature_data_frame_qualitative

# quantitative_features
quanti_features <- c(
  "Length", "cHmmTssA", "cHmmTssAFlnk", "cHmmTxFlnk", "cHmmTx", "cHmmTxWk", "cHmmEnhG", "cHmmEnh", "cHmmZnfRpts", "cHmmHet", "cHmmTssBiv",
  "cHmmBivFlnk", "cHmmEnhBiv", "cHmmReprPC", "cHmmReprPCWk", "cHmmQuies", "minDistTSS", "minDistTSE", "Freq100bp", "Rare100bp", "Sngl100bp", "Freq1000bp",
  "Rare1000bp", "Sngl1000bp", "Freq10000bp", "Rare10000bp", "Sngl10000bp", "gnomAD_AF", "Dst2Splice", "TFBS", "TFBSPeaks", "TFBSPeaksMax", "AnnoType",
  "Consequence", "GC", "CpG", "motifECount", "motifEHIPos", "motifEScoreChng", "oAA", "nAA", "cDNApos", "relcDNApos", "CDSpos", "relCDSpos", "protPos",
  "relProtPos", "Domain", "SIFTval", "PolyPhenVal", "priPhCons", "mamPhCons", "verPhCons", "priPhyloP", "mamPhyloP", "verPhyloP", "bStatistic", "targetScan",
  "mirSVR-Score", "mirSVR-E", "mirSVR-Aln", "GerpRS", "GerpRSpval", "GerpN", "GerpS", "tOverlapMotifs", "motifDist", "EncH3K27Ac", "EncH3K4Me1", "EncH3K4Me3",
  "EncExp", "EncNucleo", "EncOCC", "EncOCCombPVal", "EncOCDNasePVal", "EncOCFairePVal", "EncOCpolIIPVal", "EncOCctcfPVal", "EncOCmycPVal", "EncOCDNaseSig",
  "EncOCFaireSig", "EncOCpolIISig", "EncOCctcfSig", "EncOCmycSig", "Grantham", "dbscSNV-ada_score", "dbscSNV-rf_score", "Dist2Mutation", "RawScore", "PHRED",
  "pLI_score", "exon_exp" 
)

vbfdf_quanti <- vbfdf[, quanti_features]  # variant_by_feature_data_frame_quantitative

pcamix_res <- PCAmix(vbfdf_quanti, vbfdf_quali, ndim = 10)

print(pcamix_res)