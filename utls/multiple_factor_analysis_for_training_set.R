#!/usr/bin/env Rscript
library(ggplot2)
library(optparse)
library(data.table)
library(PCAmixdata)

parser <- OptionParser(
   description = "Do mulitple factor analysis (MAF) on training data set"
)
parser <- add_option(
    parser, c("-i", "--input-file"), action = "store", type = "character",
    dest = "input_file", help = "The input file"
)
parser <- add_option(
    parser, c("-p", "--draw-pic"), action = "store_true", type = "boolean",
    dest = "draw_pic", help = "Whether draw pics."
)
parser <- add_option(
    parser, c("-o", "--output-file"), action = "store", type = "character",
    dest = "output_file", help = "The output file"
)

parsed_args <- parse_args2(parser)
args <- parsed_args$args
opts <- parsed_args$options

input_file <- opts$input_file
output_file <- opts$output_file
draw_pic <- opts$draw_pic

if (is.null(draw_pic)) {
  draw_pic <- FALSE
}

# variant_by_feature_data_frame
if (is.null(input_file)) {
    stop("The -i/--input-file is required")
}

vbfdf <- fread(
    input_file, data.table = FALSE, verbose = FALSE, stringsAsFactors = TRUE
)
vbfdf <- vbfdf[vbfdf$group_size >= 10, ]


features <- colnames(vbfdf)

# removed_features
rmd_features <- c(
    "Chrom", "Pos", "Annotype", "motifEName", "FeatureID", "GeneID", "GeneName",
    "CCDS", "Intron", "Exon", "ConsScore", "ConsDetail",
    # New added
    "Type",  # Only SNV
    "Length" # The length of SNV is 0, which is the length(SNV) - 1
)

vbfdf[, "gnomAD_AF"] <- as.double(vbfdf[, "gnomAD_AF"])

# qualitative_feature
quali_features <- c(
    "Ref", "Alt", "Consequence", "motifEHIPos", "oAA", "nAA", "Domain",
    "Dst2SplType", "SIFTcat", "PolyPhenCat", "bb_ASE"# , "Segway"
)

# variant_by_feature_data_frame_qualitative
vbfdf_quali <- vbfdf[, quali_features]
vbfdf_quali[, "motifEHIPos"] <- factor(vbfdf_quali[, "motifEHIPos"])
vbfdf_quali[, "bb_ASE"] <- factor(vbfdf_quali[, "bb_ASE"])

# quantitative_features
quanti_features <- c(
  "cHmmTssA", "cHmmTssAFlnk", "cHmmTxFlnk", "cHmmTx", "cHmmTxWk", "cHmmEnhG",
  "cHmmEnh", "cHmmZnfRpts", "cHmmHet", "cHmmTssBiv", "cHmmBivFlnk",
  "cHmmEnhBiv", "cHmmReprPC", "cHmmReprPCWk", "cHmmQuies", "minDistTSS",
  "minDistTSE", "Freq100bp", "Rare100bp", "Sngl100bp", "Freq1000bp",
  "Rare1000bp", "Sngl1000bp", "Freq10000bp", "Rare10000bp", "Sngl10000bp",
  "gnomAD_AF", "Dst2Splice", "TFBS", "TFBSPeaks", "TFBSPeaksMax", "GC", "CpG",
  "motifECount", "motifEScoreChng", "cDNApos", "relcDNApos", "CDSpos",
  "relCDSpos", "protPos", "relProtPos", "SIFTval", "PolyPhenVal", "priPhCons",
  "mamPhCons", "verPhCons", "priPhyloP", "mamPhyloP", "verPhyloP", "bStatistic",
  "targetScan", "mirSVR-Score", "mirSVR-E", "mirSVR-Aln", "GerpRS",
  "GerpRSpval", "GerpN", "GerpS", "EncH3K27Ac", "motifDist", "EncH3K4Me1",
  "EncH3K4Me3", "EncExp", "EncNucleo", "EncOCC", "EncOCCombPVal",
  "EncOCDNasePVal", "EncOCFairePVal", "EncOCpolIIPVal", "EncOCctcfPVal",
  "EncOCmycPVal", "EncOCDNaseSig", "EncOCFaireSig", "EncOCpolIISig",
  "EncOCctcfSig", "EncOCmycSig", "Grantham", "dbscSNV-ada_score",
  "tOverlapMotifs", "dbscSNV-rf_score", "Dist2Mutation", "RawScore", "PHRED",
  "pLI_score", "exon_exp"
)

# variant_by_feature_data_frame_quantitative
vbfdf_quanti <- vbfdf[, quanti_features]
vbfdf_quanti[, "dbscSNV-rf_score"] <- as.double(vbfdf_quanti[, "dbscSNV-rf_score"])

vbfdf_quali[is.na(vbfdf_quali)] <- 0
vbfdf_quanti[is.na(vbfdf_quanti)] <- 0

pcamix_res <- PCAmix(
    vbfdf_quanti, vbfdf_quali, ndim = 80, rename.level = TRUE, graph = FALSE
)

summary(pcamix_res)

if (draw_pic == TRUE) {
    png("PCAmixdata_allVariables.png", width = 1920, height = 1920)
    plot(
      pcamix_res, choice = "sqload", coloring.var = TRUE, leg = TRUE,
      posleg = "topright", main = "All variables", lim.cos2.plot = 0.01
    )
    dev.off()

    bb_ase <- vbfdf_quali$bb_ase
    png("PCAmixdata_ind.png", width = 1920, height = 1920)
    plot(
      pcamix_res, axes = c(1, 2), choice = "ind", label = FALSE,
      coloring.ind = bb_ase, main = "Observations (ind)"
    )
    dev.off()
}

# plot(
#   seq_len(length(pcamix_res$eig[, "Cumulative"])),
#   pcamix_res$eig[, "Cumulative"]
# )

# pcarot_res <- PCArot(pcamix_res, dim = 10, graph = FALSE)
# print(pcarot_res$eig)

# png("PCAmixdata_rot_cor.png", width = 1920, height = 1920)
# plot(
#   pcarot_res,
#     choice = "cor",
#     coloring.var = TRUE,
#     leg = TRUE,
#     main = "Numerical variables after rotation(cor)"
# )
# dev.off()

# png("PCAmixdata_numericalVariables.png", width = 1920, height = 1920)
# plot(
#     pcamix_res,
#     choice = "cor",
#     lim.cos2.plot = 0.01,
#     main = "Numerical variables (cor)"
# )
# dev.off()

# png("PCAmixdata_levels.png", width = 1920, height = 1920)
# plot(
#     pcamix_res,
#     choice = "levels",
#     coloring.var = TRUE,
#     main = "Levels (levels)",
#     lim.cos2.plot = 0.01
# )
# dev.off()
