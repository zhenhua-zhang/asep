#!/usr/bin/env Rscript
#
# Begin excluding linting
# TODO:
#    1. ndim should be configable by command line options.

# NOTE:
#    1. On cluster (boxy, calculon), please use `R/3.5.1-foss-2015b-bare`
# End excluding linting

library(randomForest)
library(data.table)
library(doParallel)
library(PCAmixdata)
library(optparse)
library(ggplot2)
library(lattice)
library(stringr)
library(caret)


parser <- OptionParser(
    description = "Do PCA and MAF on training data set"
)

## Flags
#
parser <- add_option(
    parser, c("-p", "--draw-pic"),
    action = "store_true", type = "boolean",
    dest = "draw_picture", help = "Whether draw pics and save them into the disk."
)

parser <- add_option(
    parser, c("-s", "--save-res"),
    action = "store_true", type = "boolean",
    dest = "save_res", help = "Whether save the PCAmix analysis results into the disk"
)
#
## Flags

## Options
#
parser <- add_option(
    parser, c("-i", "--input-file"),
    action = "store", type = "character",
    dest = "input_file", help = "The input file"
)

parser <- add_option(
    parser, c("--first-n-pcs"),
    action = "store", type = "integer", default = 3,
    dest = "first_n_pcs", help = "Draw scatter plot of individuals using first n PCs"
)

parser <- add_option(
    parser, c("--first-n-dims"),
    action = "store", type = "integer", default = 80,
    dest = "first_n_dims", help = "How many dims will be kept for the training"
)

parser <- add_option(
    parser, c("--min-group-size"),
    action = "store", type = "integer", default = 5,
    dest = "min_group_size", help = "Minimal group size for variants to be used"
)

parser <- add_option(
    parser, c("--max-group-size"),
    action = "store", type = "integer", default = 10000,
    dest = "max_group_size", help = "Maximum group size for variants to be used"
)

parser <- add_option(
    parser, c("--remove-features"),
    action = "store", type = "character",
    dest = "rmd_features", help = "Features excluded in the analysis, delimited by comma"
)

parser <- add_option(
    parser, c("--quantitative-features"),
    action = "store", type = "character",
    dest = "quant_features", help = "Quantitative features, delimited by comma"
)

parser <- add_option(
    parser, c("--quali-features"),
    action = "store", type = "character",
    dest = "quali_features", help = "Qualitative features, delimited by comma"
)
#
## Options

parsed_args <- parse_args2(parser)
args <- parsed_args$args
opts <- parsed_args$options

save_res <- opts$save_res
draw_picture <- opts$draw_picture

input_file <- opts$input_file
first_n_pcs <- opts$first_n_pcs
first_n_dims <- opts$first_n_dims

min_group_size <- opts$min_group_size
max_group_size <- opts$max_group_size

rmd_features <- opts$rmd_features
quali_features <- opts$quali_features
quanti_features <- opts$quanti_features

if (is.null(save_res)) {
    save_res <- FALSE
}

if (is.null(draw_picture)) {
    draw_picture <- FALSE
}

# variant_by_feature_data_frame
if (is.null(input_file)) {
    stop("The -i/--input-file is required")
}

vbfdf <- fread(
    input_file,
    data.table = FALSE, verbose = FALSE, stringsAsFactors = TRUE
)

vbfdf <- vbfdf[vbfdf$group_size >= min_group_size, ]
vbfdf <- vbfdf[vbfdf$group_size <= max_group_size, ]

features <- colnames(vbfdf)

## removed_features
#
rmd_features <- c(
    "Chrom", "Pos", "Annotype", "motifEName", "FeatureID", "GeneID", "GeneName",
    "CCDS", "Intron", "Exon", "ConsScore", "ConsDetail",
    # New added
    "Type", # Only SNV
    "Length" # The length of SNV is 0, which is the length(SNV) - 1
)
#
## removed_features

## qualitative_feature
#
quali_features <- c(
    "Ref", "Alt", "Consequence", "motifEHIPos", "oAA", "nAA", "Domain",
    "Dst2SplType", "SIFTcat", "PolyPhenCat", "Segway"
)

# variant_by_feature_data_frame_qualitative
vbfdf_quali <- vbfdf[, quali_features]
vbfdf_quali[, "motifEHIPos"] <- factor(vbfdf_quali[, "motifEHIPos"])
#
## qualitative_feature

## quantitative_features
#
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
    "pLI_score" #, "exon_exp"
)

# variant_by_feature_data_frame_quantitative
vbfdf_quanti <- vbfdf[, quanti_features]
vbfdf_quanti[, "gnomAD_AF"] <- as.double(vbfdf_quanti[, "gnomAD_AF"])
vbfdf_quanti[, "dbscSNV-rf_score"] <- as.double(vbfdf_quanti[, "dbscSNV-rf_score"])
#
## quantitative_features

pcamix_res <- PCAmix(
    vbfdf_quanti, vbfdf_quali,
    ndim = 100, rename.level = TRUE, graph = FALSE
)

eigs <- as.data.frame(pcamix_res$eig)
ind_coord <- as.data.frame(pcamix_res$ind$coord)
ind_coord[, "bb_ASE"] <- as.factor(vbfdf[, "bb_ASE"])

if (save_res == TRUE) {
    fwrite(eigs, file = "PCAmixdata_eigenvalues.tsv", sep = "\t", row.names = TRUE)
    fwrite(ind_coord, file = "PCAmixdata_indCoord.tsv", sep = "\t", row.names = TRUE)
}

if (draw_picture == TRUE) {
    g <- ggplot(data = eigs) + theme_bw()
    g <- g + geom_point(aes(x = seq_len(length(eigs$Cumulative)), y = Cumulative))
    g <- g + geom_vline(xintercept = first_n_dims)
    g <- g + geom_hline(yintercept = eigs[first_n_dims, "Cumulative"])
    ggsave("PCAmixdata_eigenvalues.png", width = 20, height = 20, units = "cm")

    ## Draw first n PCs
    #
    bb_ase <- factor(vbfdf[, "bb_ASE"])
    for (x_dim in seq_len(first_n_pcs)) {
        for (y_dim in seq_len(first_n_pcs)) {
            if (x_dim >= y_dim) {
                next
            }

            pcamixdata_pc_plot <- str_glue("PCAmixdata_ind_PC{x_dim}PC{y_dim}.png")
            pcamixdata_pc_main <- str_glue("Observations (ind, PC{y_dim} ~ PC{x_dim})")
            png(pcamixdata_pc_plot, width = 1920, height = 1920)
            plot(
                pcamix_res,
                axes = c(x_dim, y_dim), choice = "ind", label = FALSE,
                coloring.ind = bb_ase, main = pcamixdata_pc_main
            )
            dev.off()
        }
    }
    #
    ## Draw first n PCs

    png("PCAmixdata_allVariables.png", width = 1920, height = 1920)
    plot(
        pcamix_res,
        choice = "sqload", coloring.var = TRUE, leg = TRUE,
        posleg = "topright", main = "All variables", lim.cos2.plot = 0.01
    )
    dev.off()
}

set.seed(1234)  # Set a seed to make the result reproducable.

## Setup training, testing data sets and controls
#
all_samples <- row.names(ind_coord)
testing_sample <- sample(all_samples, size = 1000)
training_sample <- all_samples[!all_samples %in% testing_sample]

training_set <- ind_coord[training_sample, ]
testing_set <- ind_coord[testing_sample, ]

train_control <- trainControl(method = "cv", number = 10)
#
## Setup training, testing data sets and controls


## GLM
#
if (FALSE) {
    glm_cv10_fit <- train(
        form = bb_ASE ~ .,
        method = "glm",
        family = "binomial",
        data = training_set,
        trControl = train_control
    )

    # Variable importance
    rf_cv10_fit_var_imp <- varImp(glm_cv10_fit, scale = FALSE)
    if (draw_picture) {
        png("PCAmixdata_glm_varImp.png", width = 1920, height = 1920)
        plot(rf_cv10_fit_var_imp, top = 20)
        dev.off()
    }

    # Performance evaluation
    glm_pred <- predict(glm_cv10_fit, newdata = testing_set)
    glm_obs_vs_pred <- data.frame(
        obs = testing_set$bb_ASE,
        pred = glm_pred
    )

    glm_confu_mtrx <- confusionMatrix(
        data = glm_obs_vs_pred$pred, reference = glm_obs_vs_pred$obs, positive = "1"
    )
    print(glm_confu_mtrx)
}
#
## GLM

## Random forest
#
if (TRUE) {
    cl <- makePSOCKcluster(5)
    registerDoParallel(cl)
    getDoParWorkers()
    rf_cv10_fit <- train(
        form = bb_ASE ~ .,
        method = "rf",
        data = training_set,
        trControl = train_control
    )
    stopCluster(cl)

    # Variable importance
    rf_cv10_fit_var_imp <- varImp(rf_cv10_fit, scale = FALSE)
    if (draw_picture) {
        png("PCAmixdata_rf_varImp.png", width = 1920, height = 1920)
        plot(glm_cv10_fit_var_imp, top = 20)
        dev.off()
    }

    # Confusion matrix
    rf_pred <- predict(rf_cv10_fit, newdata = testing_set)
    rf_obs_vs_pred <- data.frame(
        obs = testing_set$bb_ASE,
        pred = glm_pred
    )

    rf_confu_mtrx <- confusionMatrix(
        data = rf_obs_vs_pred$pred, reference = rf_obs_vs_pred$obs, positive = "1"
    )
    print(rf_confu_mtrx)
}
#
## Random forest


# Begin excluding linting

# summary_tc <- twoClassSummary(obs_vs_pred, lev = levels(obs_vs_pred$obs))
# print(summary_tc)

# summary_pr <- prSummary(obs_vs_pred, lev = levels(obs_vs_pred$obs))
# print(summary_pr)


# all_samples <- row.names(ind_coord)
# testing_sample <- sample(all_samples, size = 1000)
# training_sample <- all_samples[!all_samples %in% testing_sample]
# 
# training_set <- ind_coord[training_sample, ]
# glm_model <- glm(bb_ASE ~ ., family = binomial(), data = training_set)
# 
# testing_set <- ind_coord[testing_sample, ]
# glm_testing_pred <- predict.glm(glm_model, newdata = testing_set, type = "response")
# 
# testing_set["bb_ASE_prob"] <- glm_testing_pred
# testing_set["bb_ASE_pred"] <- sapply(
#     glm_testing_pred,
#     function(e) {
#         if (e <= 0.3) {
#             return(0)
#         } else {
#             return(1)
#         }
#     }
# )
# 
# ase_set <- testing_set[, c("bb_ASE", "bb_ASE_pred", "bb_ASE_prob")]
# ase_set[, "bb_ASE"] <- factor(ase_set[, "bb_ASE"])
# ase_set[, "bb_ASE_pred"] <- factor(ase_set[, "bb_ASE_pred"])


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

# End excluding linting
