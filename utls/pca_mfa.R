#!/usr/bin/env Rscript

# Create date: 2019-Nov-29
# Last update: 2019-Dec-4
# Version    : 0.1.0
# Author     : Zhenhua Zhang
# Email      : zhenhua.zhang217@gmail.com

# nolint start
# TODO:
#    1. When training by RandomForest, it fails to exploiting multiple cores.
#    However, I need a way to speed up the training.

# NOTE:
#    1. On cluster (boxy, calculon), please use `R/3.5.1-foss-2015b-bare`
#    2. Required packages: e1071, ggplot2, randomForest, data.table, PCAmixdata,
#    optparse, stringr, caret, dplyr, tidyr, MLmetrics
#    3. The bb_ASE == 0 is encoded as "NonASE", while the bb_ASE == 1 is encode as
#    "ASE".
# nolint end

loadings <- list(
    "data.table", "doParallel", "PCAmixdata", "MLmetrics",
    "optparse", "stringr", "MLeval", "caret", "dplyr", "tidyr"
)

loaded <- lapply(loadings, library, character.only = TRUE)


parser <- OptionParser(
    description = "Do PCA and MAF on training data set"
)

#<< Flags
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
#>> Flags

#<< Options
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

parser <- add_option(
    parser, c("--pcamix-ndim"),
    action = "store", type = "integer", default = 100,
    dest = "pcamix_ndim", help = "The `ndim` parameters in PCAmix function"
)
#>> Options

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

pcamix_ndim <- opts$pcamix_ndim

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

#<< Function to train GLM/GDM on give data set
trainer <- function(dtfm, fml, tmd, prx, cvm = "cv", cvn = 6, sms = 1000, rsd = 31415) {
    # dtfm -> dataset for training and testing
    # fml -> formula for training
    # tmd -> method for training
    # prx -> prefix for the output ROC curves
    # cvm -> cross-validation methods
    # cvn -> number of cross-validation
    # sms -> sample size for testing
    # rsd -> random seed. default: 31415

    set.seed(rsd) # Set a seed to make the result reproducable.
    all_samples <- row.names(dtfm)
    testing_sample <- sample(all_samples, size = sms)
    training_sample <- all_samples[!all_samples %in% testing_sample]

    training_set <- dtfm[training_sample, ]
    testing_set <- dtfm[testing_sample, ]

    train_control <- trainControl(
        method = cvm, number = cvn, classProbs = TRUE,
        summaryFunction = twoClassSummary, savePredictions = TRUE
    )

    if (tmd == "glm") {
        fitted <- train(
            form = fml, method = "glm", family = "binomial", data = training_set,
            trControl = train_control
        )
    } else if (tmd == "gdm") {
        clusters <- makePSOCKcluster(5)
        registerDoParallel(clusters)
        fitted <- train(
            form = fml, method = "gdm", data = training_set, trControl = train_control
        )
        stopCluster(clusters)
    } else {
        warning("No traning method is given, using glm")
        fitted <- train(
            form = fml, method = "glm", family = "binomial", data = training_set,
            trControl = train_control
        )
    }

    # Variable importance
    cat("\n")
    fitted_varimp <- varImp(fitted, scale = FALSE)
    print(fitted_varimp)
    cat("-------------------------------------------------------------------\n\n")

    pred <- predict(fitted, newdata = testing_set)
    prob <- predict(fitted, newdata = testing_set, type = "prob")
    obs_vs_pred <- data.frame(
        obs = testing_set$bb_ASE, pred = pred, ASE = prob$ASE, NonASE = prob$NonASE
    )

    # Roc curve for the training set
    pdf(str_glue("{prx}_ROC_traingSet.pdf"))
    fitted_eval <- evalm(fitted, positive = "ASE")
    dev.off()

    # ROC curve for the testing set
    pdf(str_glue("{prx}_ROC_testingSet.pdf"))
    fitted_eval <- evalm(obs_vs_pred[c("ASE", "NonASE", "obs")], positive = "ASE")
    dev.off()

    confu_mtrx <- confusionMatrix(
        data = obs_vs_pred$pred, reference = obs_vs_pred$obs, positive = "ASE"
    )

    print(confu_mtrx)
    cat("-------------------------------------------------------------------\n\n")

    lev <- c("NonASE", "ASE")

    summary_df <- defaultSummary(obs_vs_pred, lev = lev)
    print(summary_df)
    cat("-------------------------------------------------------------------\n\n")

    summary_pr <- prSummary(obs_vs_pred, lev = lev)
    print(summary_pr)
    cat("-------------------------------------------------------------------\n\n")

    summary_tc <- twoClassSummary(obs_vs_pred, lev = lev)
    print(summary_tc)
    cat("-------------------------------------------------------------------\n\n")

    return(fitted)
}
#>> Function to train GLM/GDM on give data set

#<< Function to save plots to depict the PCA + MFA transform
plot_pcamix <- function(pm_res, first_n_dims, first_n_pcs, color_ind) {
    # pm_res -> PCAmix results
    # first_n_dims -> pin-point the first n dims
    # first_n_pcs -> draw individuals plots for the first n PCs
    # coloc_ind -> a factor vector to draw color the individuals in the individuals plot

    eigs <- pm_res$eigs
    g <- ggplot(data = eigs) + theme_bw()
    g <- g + geom_point(aes(x = seq_len(length(eigs$Cumulative)), y = Cumulative))
    g <- g + geom_vline(xintercept = first_n_dims)
    g <- g + geom_hline(yintercept = eigs[first_n_dims, "Cumulative"])
    ggsave("PCAmixdata_eigenvalues.png", width = 20, height = 20, units = "cm")

    #<< Draw first n PCs
    for (x_dim in seq_len(first_n_pcs)) {
        for (y_dim in seq_len(first_n_pcs)) {
            if (x_dim >= y_dim) {
                next
            }

            pcamixdata_pc_plot <- str_glue("PCAmixdata_ind_PC{x_dim}PC{y_dim}.png")
            pcamixdata_pc_main <- str_glue("Observations (ind, PC{y_dim} ~ PC{x_dim})")
            png(pcamixdata_pc_plot, width = 1920, height = 1920)
            plot(
                pm_res,
                axes = c(x_dim, y_dim), choice = "ind", label = FALSE,
                coloring.ind = color_ind, main = pcamixdata_pc_main
            )
            dev.off()
        }
    }
    #>> Draw first n PCs

    png("PCAmixdata_allVariables.png", width = 1920, height = 1920)
    plot(
        pm_res,
        choice = "sqload", coloring.var = TRUE, leg = TRUE,
        posleg = "topright", main = "All variables", lim.cos2.plot = 0.01
    )
    dev.off()

    png("PCAmixdata_numericalVariables.png", width = 1920, height = 1920)
    plot(
        pm_res,
        choice = "cor",
        lim.cos2.plot = 0.01,
        main = "Numerical variables (cor)"
    )
    dev.off()

    png("PCAmixdata_levels.png", width = 1920, height = 1920)
    plot(
        pm_res,
        choice = "levels",
        coloring.var = TRUE,
        main = "Levels (levels)",
        lim.cos2.plot = 0.01
    )

    pr_res <- PCArot(pm_res, dim = 10, graph = FALSE)
    png("PCAmixdata_rot_cor.png", width = 1920, height = 1920)
    plot(
         pr_res,
         choice = "cor",
         coloring.var = TRUE,
         leg = TRUE,
         main = "Numerical variables after rotation(cor)"
    )
    dev.off()
}
#>> Function to save plots to depict the PCA + MFA transform


vbfdf <- fread(
    input_file,
    data.table = FALSE, verbose = FALSE, stringsAsFactors = FALSE
)

vbfdf <- vbfdf[vbfdf$group_size >= min_group_size, ]
vbfdf <- vbfdf[vbfdf$group_size <= max_group_size, ]
colnames(vbfdf) <- make.names(colnames(vbfdf))

vbfdf[, "bb_ASE"] <- sapply(
    vbfdf[, "bb_ASE"],
    FUN = function(e) {
        if (e == 0) {
            return("NonASE")
        }
        else if (e == 1) {
            return("ASE")
        }
    }
)

## removed_features
#
rmd_features <- c(
    "Chrom", "Pos", "Annotype", "motifEName", "FeatureID", "GeneID", "GeneName",
    "CCDS", "Intron", "Exon",
    "ConsScore",  # parameters given manually
    "ConsDetail",  # parameters given manually
    "Type", # Only SNV
    "Length" # The length of SNV is 0, which is the length(SNV) - 1
)
#
## removed_features

#<< qualitative_feature
# nolint start
quali_features <- make.names(c("Ref", "Alt", "Consequence", "motifEHIPos", "oAA", "nAA", "Domain", "Dst2SplType", "SIFTcat", "PolyPhenCat", "Segway"))
# nolint end

vbfdf[, "motifEHIPos"] <- as.factor(vbfdf[, "motifEHIPos"])
vbfdf_quali <- vbfdf[, quali_features]  # variant_by_feature_data_frame_qualitative
#>> qualitative_feature

#<< quantitative_features
# nolint start
quanti_features <- make.names(c( "cHmmTssA", "cHmmTssAFlnk", "cHmmTxFlnk", "cHmmTx", "cHmmTxWk", "cHmmEnhG", "cHmmEnh", "cHmmZnfRpts", "cHmmHet", "cHmmTssBiv", "cHmmBivFlnk", "cHmmEnhBiv", "cHmmReprPC", "cHmmReprPCWk", "cHmmQuies", "minDistTSS", "minDistTSE", "Freq100bp", "Rare100bp", "Sngl100bp", "Freq1000bp", "Rare1000bp", "Sngl1000bp", "Freq10000bp", "Rare10000bp", "Sngl10000bp", "gnomAD_AF", "Dst2Splice", "TFBS", "TFBSPeaks", "TFBSPeaksMax", "GC", "CpG", "motifECount", "motifEScoreChng", "cDNApos", "relcDNApos", "CDSpos", "relCDSpos", "protPos", "relProtPos", "SIFTval", "PolyPhenVal", "priPhCons", "mamPhCons", "verPhCons", "priPhyloP", "mamPhyloP", "verPhyloP", "bStatistic", "targetScan", "mirSVR-Score", "mirSVR-E", "mirSVR-Aln", "GerpRS", "GerpRSpval", "GerpN", "GerpS", "EncH3K27Ac", "motifDist", "EncH3K4Me1", "EncH3K4Me3", "EncExp", "EncNucleo", "EncOCC", "EncOCCombPVal", "EncOCDNasePVal", "EncOCFairePVal", "EncOCpolIIPVal", "EncOCctcfPVal", "EncOCmycPVal", "EncOCDNaseSig", "EncOCFaireSig", "EncOCpolIISig", "EncOCctcfSig", "EncOCmycSig", "Grantham", "dbscSNV-ada_score", "tOverlapMotifs", "dbscSNV-rf_score", "Dist2Mutation", "RawScore", "PHRED", "pLI_score")) #, "exon_exp")
# nolint end

vbfdf[, "gnomAD_AF"] <- as.double(vbfdf[, "gnomAD_AF"])
vbfdf[, "dbscSNV.rf_score"] <- as.double(vbfdf[, "dbscSNV.rf_score"])
vbfdf_quanti <- vbfdf[, quanti_features]  # variant_by_feature_data_frame_quantitative
#>> quantitative_features

pcamix_res <- PCAmix(
    vbfdf_quanti, vbfdf_quali,
    ndim = pcamix_ndim, rename.level = TRUE, graph = FALSE
)

eigs <- as.data.frame(pcamix_res$eig)
ind_coord <- as.data.frame(pcamix_res$ind$coord)

#<< Save PCA + MAF results
if (save_res) {
    fwrite(eigs, file = "PCAmixdata_eigenvalues.tsv", sep = "\t", row.names = TRUE)
    fwrite(ind_coord, file = "PCAmixdata_indCoord.tsv", sep = "\t", row.names = TRUE)
}
#>> Save PCA + MAF results

#<< Save plots to show the decomposition results
if (draw_picture) {
    plot_pcamix(
        pm_res = pcamix_res, first_n_dims = first_n_dims,
        first_n_pcs = first_n_pcs, color_ind = as.factor(vbfdf[, "bb_ASE"])
    )
}
#>> Save plots to show the decomposition results

#<< Train the model on PCA + MAF transformed data
ind_coord[, "bb_ASE"] <- vbfdf[, "bb_ASE"]
glm_idcd <- trainer(dtfm = ind_coord, fml = bb_ASE ~ ., tmd = "glm", prx = "glm_idcd")
gdm_idcd <- trainer(dtfm = ind_coord, fml = bb_ASE ~ ., tmd = "gdm", prx = "gdm_idcd")
#>> Train the model on PCA + MAF transformed data

#<< Train the model on raw data
# nolint start
default_names <- make.names(c("motifEName", "GeneID", "GeneName", "CCDS", "Intron", "Exon", "ref", "alt", "Consequence", "GC", "CpG", "motifECount", "motifEScoreChng", "motifEHIPos", "oAA", "nAA", "cDNApos", "relcDNApos", "CDSpos", "relCDSpos", "protPos", "relProtPos", "Domain", "Dst2Splice", "Dst2SplType", "minDistTSS", "minDistTSE", "SIFTcat", "SIFTval", "PolyPhenCat", "PolyPhenVal", "priPhCons", "mamPhCons", "verPhCons", "priPhyloP", "mamPhyloP", "verPhyloP", "bStatistic", "targetScan", "mirSVR-Score", "mirSVR-E", "mirSVR-Aln", "cHmmTssA", "cHmmTssAFlnk", "cHmmTxFlnk", "cHmmTx", "cHmmTxWk", "cHmmEnhG", "cHmmEnh", "cHmmZnfRpts", "cHmmHet", "cHmmTssBiv", "cHmmBivFlnk", "cHmmEnhBiv", "cHmmReprPC", "cHmmReprPCWk", "cHmmQuies", "GerpRS", "GerpRSpval", "GerpN", "GerpS", "TFBS", "TFBSPeaks", "TFBSPeaksMax", "tOverlapMotifs", "motifDist", "Segway", "EncH3K27Ac", "EncH3K4Me1", "EncH3K4Me3", "EncExp", "EncNucleo", "EncOCC", "EncOCCombPVal", "EncOCDNasePVal", "EncOCFairePVal", "EncOCpolIIPVal", "EncOCctcfPVal", "EncOCmycPVal", "EncOCDNaseSig", "EncOCFaireSig", "EncOCpolIISig", "EncOCctcfSig", "EncOCmycSig", "Grantham", "Dist2Mutation", "Freq100bp", "Rare100bp", "Sngl100bp", "Freq1000bp", "Rare1000bp", "Sngl1000bp", "Freq10000bp", "Rare10000bp", "Sngl10000bp", "dbscSNV-ada_score", "dbscSNV-rf_score", "gnomAD_AF", "pLI_score"))
default_values <- list( "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "N", "N", "UNKNOWN", 0.42, 0.02, 0, 0, 0, "unknown", "unknown", 0, 0, 0, 0, 0, 0, "UD", 0, "unknown", 5.5, 5.5, "UD", 0, "unknown", 0, 0.115, 0.079, 0.094, -0.033, -0.038, 0.017, 800, 0, 0, 0, 0, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.667, 0.667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0, 0, 1.91, -0.2, 0, 0, 0, 0, 0, "unknown", 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.303188)
names(default_values) <- default_names
# nolint end

vbfdf_raw <- vbfdf[, c("bb_ASE", quali_features, quanti_features)]
vbfdf_raw <- vbfdf_raw %>% replace_na(default_values)

for (col_name in colnames(vbfdf_raw)) {
    if (typeof(vbfdf_raw[, col_name]) == "character" || col_name == "bb_ASE") {
        vbfdf_raw[, col_name] <- factor(vbfdf_raw[, col_name])
    }
}

glm_raw <- trainer(dtfm = vbfdf_raw, fml = bb_ASE ~ ., tmd = "glm", prx = "glm_raw")
gdm_raw <- trainer(dtfm = vbfdf_raw, fml = bb_ASE ~ ., tmd = "gdm", prx = "gdm_raw")
#>> Train the model on raw data
