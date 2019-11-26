#!/usr/bin/env Rscript
library(ggplot2)
library(optparse)
library(data.table)
library(FactoMineR)
library(factoextra)

parser <- OptionParser(description = "Do mulitple factor analysis (MAF) on training data set")
parser <- add_option(parser, c("-i", "--input-file"), action = "store", type = "character", dest = "input_file", help = "The input file")
parser <- add_option(parser, c("-o", "--output-file"), action = "store", type = "character", dest = "output_file", help = "The output file")

parsed_args <- parse_args2(parser)
args <- parsed_args$args
opts <- parsed_args$options


data(wine)
dim(wine)
str(wine)
res.famd <- FAMD(wine, ncp = 31, graph = FALSE)

print(res.famd$quanti.var)
