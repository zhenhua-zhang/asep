#!/usr/bin/env python3
# -*- utf-8 -*-

# Create date: 10th Dec. 2019
# Last update: 11th Dec. 2019
# Version    : 0.1.0
# License    : Do whatever you want.
# Author     : Zhenhua Zhang
# Email      : zhenhua.zhang217@gmail.com

"""An naive implementation of GAN

TODO:
    1. Handle categorical features. One idea is to use the PCAmixdata indCoord results.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import sequential, layers


def get_args():
    """Get CLI opttions.
    """
    _parser = argparse.ArgumentParser(description="GAN for ASE project")
    _parser.add_argument(
        "-i", "--input",
        action="store", type=str, dest="input_file", required=True,
        help="The whole set."
    )

    return _parser


def dtfm2dtst(dtfm: pd.DataFrame, target="target", shuffle=True, batch_size=32):
    """Convert DataFrame to dataset.

    Reference:
        1. https://www.tensorflow.org/tutorials/structured_data/feature_columns#create_an_input_pipeline_using_tfdata
    """
    dtfm = dtfm.copy()
    labels = dtfm.pop(target)
    # nparr = dtfm.to_numpy()
    # nparr = nparr.reshape(*dtfm.shape, 1)
    dtst = tf.data.Dataset.from_tensor_slices((dict(dtfm), labels))

    if shuffle:
        dtst = dtst.shuffle(buffer_size=len(dtfm))

    return dtst.batch(batch_size)


class TrainingSet:
    """Operations on the training set.

    Note:
        1. The spawn instance is lazy, meaning you have to use load_file() to laod the given file.
    """
    def __init__(self, input_file):
        self.input_file = input_file
        self.dtfm = None
        self.dtst = None
        self.dtypes = dict(numeric_cols=[], bucket_cols=[], indicator_cols={})

    def check_dtypes(self, skip=("bb_ASE",)):
        """Check the datatypes for each columns.
        """
        for key, value in self.dtfm.dtypes.to_dict().items():
            if key in skip:
                pass
            elif value in ["float64", "float32", "int64", "int32"]:
                self.dtypes["numeric_cols"].append(key)
            elif value in ["object"]:
                indicators = list(self.dtfm.loc[:, key].unique())
                self.dtypes["indicator_cols"][key] = indicators
                print("Velue: {} is object with {} labels".format(key, len(self.dtfm.loc[:, key].unique())))
            else:
                print("Unkonwn type of columns: {} for {}".format(key, value))

    def load_file(self, **kwargs):
        """Load the given file.
        """
        self.dtfm = pd.read_csv(self.input_file, low_memory=False, **kwargs)
        return self

    def slice_dtfm(self, rows=None, cols=None, mask=None, remove=True, mask_first=True):
        """Slice the DataFrame base on rows, columns, and mask.
        """
        def do_mask(wkdtfm, mask, remove):
            if mask is not None:
                if remove:
                    reverse_mask = "~({})".format(mask)  # reverse of mask
                    wkdtfm.query(reverse_mask, inplace=True)
                else:
                    wkdtfm.query(mask, inplace=True)
            return wkdtfm

        def do_trim(wkdtfm, cols, rows, remove):
            if remove:
                if cols is not None or rows is not None:
                    wkdtfm.drop(index=rows, columns=cols, inplace=True)
            else:
                if rows is None:
                    rows = wkdtfm.index
                if cols is None:
                    cols = wkdtfm.columns
                wkdtfm = wkdtfm.loc[rows, cols]
            return wkdtfm

        if cols:
            cols = [x for x in cols if x in self.dtfm.columns]

        if mask_first:
            self.dtfm = do_mask(self.dtfm, mask=mask, remove=remove)
            self.dtfm = do_trim(self.dtfm, cols=cols, rows=rows, remove=remove)
        else:
            self.dtfm = do_trim(self.dtfm, cols=cols, rows=rows, remove=remove)
            self.dtfm = do_mask(self.dtfm, mask=mask, remove=remove)

        return self

    def trim_nif(self, max_na_ratio=0.6):
        """Remove features) with high NA ratio, aka No Information Feature.
        """
        na_count_table = self.dtfm.isna().sum()
        nr_rows, _ = self.dtfm.shape
        na_freq_table = na_count_table / float(nr_rows)
        dropped_cols = na_freq_table.loc[na_freq_table >= max_na_ratio].index

        self.dtfm = self.dtfm.loc[:, na_freq_table <= max_na_ratio]
        na_percentage = "".join(
            [
                "  {}: {:.3f}%\n".format(x, y*100)
                for x, y in zip(dropped_cols, na_freq_table[dropped_cols])
            ]
        )

        print(
            "No information features because of too many NAs ({}%):".format(max_na_ratio * 100),
            na_percentage,
            sep="\n"
        )

        return self

    def imputer(self, targets=(np.NaN, '.'), imputation_dict=None):
        """A simple imputater based on pandas DataFrame.replace method.
        """
        if imputation_dict is None:
            imputation_dict = {
                'motifEName': 'unknown', 'GeneID': 'unknown', 'GeneName': 'unknown',
                'CCDS': 'unknown', 'Intron': 'unknown', 'Exon': 'unknown', 'ref':
                'N', 'alt': 'N', 'Consequence': 'UNKNOWN', 'GC': 0.42, 'CpG': 0.02,
                'motifECount': 0, 'motifEScoreChng': 0, 'motifEHIPos': 0, 'oAA':
                'unknown', 'nAA': 'unknown', 'cDNApos': 0, 'relcDNApos': 0,
                'CDSpos': 0, 'relCDSpos': 0, 'protPos': 0, 'relProtPos': 0,
                'Domain': 'UD', 'Dst2Splice': 0, 'Dst2SplType': 'unknown',
                'minDistTSS': 5.5, 'minDistTSE': 5.5, 'SIFTcat': 'UD', 'SIFTval': 0,
                'PolyPhenCat': 'unknown', 'PolyPhenVal': 0, 'priPhCons': 0.115,
                'mamPhCons': 0.079, 'verPhCons': 0.094, 'priPhyloP': -0.033,
                'mamPhyloP': -0.038, 'verPhyloP': 0.017, 'bStatistic': 800,
                'targetScan': 0, 'mirSVR-Score': 0, 'mirSVR-E': 0, 'mirSVR-Aln': 0,
                'cHmmTssA': 0.0667, 'cHmmTssAFlnk': 0.0667, 'cHmmTxFlnk': 0.0667,
                'cHmmTx': 0.0667, 'cHmmTxWk': 0.0667, 'cHmmEnhG': 0.0667, 'cHmmEnh':
                0.0667, 'cHmmZnfRpts': 0.0667, 'cHmmHet': 0.667, 'cHmmTssBiv':
                0.667, 'cHmmBivFlnk': 0.0667, 'cHmmEnhBiv': 0.0667, 'cHmmReprPC':
                0.0667, 'cHmmReprPCWk': 0.0667, 'cHmmQuies': 0.0667, 'GerpRS': 0,
                'GerpRSpval': 0, 'GerpN': 1.91, 'GerpS': -0.2, 'TFBS': 0,
                'TFBSPeaks': 0, 'TFBSPeaksMax': 0, 'tOverlapMotifs': 0, 'motifDist':
                0, 'Segway': 'unknown', 'EncH3K27Ac': 0, 'EncH3K4Me1': 0,
                'EncH3K4Me3': 0, 'EncExp': 0, 'EncNucleo': 0, 'EncOCC': 5,
                'EncOCCombPVal': 0, 'EncOCDNasePVal': 0, 'EncOCFairePVal': 0,
                'EncOCpolIIPVal': 0, 'EncOCctcfPVal': 0, 'EncOCmycPVal': 0,
                'EncOCDNaseSig': 0, 'EncOCFaireSig': 0, 'EncOCpolIISig': 0,
                'EncOCctcfSig': 0, 'EncOCmycSig': 0, 'Grantham': 0, 'Dist2Mutation':
                0, 'Freq100bp': 0, 'Rare100bp': 0, 'Sngl100bp': 0, 'Freq1000bp': 0,
                'Rare1000bp': 0, 'Sngl1000bp': 0, 'Freq10000bp': 0, 'Rare10000bp':
                0, 'Sngl10000bp': 0, 'dbscSNV-ada_score': 0, 'dbscSNV-rf_score': 0,
                'gnomAD_AF': 0.0, 'pLI_score': 0.303188,
            }

        for target in targets:
            self.dtfm.replace(target, imputation_dict, inplace=True)

        return self

    def dtfm2dtst(self, **kwargs):
        """Get the dataset transformed from a DataFrame.
        """
        self.dtst = dtfm2dtst(self.dtfm, **kwargs)

        return self

    def process(self):
        """A default way to do the processing.

        Note:
            1. The simplest way to get a ready training set is to use this method.
        """
        self.load_file() \
                .slice_dtfm() \
                .trim_nif(max_na_ratio=0.6) \
                .imputer() \
                .dtfm2dtst(target="bb_ASE")


class GAN():
    """An implementation of GAN.
    """
    checkpoint = None
    generator = None
    discriminator = None
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __init__(self, dataset=None, epochs=50, batch_size=64, latent_dim=100, input_shape=64):
        self.epochs = epochs
        self.dataset = dataset
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.feature_cols = None
        self.gen_opt = tf.keras.optimizers.Adam(1e-4)
        self.disc_opt = tf.keras.optimizers.Adam(1e-4)

    def make_feature_column(self, **kwargs):
        """Create the feature column.

        Reference:
            1. https://www.tensorflow.org/tutorials/structured_data/feature_columns#choose_which_columns_to_use
        """
        feature_columns = []

        for key, values in kwargs.items():
            if key in "numeric_cols":
                for col in values:
                    feature_columns.append(feature_column.numeric_column(col))
            elif key in "bucket_cols":
                for col in values:
                    feature_columns.append(feature_column.bucketized_column(col))
            elif key in "indicator_cols":
                for col in values:
                    indicator = feature_column \
                            .categorical_column_with_vocabulary_list(
                                col, kwargs["indicator_cols"][col]
                            )
                    feature_columns.append(feature_column.indicator_column(indicator))
            else:
                print("Unsupported type of cols...")
                print("For more information, please check: https://www.tensorflow.org/tutorials/structured_data/feature_columns#choose_which_columns_to_use")

        self.feature_cols = feature_columns

    def make_generator(self, convkd=(5, 5), convst=(2, 2)):
        """Make the generator model.
        """
        model = Sequential()

        # Init layer
        model.add(layers.Dense(2*2*64, use_bias=False, input_shape=(self.latent_dim, )))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # Reshape the vector to a 2D matrix
        model.add(layers.Reshape((2, 2, 64)))

        # First convolutional layer
        model.add(layers.Conv2DTranspose(32, convkd, strides=convst, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # Second convolutional layer
        model.add(layers.Conv2DTranspose(8, convkd, strides=convst, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # Third convolutional layer
        model.add(layers.Conv2DTranspose(2, convkd, strides=convst, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # Flatten the matrix to fit the input shape
        model.add(layers.Flatten())
        model.add(layers.Dense(self.input_shape, use_bias=False, activation="tanh"))

        return model

    def make_descrimnator(self):
        """Make the descriminator model.
        """
        model = Sequential()

        if self.feature_cols:
            model.add(layers.DenseFeatures(self.feature_cols))

        # First convolutional layer
        model.add(layers.Dense(32, input_shape=(self.input_shape, 1)))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        # Second convolutional layer
        model.add(layers.Dense(64))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        # Third convolutional layer
        model.add(layers.Dense(128))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(1, activation="tanh"))

        return model

    def discriminator_loss(self, real_output, fake_output):
        """Loss function of discriminator.
        """
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

    def generator_loss(self, fake_output):
        """Loss function of generator.
        """
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def save_checkpoint(self, checkpoint_prefix, checkpoint_dir="./checkpoint"):
        """Save checkpoints.
        """
        if (self.generator is None or self.discriminator is None):
            print("No generator or discriminator is defined. Skipping save_checkpoint.")
        else:
            self.checkpoint = tf.train.Checkpoint(
                generator_optimizer=self.gen_opt,
                discrimiator_optimizer=self.disc_opt,
                generator=self.generator,
                discriminator=self.discriminator
            )

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self.checkpoint.save(file_prefix=os.path.join(checkpoint_dir, checkpoint_prefix))

    def save_model(self, path):
        """Save the trained model.
        """

    @tf.function
    def train_steps(self, train_batch):
        """Train steps which could be compiled to accelerate the training.
        """
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        self.generator = self.make_generator()
        self.discriminator = self.make_descrimnator()

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_sample = self.generator(noise, training=True)

            real_output = self.discriminator(train_batch, training=True)
            fake_output = self.discriminator(generated_sample, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gen_grad = gen_tape.gradients(gen_loss, self.generator.trainable_variables)
        disc_grad = disc_tape.gradients(disc_loss, self.discriminator.trainable_variables)

        self.gen_opt.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
        self.disc_opt.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))

    def train(self):
        """The main entry of GAN to train the model.
        """
        for epoch in range(self.epochs):
            start = time.time()
            for batch in self.dataset:
                self.train_steps(batch[0])

            if (epoch + 1) % 20 == 0:
                self.save_checkpoint("ckpt")

            print("Time for epoch {} is {} sec".format(epoch + 1, time.time()-start))


def main():
    """Main function for the module.
    """

    opts = get_args().parse_args()
    input_file = opts.input_file

    remove_cols = [
        "Chrom", "Pos", "Type", "Length", "AnnoType", "Intron", "GeneID", "FeatureID", "Exon",
        "CCDS", "GeneName", "ConsDetail", "log2FC", "bn_p", "bn_p_adj", "bb_p", "bb_p_adj",
        "group_size", "bn_ASE"
    ]

    trainset = TrainingSet(input_file)
    trainset.load_file(sep="\t", na_values=("", ".", "NA")) \
            .slice_dtfm(mask="group_size < 4 | bb_ASE == 0", cols=remove_cols) \
            .trim_nif() \
            .imputer() \
            .dtfm2dtst(target="bb_ASE", batch_size=64) \
            .check_dtypes(skip=["bb_ASE"])

    input_dtst = trainset.dtst

    gan = GAN(dataset=input_dtst, input_shape=trainset.dtfm.shape[1])
    # gan.make_feature_column(**trainset.dtypes)
    gan.train()


if __name__ == "__main__":
    main()
    sys.exit()
