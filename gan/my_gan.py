#!/usr/bin/env python3
# -*- utf-8 -*-
"""An naive implementation of GAN
"""

import time
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split


def get_args():
    """Get CLI opttions.
    """
    _parser = argparse.ArgumentParser(description="GAN for ASE project")
    _parser.add_argument("-t", "--training-set", dest="training_set", help="The training set.")

    return _parser


def dtfm2dtst(dtfm, target="target", shuffle=True, batch_size=32):
    """Convert DataFrame to dataset.

    Reference:
        1. https://www.tensorflow.org/tutorials/structured_data/feature_columns#create_an_input_pipeline_using_tfdata
    """
    dtfm = dtfm.copy()
    labels = dtfm.pop(target)
    dtst = tf.data.Dataset.from_tensor_slices((dict(dtfm), labels))

    if shuffle:
        dtst = dtst.shuffle(buffer_size=len(dtfm))

    return dtst.batch(batch_size)

class TrainingDataset():
    """Processing Structured Data.
    """
    def __init__(self):
        self.feature_columns = None

    def make_feature_column(self, **kwargs):
        """Create the feature column.

        Reference:
            1. https://www.tensorflow.org/tutorials/structured_data/feature_columns#choose_which_columns_to_use
        """
        feature_columns = []

        if "numeric_cols" in kwargs:
            numeric_cols = kwargs["numeric_cols"]
            for col in numeric_cols:
                feature_columns.append(feature_column.numeric_column(col))

        if "bucket_cols" in kwargs:
            bucket_cols = kwargs["bucket_cols"]
            for col in bucket_cols:
                feature_columns.append(feature_column.bucketized_column(col))

        if "indicator_cols" in kwargs:
            indicator_cols = kwargs["indicator_cols"]
            for col in indicator_cols:
                feature_columns.append(feature_column.indicator_column(col))

        self.feature_columns = feature_columns

class GAN():
    """An implementation of GAN.
    """

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __init__(self):
        self.epochs = 50
        self.dataset = []
        self.batch_size = 64
        self.latent_dim = 100  # The latent vector length
        self.input_shape = 64 # Number of chosen features
        self.checkpoint = None
        self.generator = None
        self.discriminator = None
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discrimiator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def load_data(self, dtpath):
        """Load training data set.
        """
        self.dataset = pd.read_table(dtpath)

    def make_generator(self, conv_kernal_dim=(5, 5), conv_strides=(2, 2)):
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
        model.add(layers.Conv2DTranspose(32, conv_kernal_dim, strides=conv_strides, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # Second convolutional layer
        model.add(layers.Conv2DTranspose(8, conv_kernal_dim, strides=conv_strides, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # Third convolutional layer
        model.add(layers.Conv2DTranspose(2, conv_kernal_dim, strides=conv_strides, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # Flatten the matrix to fit the input shape
        model.add(layers.Flatten())
        model.add(layers.Dense(self.input_shape, use_bias=False, activation="tanh"))

        return model

    def make_descrimnator(self, conv_kernal_dim=5, conv_strides=2):
        """Make the descriminator model.
        """
        model = Sequential()

        # First convolutional layer
        model.add(layers.Conv1D(32, conv_kernal_dim, strides=conv_strides, padding='same', input_shape=[self.input_shape, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        # Second convolutional layer
        model.add(layers.Conv1D(64, conv_kernal_dim, strides=conv_strides, padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        # Third convolutional layer
        model.add(layers.Conv1D(128, conv_kernal_dim, strides=conv_strides, padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        """Loss function of discriminator.
        """
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

    def generator_loss(self, fake_output):
        """Loss function of generator.
        """
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def save_checkpoint(self, checkpoint_prefix):
        """Save checkpoints.
        """
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discrimiator_optimizer=self.discrimiator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

        self.checkpoint.save(file_prefix=checkpoint_prefix)

    def save_model(self, path):
        """Save the trained model.
        """

    @tf.function
    def train_steps(self, batch):
        """Train steps which could be compiled to accelerate the training.
        """

        noise = tf.random.normal([self.latent_dim, ])
        self.generator = self.make_generator()
        self.discriminator = self.make_descrimnator()

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_sample = self.generator(noise, training=True)

            real_output = self.discriminator(batch, training=True)
            fake_output = self.discriminator(generated_sample, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_gen = gen_tape.gradients(gen_loss, self.generator.trainable_variables)
        gradients_of_disc = disc_tape.gradients(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_gen, self.generator.trainable_variables))
        self.discrimiator_optimizer.apply_gradients(zip(gradients_of_disc, self.discriminator.trainable_variables))


    def train(self):
        """The main entry of GAN to train the model.
        """
        for epoch in range(self.epochs):
            start = time.time()
            for batch in self.dataset:
                self.train_steps(batch)

            if (epoch + 1) / 20 == 0:
                self.save_checkpoint("./checkpoints")

            print("Time for epoch {} is {} sec".format(epoch + 1, time.time()-start))


if __name__ == "__main__":
    OPTS = get_args().parse_args()

    MYGAN = GAN()
    MYGAN.train()
