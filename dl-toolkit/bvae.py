from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import os

from keras import Model
from datetime import datetime
from keras.initializers import Constant
from keras.callbacks import TensorBoard, Callback
from keras.losses import binary_crossentropy
from keras.layers import (Conv2D, Conv2DTranspose, Dense,
                          Input, Lambda, Flatten, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
from keras import backend as K

from utils import (prune_dataset, load_dsprites, animate_greyscale_dataset,
                   load_dsprites_translational, load_dsprites_duo, load_dsprites_one_fixed)

class Sigma(Callback):

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def on_epoch_end(self, epoch, logs=None):

        # run encoder on validation set
        pred = self.vae.encoder.predict(self.validation_data[0])

        # calculate mean of varinaces
        variance_mean = np.mean(np.exp(pred[1]), axis=0)

        print('Current variance: {}'.format(variance_mean))
        print('Current variance: {}'.format(np.exp(pred[1])[0]))


class bVAE(object):

    def __init__(self, input_shape, latent_dim=10, initial_bias=.1,
                 beta=1., debug=False, plot_models=False):

        # define encoder input layer
        self.inputs = Input(shape=input_shape)

        kernel = 4
        strides = (2, 2)
        filter = 32

        # prepare encoder
        x = Conv2D(filter, kernel, strides=strides, padding='same',
                    data_format='channels_last', activation='relu',
                    bias_initializer=Constant(initial_bias),
                    name='enc-conv-1')(self.inputs)

        x = Conv2D(filter, kernel, strides=strides, padding='same',
                   data_format='channels_last', activation='relu',
                   bias_initializer=Constant(initial_bias),
                   name='enc-conv-2')(x)

        x = Conv2D(filter*2, kernel, strides=strides, padding='same',
                   data_format='channels_last', activation='relu',
                   bias_initializer=Constant(initial_bias),
                   name='enc-conv-3')(x)

        x = Conv2D(filter*2, kernel, strides=strides, padding='same',
                   data_format='channels_last', activation='relu',
                   bias_initializer=Constant(initial_bias),
                   name='enc-conv-4')(x)

        # shape info needed to build decoder model
        conv_shape = K.int_shape(x)

        x = Flatten()(x)
        x = Dense(256, activation='relu', bias_initializer=Constant(initial_bias),
                  name='enc-dense')(x)

        # latent variables means and log(standard deviation)
        # leave the z activations linear!
        self.z_mean = Dense(latent_dim)(x)
        self.z_log_var = Dense(latent_dim)(x)
        self.z = Lambda(self._sample, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])

        # final encoder layer is sampling layer
        self.encoder = Model(self.inputs, [self.z_mean ,self.z_log_var, self.z], name='encoder')

        # define decoder input layer
        self.de_inputs = Input(shape=(latent_dim,))

        # prepare decoder
        # this part is not explicitly given in the paper. it reads just 'reverse order'
        x = Dense(256, activation='relu', bias_initializer=Constant(initial_bias),
                  name='dec-dense')(self.de_inputs)

        x = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='linear',
                  bias_initializer=Constant(initial_bias), name='dec-reshape')(x)

        x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)

        x = Conv2DTranspose(filter*2, kernel, strides=strides, padding='same',
                           data_format='channels_last', activation='relu',
                           bias_initializer=Constant(initial_bias),
                           name='dec-convT-1')(x)

        x = Conv2DTranspose(filter*2, kernel, strides=strides, padding='same',
                           data_format='channels_last', activation='relu',
                           bias_initializer=Constant(initial_bias),
                           name='dec-convT-2')(x)

        x = Conv2DTranspose(filter, kernel, strides=strides, padding='same',
                           data_format='channels_last', activation='relu',
                           bias_initializer=Constant(initial_bias),
                           name='dec-convT-3')(x)

        x = Conv2DTranspose(filter, kernel, strides=strides, padding='same',
                           data_format='channels_last', activation='relu',
                           bias_initializer=Constant(initial_bias),
                           name='dec-convT-4')(x)

        # this one is mainly to normalize outputs and reduce depth to the original one
        x = Conv2DTranspose(1, kernel, strides=1, padding='same',
                           data_format='channels_last', activation='sigmoid',
                           bias_initializer=Constant(initial_bias),
                           name='dec-output')(x)

        # decoder restores input in last layer
        self.decoder = Model(self.de_inputs, x, name='decoder')

        # complete auto encoder
        # the encoder ouput index passed to the decoder MUST mach z.
        # otherwise, no gradient can be computed.
        self.outputs = self.decoder(self.encoder(self.inputs)[2])
        self.vae = Model(self.inputs, self.outputs, name='full-vae')

        # define vae specific variables
        self.beta = beta
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # print summaries
        print('################### ENCODER ###################')
        self.encoder.summary()
        print('################### DECODER ###################')
        self.decoder.summary()
        print('###################  MODEL  ###################')
        self.vae.summary()

        if plot_models:
            plot_model(self.encoder,
                       to_file='encoder.png',
                       show_shapes=True)
            plot_model(self.decoder,
                       to_file='decoder.png',
                       show_shapes=True)
            plot_model(self.vae,
                       to_file='vae.png',
                       show_shapes=True)

        if debug:
            from tensorflow.python import debug as tf_debug
            K.set_session(
                tf_debug.TensorBoardDebugWrapperSession(tf.Session(), 'localhost:7000'))

    def _sample(self, inputs):

        # unpack input
        z_mean, z_log_var = inputs

        # determine batch size for sampling
        batch = K.shape(z_mean)[0]

        # determine data dimensionality
        dim = K.int_shape(z_mean)[1]

        # by default, random_normal has mean=0 and sd=1.0
        epsilon = K.random_normal(shape=(batch, dim))

        # finally, compute the z value
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def compile(self, optimizer='adam'):

        # define recontruction loss
        re_loss = binary_crossentropy(K.flatten(self.inputs), K.flatten(self.outputs))
        re_loss *= self.input_shape[1]**2

        # define kullback leibler divergence
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)

        self.vae_loss = K.mean(re_loss + self.beta * kl_loss)

        # register loss function
        self.vae.add_loss(self.vae_loss)

        # compile entire auto encoder
        self.vae.compile(optimizer, metrics=['accuracy'])

    def fit(self, train, val=None, epochs=50, batch_size=128, savefile=None):

        # prune datasets to avoid error
        train = prune_dataset(train, batch_size)

        # define tensorboard callback
        tb = TensorBoard(log_dir='/tmp/graph', histogram_freq=0, batch_size=batch_size,
                         write_graph=True, write_images=True, update_freq=1000)

        sc = Sigma(self)

        # train vae
        start = time.time()

        if val is not None:
            self.vae.fit(train, epochs=epochs, batch_size=batch_size,
                         callbacks=[tb, sc], validation_data=(val, None))
        else:
            self.vae.fit(train, epochs=epochs, batch_size=batch_size,
                         callbacks=[tb, sc])

        end = time.time()
        print('Training took {}.'.format(end-start))

        if savefile is not None:
            dest = os.environ['HOME'] + '/' + savefile + '.h5'
            self.vae.save_weights(dest, overwrite=True)


# simple test using dsprites dataset
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    # load data
    x_train, _ = load_dsprites_translational(repetitions=5)
    x_val = load_dsprites_one_fixed()

    # get image size
    image_size = x_train.shape[1]

    vae = bVAE((image_size, image_size, 1), latent_dim=5, beta=4)
    vae.compile()
    vae.fit(x_train, val=x_val, epochs=100, savefile=args.save)

