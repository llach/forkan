import sys
import logging
import numpy as np
import keras.backend as K

from keras import Model
from keras.metrics import binary_crossentropy
from keras.initializers import Constant
from keras.layers import (Conv2D, Conv2DTranspose, Dense, Layer, BatchNormalization,
                          Input, Lambda, Flatten, Reshape)

logger = logging.getLogger(__name__)


class ReconstructionLossLayer(Layer):
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        self.is_placeholder = True
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs[0]
        x_hat = inputs[1]
        xent_loss = self.input_dim * binary_crossentropy(K.flatten(x), K.flatten(x_hat))
        self.add_loss(xent_loss, inputs=inputs)
        return xent_loss


class KLLossLayer(Layer):
    def __init__(self, beta, **kwargs):
        self.is_placeholder = True
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, inputs, **kwargs):
        z_mean = inputs[0]
        z_log_var = inputs[1]

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)

        self.add_loss(self.beta*kl_loss, inputs=inputs)
        return kl_loss


def _sample(inputs):
    """ Sampling from the Gaussians produced by the encoder. """

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


def create_bvae_network(input_shape, latent_dim, beta, encoder_conf, decoder_conf,
                        batch_norm=False, hiddens=256, initial_bias=0.1):

    # define encoder input layer
    vae_input = Input(shape=input_shape, name='encoder-input')

    x = vae_input
    for n, (filters, kernel_size, stride) in enumerate(encoder_conf):

        # prepare encoder
        x = Conv2D(filters, kernel_size, strides=stride, padding='same',
                   data_format='channels_last', activation='relu',
                   bias_initializer=Constant(initial_bias),
                   name='enc-conv-{}'.format(n))(x)
        if batch_norm:
            x = BatchNormalization()(x)

    # shape info needed to build decoder model
    conv_shape = K.int_shape(x)

    x = Flatten()(x)
    x = Dense(hiddens, activation='relu', bias_initializer=Constant(initial_bias),
              name='enc-dense')(x)

    # latent variables means and log(variance)s
    # leave the z activations linear!
    z_mean = Dense(latent_dim, name='mean')(x)
    z_log_var = Dense(latent_dim, name='logvar')(x)
    z = Lambda(_sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # we define the individual loss compnents as layers so we can log them in callbacks
    y_kl = KLLossLayer(beta=beta, name='KLLossLayer')([z_mean, z_log_var])

    # final encoder layer is sampling layer
    encoder = Model(vae_input, [z_mean, z_log_var, z, y_kl], name='encoder')

    # define decoder input layer
    de_inputs = Input(shape=(latent_dim,), name='decoder-input')

    # prepare decoder
    # this part is not explicitly given in the paper. it reads just 'reverse order'
    x = Dense(hiddens, activation='relu', bias_initializer=Constant(initial_bias),
              name='dec-dense')(de_inputs)

    x = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='linear',
              bias_initializer=Constant(initial_bias), name='dec-reshape')(x)

    x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)

    for n, (filters, kernel_size, stride) in enumerate(decoder_conf):

        # prepare decoder
        x = Conv2DTranspose(filters, kernel_size, strides=stride, padding='same',
                   data_format='channels_last', activation='relu',
                   bias_initializer=Constant(initial_bias),
                   name='dec-conv-{}'.format(n))(x)

        if batch_norm:
            x = BatchNormalization()(x)

    # this one is mainly to normalize outputs and reduce depth to the original one
    x_hat = Conv2DTranspose(1, 1, strides=1, padding='same',
                        data_format='channels_last', activation='sigmoid',
                        bias_initializer=Constant(initial_bias),
                        name='dec-output')(x)

    y_ent = ReconstructionLossLayer(input_dim=np.prod(input_shape), name='ReconstructionLossLayer')([vae_input, x_hat])

    # decoder restores input in last layer
    decoder = Model([de_inputs, vae_input], [x_hat, y_ent], name='decoder')

    # complete auto encoder
    # the encoder ouput index passed to the decoder MUST mach z.
    # otherwise, no gradient can be computed.
    vae_output = decoder([encoder(vae_input)[2], vae_input])

    vae = Model(vae_input, vae_output, name='full-vae')

    return (vae_input, x_hat), (encoder, decoder, vae), (z_mean, z_log_var, z)


def build_network(input_shape, latent_dim, beta, batch_norm=False, network='dsprites', initial_bias=0.1):
    ############################################
    #####               DSPRITES           #####
    ############################################
    if network == 'dsprites' or network == 'pendulum':
        hiddens = 256

        encoder_conf = zip([32, 32, 64, 64],  # num filter
                           [4] * 4,  # kernel size
                           [(2, 2)] * 4)  # strides

        decoder_conf = zip([64, 64, 32, 32],  # num filter
                           [4] * 4,  # kernel size
                           [(2, 2)] * 4)  # strides

    ############################################
    #####                ATARI             #####
    ############################################
    elif network == 'atari':
        hiddens = 256

        encoder_conf = zip([32, 64],  # num filter
                           [2, 2],  # kernel size
                           [(2, 2), (2, 2)]) # strides

        decoder_conf = zip([64, 32],  # num filter
                           [2, 2],  # kernel size
                           [(2, 2), (2, 2)]) # strides

    else:
        logger.critical('Network type {} does not exist for bVAE'.format(network))
        sys.exit(1)

    return create_bvae_network(input_shape, latent_dim, beta, encoder_conf, decoder_conf,
                               batch_norm=batch_norm, hiddens=hiddens, initial_bias=initial_bias)
