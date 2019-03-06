import sys
import logging
import keras.backend as K

from keras import Model
from keras.initializers import Constant
from keras.layers import (Conv2D, Conv2DTranspose, Dense,
                          Input, Lambda, Flatten, Reshape)

logger = logging.getLogger(__name__)


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


def create_bvae_network(input_shape, latent_dim, encoder_conf, decoder_conf, hiddens=256, initial_bias=0.1):
    # define encoder input layer
    inputs = Input(shape=input_shape)

    x = inputs
    for n, (filters, kernel_size, stride) in enumerate(encoder_conf):

        # prepare encoder
        x = Conv2D(filters, kernel_size, strides=stride, padding='same',
                   data_format='channels_last', activation='relu',
                   bias_initializer=Constant(initial_bias),
                   name='enc-conv-{}'.format(n))(x)


    # shape info needed to build decoder model
    conv_shape = K.int_shape(x)

    x = Flatten()(x)
    x = Dense(hiddens, activation='relu', bias_initializer=Constant(initial_bias),
              name='enc-dense')(x)

    # latent variables means and log(variance)s
    # leave the z activations linear!
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(_sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # final encoder layer is sampling layer
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # define decoder input layer
    de_inputs = Input(shape=(latent_dim,))

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
                   name='enc-conv-{}'.format(n))(x)


    # this one is mainly to normalize outputs and reduce depth to the original one
    x = Conv2DTranspose(1, 1, strides=1, padding='same',
                        data_format='channels_last', activation='sigmoid',
                        bias_initializer=Constant(initial_bias),
                        name='dec-output')(x)

    # decoder restores input in last layer
    decoder = Model(de_inputs, x, name='decoder')

    # complete auto encoder
    # the encoder ouput index passed to the decoder MUST mach z.
    # otherwise, no gradient can be computed.
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='full-vae')

    return (inputs, outputs), (encoder, decoder, vae), (z_mean, z_log_var, z)


def build_network(input_shape, latent_dim, network='dsprites', initial_bias=0.1):
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

    return create_bvae_network(input_shape, latent_dim, encoder_conf, decoder_conf,
                               hiddens=hiddens, initial_bias=initial_bias)
