import sys
import logging
import keras.backend as K

from keras import Model
from keras.initializers import Constant
from keras.layers import (Conv2D, Conv2DTranspose, Dense,
                          Input, Lambda, Flatten, Reshape)

logger = logging.getLogger(__name__)

def create_bvae_network(model, input_shape, latent_dim,
                        network_type='dsprites', initial_bias=0.1,):
    
    # define encoder input layer
    inputs = Input(shape=input_shape)

    ############################################
    #####               DSPRITES           #####
    ############################################
    if network_type == 'dsprites':
        kernel = 4
        strides = (2, 2)
        filter = 32

        # prepare encoder
        x = Conv2D(filter, kernel, strides=strides, padding='same',
                   data_format='channels_last', activation='relu',
                   bias_initializer=Constant(initial_bias),
                   name='enc-conv-1')(inputs)

        x = Conv2D(filter, kernel, strides=strides, padding='same',
                   data_format='channels_last', activation='relu',
                   bias_initializer=Constant(initial_bias),
                   name='enc-conv-2')(x)

        x = Conv2D(filter * 2, kernel, strides=strides, padding='same',
                   data_format='channels_last', activation='relu',
                   bias_initializer=Constant(initial_bias),
                   name='enc-conv-3')(x)

        x = Conv2D(filter * 2, kernel, strides=strides, padding='same',
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
        z_mean = Dense(latent_dim)(x)
        z_log_var = Dense(latent_dim)(x)
        z = Lambda(model._sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # final encoder layer is sampling layer
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # define decoder input layer
        de_inputs = Input(shape=(latent_dim,))

        # prepare decoder
        # this part is not explicitly given in the paper. it reads just 'reverse order'
        x = Dense(256, activation='relu', bias_initializer=Constant(initial_bias),
                  name='dec-dense')(de_inputs)

        x = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='linear',
                  bias_initializer=Constant(initial_bias), name='dec-reshape')(x)

        x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)

        x = Conv2DTranspose(filter * 2, kernel, strides=strides, padding='same',
                            data_format='channels_last', activation='relu',
                            bias_initializer=Constant(initial_bias),
                            name='dec-convT-1')(x)

        x = Conv2DTranspose(filter * 2, kernel, strides=strides, padding='same',
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
        decoder = Model(de_inputs, x, name='decoder')

        # complete auto encoder
        # the encoder ouput index passed to the decoder MUST mach z.
        # otherwise, no gradient can be computed.
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='full-vae')

    ############################################
    #####                ATARI             #####
    ############################################
    elif network_type == 'atari':
        kernel = 4
        strides = (2, 2)
        filter = 32

        # prepare encoder
        x = Conv2D(filter, kernel, strides=strides, padding='same',
                   data_format='channels_last', activation='relu',
                   bias_initializer=Constant(initial_bias),
                   name='enc-conv-1')(inputs)

        x = Conv2D(filter, kernel, strides=strides, padding='same',
                   data_format='channels_last', activation='relu',
                   bias_initializer=Constant(initial_bias),
                   name='enc-conv-2')(x)

        x = Conv2D(filter * 2, kernel, strides=strides, padding='same',
                   data_format='channels_last', activation='relu',
                   bias_initializer=Constant(initial_bias),
                   name='enc-conv-3')(x)

        x = Conv2D(filter * 2, kernel, strides=1, padding='same',
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
        z_mean = Dense(latent_dim)(x)
        z_log_var = Dense(latent_dim)(x)
        z = Lambda(model._sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # final encoder layer is sampling layer
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # define decoder input layer
        de_inputs = Input(shape=(latent_dim,))

        # prepare decoder
        # this part is not explicitly given in the paper. it reads just 'reverse order'
        x = Dense(256, activation='relu', bias_initializer=Constant(initial_bias),
                  name='dec-dense')(de_inputs)

        x = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='linear',
                  bias_initializer=Constant(initial_bias), name='dec-reshape')(x)

        x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)

        x = Conv2DTranspose(filter * 2, kernel, strides=1, padding='same',
                            data_format='channels_last', activation='relu',
                            bias_initializer=Constant(initial_bias),
                            name='dec-convT-1')(x)

        x = Conv2DTranspose(filter * 2, kernel, strides=strides, padding='same',
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
        x = Conv2DTranspose(3, kernel, strides=1, padding='same',
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
    else:
        logger.critical('Network type {} does not exist for bVAE'.format(network_type))
        sys.exit(1)


    return (inputs, outputs), (encoder, decoder, vae), (z_mean, z_log_var, z)

