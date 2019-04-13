import logging

import numpy as np
import tensorflow as tf

log = logging.getLogger('vae-nets')


def _build_encoder(x, encoder_conf, network_type, latent_dim, hiddens):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        log.info('===== {}-network ===== '.format(network_type))
        log.info('input: {}'.format(x.shape))
        log.info('===== encoder')
        for n, (filters, kernel_size, stride) in enumerate(encoder_conf):
            x = tf.contrib.layers.conv2d(inputs=x,
                                         num_outputs=filters,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         activation_fn=tf.nn.relu)
            log.info('conv {} => {}'.format(n, x.shape))

        encoder_last_conv_shape = x.shape

        flat_encoder = tf.layers.flatten(x)
        log.info('flatten => {}'.format(flat_encoder.shape))

        fc = tf.contrib.layers.fully_connected(flat_encoder, hiddens, activation_fn=tf.nn.relu)
        log.info('fc [ReLu] => {}'.format(fc.shape))
        log.info('===== latent-{}'.format(latent_dim))

        with tf.variable_scope('mus'):
            mus = tf.contrib.layers.fully_connected(fc, latent_dim, activation_fn=None)
        with tf.variable_scope('logvars'):
            logvars = tf.contrib.layers.fully_connected(fc, latent_dim, activation_fn=None)

        log.info('mus: {}'.format(mus.shape))
        log.info('logvars: {}'.format(logvars.shape))

        z = mus + tf.exp(0.5 * logvars) * tf.random_normal(tf.shape(mus))

        log.info('==> z {}'.format(z.shape))
        log.info('===== encoder')

        return mus, logvars, z, encoder_last_conv_shape


def _build_decoder(decoder_conf, z, hiddens, num_channels, encoder_last_conv_shape, rec_shape,
                   x_shape):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        fcd = tf.contrib.layers.fully_connected(z, hiddens, activation_fn=tf.nn.relu)
        log.info('fc [ReLu] => {}'.format(fcd.shape))

        fcd2 = tf.contrib.layers.fully_connected(fcd, int(np.prod(encoder_last_conv_shape[1:])), activation_fn=None)
        log.info('fc [None] => {}'.format(fcd2.shape))

        x = tf.reshape(fcd2, shape=rec_shape)

        log.info('reshape => {}'.format(x.shape))

        for n, (filters, kernel_size, stride) in enumerate(decoder_conf):
            x = tf.contrib.layers.conv2d_transpose(inputs=x,
                                                   num_outputs=filters,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   activation_fn=tf.nn.relu)
            log.info('conv.T {} => {}'.format(n, x.shape))

        x_hat = tf.contrib.layers.conv2d_transpose(inputs=x,
                                                 num_outputs=num_channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 activation_fn=tf.nn.sigmoid)

    log.info('output: {}'.format(x_hat.shape))
    assert x_shape[1:] == x_hat.shape[1:], 'input\'s {} and ouput\'s {} shape need to match!'.format(x_shape[1:],
                                                                                                   x_hat.shape[1:])

    return x_hat

def build_encoder(x, x_shape, latent_dim=10, network_type='atari'):

    if network_type == 'atari':
        # this must not depend on input tensor, otherwise the decoder graph
        # could not be run independently form the encoder
        rec_shape = (-1, int(x_shape[1] / 4), int(x_shape[2] / 4), 64)

        encoder_conf = zip([32, 64],  # num filter
                           [2, 2],  # kernel size
                           [(2, 2), (2, 2)]) # strides

        hiddens = 512
    elif network_type == 'dsprites' or network_type == 'pendulum':
        # this must not depend on input tensor, otherwise the decoder graph
        # could not be run independently form the encoder
        rec_shape = (-1, int(x_shape[1] / 16), int(x_shape[2] / 16), 64)

        encoder_conf = zip([32, 32, 64, 64], # num filter
                           [4]*4, # kernel size
                           [(2, 2)]*4) # strides

        hiddens = 256
    elif network_type == 'pendulum-mini':
        # this must not depend on input tensor, otherwise the decoder graph
        # could not be run independently form the encoder
        rec_shape = (-1, int(x_shape[1] / 8), int(x_shape[2] / 8), 64)

        encoder_conf = zip([32, 64,], # num filter
                           [4]*2, # kernel size
                           [(2, 2)]*2) # strides

        hiddens = 256
    else:
        log.critical('network \'{}\' unknown'.format(network_type))
        exit(1)

    with tf.variable_scope('vae', reuse=tf.AUTO_REUSE):
         mus, logvars, z, encoder_last_conv_shape = _build_encoder(x, encoder_conf, network_type, latent_dim, hiddens)

    return mus, logvars, z


def build_network(x, x_shape, latent_dim=10, network_type='atari'):

    if network_type == 'atari':
        # this must not depend on input tensor, otherwise the decoder graph
        # could not be run independently form the encoder
        rec_shape = (-1, int(x_shape[1] / 4), int(x_shape[2] / 4), 64)

        encoder_conf = zip([32, 64],  # num filter
                           [2, 2],  # kernel size
                           [(2, 2), (2, 2)]) # strides

        decoder_conf = zip([64, 32],  # num filter
                           [2, 2],  # kernel size
                           [(2, 2), (2, 2)]) # strides

        hiddens = 512
    elif network_type == 'dsprites' or network_type == 'pendulum':
        # this must not depend on input tensor, otherwise the decoder graph
        # could not be run independently form the encoder
        rec_shape = (-1, int(x_shape[1] / 16), int(x_shape[2] / 16), 64)

        encoder_conf = zip([32, 32, 64, 64], # num filter
                           [4]*4, # kernel size
                           [(2, 2)]*4) # strides

        decoder_conf = zip([64, 64, 32, 32],  # num filter
                           [4]*4,  # kernel size
                           [(2, 2)]*4)  # strides

        hiddens = 256
    elif network_type == 'pendulum-mini':
        # this must not depend on input tensor, otherwise the decoder graph
        # could not be run independently form the encoder
        rec_shape = (-1, int(x_shape[1] / 4), int(x_shape[2] / 4), 64)
        print(rec_shape)
        encoder_conf = zip([32, 64],  # num filter
                           [4] * 2,  # kernel size
                           [(2, 2)] * 2)  # strides

        decoder_conf = zip([64, 32],  # num filter
                           [4] * 2,  # kernel size
                           [(2, 2)] * 2)  # strides

        hiddens = 256
    else:
        log.critical('network \'{}\' unknown'.format(network_type))
        exit(1)

    num_channels = x.shape[-1]

    with tf.variable_scope('vae', reuse=tf.AUTO_REUSE):

         mus, logvars, z, encoder_last_conv_shape = _build_encoder(x, encoder_conf, network_type, latent_dim, hiddens)

         x_hat = _build_decoder(decoder_conf, z, hiddens, num_channels, encoder_last_conv_shape, rec_shape, x_shape)

    return mus, logvars, z, x_hat
