import numpy as np
import tensorflow as tf


def _build_conv(x, x_shape, rec_shape, latent_dim, network_type,
                encoder_conf, decoder_conf, hiddens):

    num_channels = x.shape[-1]

    with tf.name_scope('encoder'):
        print('\n===== VAE-{} ===== '.format(network_type))
        print('input: {}'.format(x.shape))
        print('\n===== encoder')
        for n, (filters, kernel_size, stride) in enumerate(encoder_conf):
            x = tf.contrib.layers.conv2d(inputs=x,
                                         num_outputs=filters,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         activation_fn=tf.nn.relu)
            print('conv {} => {}'.format(n, x.shape))

        encoder_last_conv_shape = x.shape

        flat_encoder = tf.layers.flatten(x)
        print('flatten => {}'.format(flat_encoder.shape))

        fc = tf.contrib.layers.fully_connected(flat_encoder, hiddens, activation_fn=tf.nn.relu)
        print('fc [ReLu] => {}'.format(fc.shape))
        print('\n===== latent-{}'.format(latent_dim
                                         ))
        mus = tf.contrib.layers.fully_connected(fc, latent_dim, activation_fn=None)
        logvars = tf.contrib.layers.fully_connected(fc, latent_dim, activation_fn=None)
        print('mus: {}'.format(mus.shape))
        print('logvars: {}'.format(logvars.shape))

        z = mus + tf.exp(0.5 * logvars) * tf.random_normal(tf.shape(mus))

        print('==> z {}'.format(z.shape))
        print('\n===== encoder')

    with tf.name_scope('decoder'):
        fcd = tf.contrib.layers.fully_connected(z, hiddens, activation_fn=tf.nn.relu)
        print('fc [ReLu] => {}'.format(fcd.shape))

        fcd2 = tf.contrib.layers.fully_connected(z, int(np.prod(encoder_last_conv_shape[1:])), activation_fn=None)
        print('fc [None] => {}'.format(fcd2.shape))

        x = tf.reshape(fcd2, shape=rec_shape)

        print('reshape => {}'.format(x.shape))

        for n, (filters, kernel_size, stride) in enumerate(decoder_conf):
            x = tf.contrib.layers.conv2d_transpose(inputs=x,
                                                   num_outputs=filters,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   activation_fn=tf.nn.relu)
            print('conv.T {} => {}'.format(n, x.shape))

        out = tf.contrib.layers.conv2d_transpose(inputs=x,
                                                 num_outputs=num_channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 activation_fn=tf.nn.sigmoid)

    print('output: {}'.format(out.shape))

    assert x_shape[1:] == out.shape[1:], 'input\'s {} and ouput\'s {} shape need to match!'.format(x_shape[1:],
                                                                                                   out.shape[1:])

    return mus, logvars, z, out


def build_network(x, x_shape, latent_dim=10, network_type='atari'):
    print(network_type)

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
    elif network_type == 'dsprites':
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
    else:
        print('network \'{}\' unknown'.format(network_type))
        exit(1)

    return _build_conv(x, x_shape, rec_shape, latent_dim, network_type, encoder_conf, decoder_conf, hiddens)
