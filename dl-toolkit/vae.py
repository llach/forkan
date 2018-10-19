from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from keras import Model
from keras.initializers import Constant
from keras.callbacks import TensorBoard
from keras.losses import binary_crossentropy, mse
from keras.layers import Dense, Input, Lambda
from keras.utils import plot_model
from keras import backend as K

from utils import load_mnist, prune_dataset, plot_ae_mnist_results


class DenseVAE(object):

    def __init__(self, input_dim, latent_dim, beta=1., debug=False,
                 initial_bias=0.1, plot_models=False):

        # define encoder input layer
        self.inputs = Input(shape=(input_dim,))

        # prepare encoder
        x = Dense(512, activation='relu',
                  bias_initializer=Constant(initial_bias))(self.inputs)
        x = Dense(128, activation='relu',
                  bias_initializer=Constant(initial_bias))(x)

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
        x = Dense(128, activation='relu',
                  bias_initializer=Constant(initial_bias))(self.de_inputs)
        x = Dense(512, activation='relu',
                  bias_initializer=Constant(initial_bias))(x)
        x = Dense(input_dim, activation='sigmoid',
                  bias_initializer=Constant(initial_bias))(x)

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
        self.input_dim = input_dim

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
            K.set_session(
                tf_debug.TensorBoardDebugWrapperSession(tf.Session(), 'localhost:7000'))

    def _sample(self, inputs):

        # unpack input
        z_mean, z_log_var = inputs

        # determine batch size for sampling
        batch = K.shape(z_mean)[0]

        # determine data dimensionality
        dim = K.int_shape(z_mean)[1]

        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))

        # finally, compute the z value
        return z_mean + K.exp(z_log_var) * epsilon # examples had .5 in exp

    def compile(self, optimizer='adam'):

        # define recontruction loss
        re_loss = binary_crossentropy(self.inputs, self.outputs)
        re_loss *= self.input_dim  # why?

        # define kullback leibler divergence
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)

        self.vae_loss = K.mean(re_loss + self.beta * kl_loss)

        # register loss function
        self.vae.add_loss(self.vae_loss)

        # compile entire auto encoder *** is .5 in exp needed?
        self.vae.compile(optimizer, metrics=['accuracy'])

    def fit(self, train, test, epochs=50, batch_size=128):

        # prune datasets to avoid error
        train = prune_dataset(train, batch_size)
        test = prune_dataset(test, batch_size)

        # define tensorboard callback
        tb = TensorBoard(log_dir='/tmp/graph', histogram_freq=0, batch_size=batch_size,
                         write_graph=True, write_images=True, update_freq=1000)

        # train vae
        self.vae.fit(train, epochs=epochs, batch_size=batch_size,
                     validation_data=(test, None), callbacks=[tb])


# simple test using MNIST dataset
if __name__ == '__main__':

    # load dataset
    (x_train, y_train), (x_test, y_test) = load_mnist()

    vae = DenseVAE(x_train.shape[1], 2, beta=1)
    vae.compile()
    vae.fit(x_train, x_test)

    plot_ae_mnist_results((vae.encoder, vae.decoder), (x_test, y_test))
