import logging
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from keras import Model
from keras.callbacks import TensorBoard
from keras.losses import  mse
from keras.layers import Dense, Input
from keras import backend as K

from forkan.datasets.mnist import load_mnist
from forkan.utils import prune_dataset


class DenseAE(object):

    def __init__(self, input_dim, latent_dim, beta=1., debug=False):

        self.logger = logging.getLogger(__name__)

        # define encoder input layer
        self.inputs = Input(shape=(input_dim,))

        # prepare encoder
        x = Dense(512, activation='relu')(self.inputs)
        x = Dense(128, activation='relu')(x)

        # latent variables means and log(standard deviation)
        self.z_mean = Dense(latent_dim, activation='sigmoid')(x)
        self.z_log_sd = Dense(latent_dim, activation='sigmoid')(x)
        # self.z = Lambda(self._sample, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_sd])

        # final encoder layer is sampling layer
        self.encoder = Model(self.inputs, self.z_mean, name='encoder')

        # define decoder input layer
        self.de_inputs = Input(shape=(latent_dim,))

        # prepare decoder
        x = Dense(128, activation='relu')(self.de_inputs)
        x = Dense(512, activation='relu')(x)
        x = Dense(input_dim, activation='sigmoid')(x)

        # decoder restores input in last layer
        self.decoder = Model(self.de_inputs, x, name='decoder')

        # complete auto encoder
        self.outputs = self.decoder(self.encoder(self.inputs))
        self.vae = Model(self.inputs, self.outputs, name='full-vae')

        # define vae specific variables
        self.beta = beta
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # log summaries
        self.logger.info('################### ENCODER ###################')
        self.encoder.summary()
        self.logger.info('################### DECODER ###################')
        self.decoder.summary()
        self.logger.info('###################  MODEL  ###################')
        self.vae.summary()

        if debug:
            K.set_session(
                tf_debug.TensorBoardDebugWrapperSession(tf.Session(), 'localhost:7000'))

    def compile(self, optimizer='adam'):

        # define recontruction loss
        self.re_loss = mse(self.inputs, self.outputs)

        # register loss function
        self.vae.add_loss(self.re_loss)

        # compile entire auto encoder
        self.vae.compile(optimizer, metrics=['accuracy'])

    def fit(self, train, test, epochs=5, batch_size=128):

        # prune datasets to avoid error
        train = prune_dataset(train, batch_size)
        test = prune_dataset(test, batch_size)

        # define tensorboard callback
        tb = TensorBoard(log_dir='/tmp/graph', histogram_freq=0,
          write_graph=True, write_images=True)

        # train vae
        self.vae.fit(train, epochs=epochs, batch_size=batch_size, validation_data=(test, None))


# simple test using MNIST dataset
if __name__ == '__main__':

    # load dataset
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=True)

    ae = DenseAE(x_train.shape[1], 10)
    ae.compile()
    ae.fit(x_train, x_test)
