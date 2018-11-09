import tensorflow as tf
import numpy as np

import logging

from datetime import datetime
from keras.callbacks import TensorBoard, Callback
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

from forkan import weights_path
from forkan.utils import prune_dataset
from forkan.models.bvae.networks import create_bvae_network

class Sigma(Callback):

    def __init__(self, vae):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.vae = vae

    def on_epoch_end(self, epoch, logs=None):

        # run encoder on validation set
        pred = self.vae.encoder.predict(self.validation_data[0])

        # calculate mean of varinaces
        variance_mean = np.mean(np.exp(pred[1]), axis=0)

        self.logger.debug('Current variance: {}'.format(variance_mean))
        self.logger.debug('Current variance: {}'.format(np.exp(pred[1])[0]))


class bVAE(object):

    def __init__(self, input_shape, latent_dim=10, beta=1., network_type='dsprites',
                 debug=False, plot_models=False, print_summaries=False):

        self.logger = logging.getLogger(__name__)

        # define vae specific variables
        self.beta = beta
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # some metadata for weight file names
        self.name = 'bvae'
        self.epochs = None

        # load network
        io, models, zs = create_bvae_network(self, input_shape, latent_dim, network_type=network_type)

        # unpack network
        self.inputs, self.outputs = io
        self.encoder, self.decoder, self.vae = models
        self.z_mean, self.z_log_var, self.z = zs

        # log summaries
        if print_summaries:
            self.logger.info('################### ENCODER ###################')
            self.encoder.summary()
            self.logger.info('################### DECODER ###################')
            self.decoder.summary()
            self.logger.info('###################  MODEL  ###################')
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

        self.logger.info('(beta) VAE with beta = {} and |z| = {}'.format(self.beta, self.latent_dim))

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

    def load(self, weight_path):
        self.logger.info('Using weights: {}'.format(weight_path.split('/')[-1]))
        self.vae.load_weights(weight_path)

    def save(self, dataset_name):
        dest = '{}/{}_{}_b{}_L{}_E{}.h5'.format(weights_path, self.name,dataset_name,
                                            self.beta, self.latent_dim, self.epochs)
        self.vae.save_weights(dest, overwrite=True)

    def encode(self, data):
        return self.encoder.predict(data)

    def decode(self, latents):
        return self.decoder.predict(latents)

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

    def fit(self, train, val=None, epochs=50, batch_size=128, log_sigma=False):

        # save epochs for weight file name
        self.epochs = epochs

        # prune datasets to avoid error
        train = prune_dataset(train, batch_size)

        # define callbacks
        callbacks = []

        tb = TensorBoard(log_dir='/tmp/graph', histogram_freq=0, batch_size=batch_size,
                         write_graph=True, write_images=True, update_freq=1000)

        callbacks.append(tb)

        if log_sigma:
            sc = Sigma(self)
            callbacks.append(sc)

        # train vae
        start = datetime.now()

        if val is not None:
            self.vae.fit(train, epochs=epochs, batch_size=batch_size,
                         callbacks=callbacks, validation_data=(val, None))
        else:
            self.vae.fit(train, epochs=epochs, batch_size=batch_size,
                         callbacks=callbacks)

        elapsed = datetime.now() - start
        self.logger.info('Training took {}.'.format(elapsed))


# simple test using dsprites dataset
if __name__ == '__main__':

    # get image size
    shape = (210, 160, 3)

    vae = bVAE(shape, latent_dim=10, beta=32,
               network_type='atari', print_summaries=True)

