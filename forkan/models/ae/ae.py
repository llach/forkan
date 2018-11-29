import logging

from keras.losses import mse
from forkan.models.ae.networks import create_ae_network
from forkan.datasets.mnist import load_mnist
from forkan.common.utils import prune_dataset


class AE(object):

    def __init__(self, input_shape, latent_dim, network='dense'):

        self.logger = logging.getLogger(__name__)

        # define vae specific variables
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # load network
        io, models, self.z = create_ae_network(input_shape, latent_dim, network=network)

        # unpack network
        self.inputs, self.outputs = io
        self.encoder, self.decoder, self.ae = models

        # log summaries
        self.logger.info('################### ENCODER ###################')
        self.encoder.summary()
        self.logger.info('################### DECODER ###################')
        self.decoder.summary()
        self.logger.info('###################  MODEL  ###################')
        self.ae.summary()

    def compile(self, optimizer='adam'):

        # define recontruction loss
        self.re_loss = mse(self.inputs, self.outputs)

        # register loss function
        self.ae.add_loss(self.re_loss)

        # compile entire auto encoder
        self.ae.compile(optimizer, metrics=['accuracy'])

    def fit(self, train, test, epochs=5, batch_size=128):

        # prune datasets to avoid error
        train = prune_dataset(train, batch_size)
        test = prune_dataset(test, batch_size)

        # train ae
        self.ae.fit(train, epochs=epochs, batch_size=batch_size, validation_data=(test, None))


# simple test using MNIST dataset
if __name__ == '__main__':

    # load dataset
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=True)

    # create auto encoder
    ae = AE(x_train[0].shape, 10)

