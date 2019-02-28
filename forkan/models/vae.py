import json
import numpy as np

import logging

from datetime import datetime
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import Callback

from forkan import model_path
from forkan.common.utils import prune_dataset, create_dir, print_dict
from forkan.models.vae_networks import build_network


"""
COPy after training
csv stuff

"""

class VAECallback(Callback):

    def __init__(self, model, val=None):
        super().__init__()

        self.m = model
        self.val = val
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.m.vae.save_weights('{}/weights.h5'.format(self.m.savepath), overwrite=True)

        # log sigmas
        if self.val is not None:
            logvars = np.mean(self.m.encode(self.val)[1], axis=0)
            sigmas = np.exp(0.5 * logvars)
            print('STD.DEV.\n', sigmas)

        self.epoch += 1


class VAE(object):

    def __init__(self,
                 input_shape=None,
                 latent_dim=10,
                 beta=1.,
                 network='dsprites',
                 name='vae',
                 plot_models=False,
                 lr=1e-3,
                 load_from=None,
                 sess=None, # might not be needed, depending on how keras handles multiple sessions
                 optimizer='adam'
                 ):
        """

        Basic implementation af a Variational Auto Encoder using Keras.
        A beta as weight for the KL in the loss term can also be given.


        Parameters
        ----------

        input_shape : tuple
            shape of desired input as tuple

        latent_dim : int
            number of nodes in bottleneck layer aka size of latent dimesion

        network : str
            string identifier for network architecture as defined in 'networks.py'

        beta : float
            weighting of the KL divergence in the loss

        name : str
            descriptive name of this model
        """

        self.log = logging.getLogger('vae')

        # define vae specific variables
        self.beta = beta
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # some metadata for weight file names
        self.name = name

        if load_from is None:  # fresh vae
            # take care of correct input dim: (BATCH, HEIGHT, WIDTH, CHANNELS)
            # add channel dim if not provided
            if len(input_shape) == 2:
                input_shape = input_shape + (1,)

            self.latent_dim = latent_dim
            self.network = network
            self.beta = beta
            self.name = name
            self.lr = lr

            self.savename = '{}-b{}-lat{}-lr{}-{}'.format(name, beta, latent_dim, lr,
                                                          datetime.now().strftime('%Y-%m-%dT%H:%M'))
            self.parent_dir = '{}vae-{}'.format(model_path, network)
            self.savepath = '{}vae-{}/{}/'.format(model_path, network, self.savename)
            create_dir(self.savepath)

            self.log.info('storing files under {}'.format(self.savepath))

            params = locals()
            params.pop('self')
            params.pop('sess')
            params.pop('load_from')

            with open('{}/params.json'.format(self.savepath), 'w') as outfile:
                json.dump(params, outfile)
        else:  # load old parameter

            self.savename = load_from
            self.parent_dir = '{}vae-{}'.format(model_path, network)
            self.savepath = '{}vae-{}/{}/'.format(model_path, network, self.savename)

            self.log.info('loading model and parameters from {}'.format(self.savepath))

            try:
                with open('{}/params.json'.format(self.savepath), 'r') as infile:
                    params = json.load(infile)

                for k, v in params.items():
                    setattr(self, k, v)

            except Exception as e:
                self.log.critical('loading {}/params.json failed!\n{}'.format(self.savepath, e))
                exit(0)

        # load network
        io, models, zs = build_network(self.input_shape, self.latent_dim, network=network)

        # unpack network
        self.inputs, self.outputs = io
        self.encoder, self.decoder, self.vae = models
        self.z_mean, self.z_log_var, self.z = zs

        # make sure that input and output shapes match
        assert self.inputs._keras_shape[1:] == self.outputs._keras_shape[1:]

        if load_from is not None:
            self.log.info('restoring graph ... ')
            self.vae.load_weights('{}/weights.h5'.format(self.savepath))
            self.log.info('done!')

        self.log.info('VAE has parameters:')
        print_dict(params, lo=self.log)

        # define recontruction loss
        re_loss = binary_crossentropy(K.flatten(self.inputs), K.flatten(self.outputs))
        re_loss *= self.input_shape[1] ** 2  # dont square, use correct dims

        # define kullback leibler divergence
        self.kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        self.kl_loss = -0.5 * K.sum(self.kl_loss, axis=-1)
        self.vae_loss = K.mean(re_loss + self.beta * self.kl_loss)

        # register loss function
        self.vae.add_loss(self.vae_loss)

        # compile entire auto encoder
        self.vae.compile(optimizer, metrics=['accuracy'])

        # csv_header = ['date', '#episode', '#batch']#, 'loss', 'kl-loss'] #+ ['z{}-kl'.format(i) for i in range(self.latent_dim)]
        # self.csv = CSVLogger('{}/progress.csv'.format(self.savepath), *csv_header)

        # log summaries
        self.log.info('ENCODER')
        self.encoder.summary()
        self.log.info('DECODER')
        self.decoder.summary()
        self.log.info(' MODEL')
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

        self.log.info('(beta) VAE for {} with beta = {} and |z| = {}'.format(self.network, self.beta, self.latent_dim))

    def encode(self, data):
        """ Encode input. Returns [z_mean, z_log_var, z]. """
        return self.encoder.predict(data)

    def decode(self, latents):
        """ Reconstructs sample from latent space. """
        return self.decoder.predict(latents)

    def train(self, data, val=None, num_episodes=50, batch_size=128):
        """
        Trains VAE on given dataset. Validation set can be given, which
        will be passed down the callbacks bound to training.
        """

        # prune datasets to avoid error
        data = prune_dataset(data, batch_size)

        cb = VAECallback(self, data[:min(512, len(data)-1),...])

        # train vae
        start = datetime.now()

        if val is not None:
            self.vae.fit(data, epochs=num_episodes, batch_size=batch_size, validation_data=(val, None))
        else:
            self.vae.fit(data, epochs=num_episodes, batch_size=batch_size, callbacks=[cb])

        elapsed = datetime.now() - start
        self.log.info('Training took {}.'.format(elapsed))


# simple test using dsprites dataset
if __name__ == '__main__':
    from forkan.datasets.dsprites import load_dsprites

    # get image size
    shape = (64, 64, 1)
    (data, _) = load_dsprites('translation', repetitions=10)

    vae = VAE(input_shape=(64, 64, 1), network='dsprites', name='trans')
    vae.train(data[:1024])
