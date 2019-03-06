import json
import numpy as np

import logging

from datetime import datetime
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import Callback
from keras import optimizers

from forkan import model_path
from forkan.common.csv_logger import CSVLogger
from forkan.common.utils import prune_dataset, create_dir, print_dict, clean_dir, copytree
from forkan.models.vae_networks import build_network


class VAECallback(Callback):

    def __init__(self, model, val=None):
        super().__init__()

        self.log = logging.getLogger('vae_CB')

        self.m = model
        self.val = val
        self.epoch = 0
        self.batch = 0

        if self.val is None:
            self.log.warning('No validation set given. Won\'t log to csv.')

    def on_batch_end(self, batch, logs=None):
        self.batch += 1

    def on_epoch_end(self, epoch, logs=None):
        self.m.vae.save_weights('{}/weights.h5'.format(self.m.savepath), overwrite=True)

        # log sigmas
        if self.val is not None:
            mus, logvars = self.m.encode(self.val)[:2]
            logvars = np.mean(logvars, axis=0)
            mus = np.mean(mus, axis=0)

            sigmas = np.exp(0.5 * logvars)

            print('STD.DEV.\n', sigmas)
            print('MU.\n', mus)

            self.m.csv.writeline(
                    datetime.now().isoformat(),
                    self.epoch,
                    self.batch,
                    *mus,
                    *sigmas,
                ) # todo add losses

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
                 optimizer=optimizers.Adam
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

        # some metadata for weight file names
        self.name = name

        if load_from is None:  # fresh vae
            # take care of correct input dim: (BATCH, HEIGHT, WIDTH, CHANNELS)
            # add channel dim if not provided
            if len(input_shape) == 2:
                input_shape = input_shape + (1,)

            self.input_shape = input_shape
            self.latent_dim = latent_dim
            self.network = network
            self.beta = beta
            self.name = name
            self.lr = lr

            if self.name is None or self.name == '':
                self.name = 'default'

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
            params.pop('optimizer')

            with open('{}/params.json'.format(self.savepath), 'w') as outfile:
                json.dump(params, outfile)

            # store from file anyways
            with open('{}from'.format(self.savepath), 'a') as fi:
                fi.write('{}\n'.format(self.savepath.split('/')[-2]))

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

        # encode, decode functions
        self.encode = lambda x: np.asarray(self.encoder.predict(x))
        self.decode = lambda x: np.asarray(self.decoder.predict(x))

        # make sure that input and output shapes match
        assert self.inputs._keras_shape[1:] == self.outputs._keras_shape[1:], 'shape mismatch: in {} out {}'.format(self.inputs._keras_shape[1:],
                                                                                                                    self.outputs._keras_shape[1:])

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
        self.vae.compile(optimizer(lr=self.lr), metrics=['accuracy'])

        csv_header = ['date', '#episode', '#batch']\
                     + ['mu-{}'.format(i) for i in range(self.latent_dim)]\
                     + ['sigma-{}'.format(i) for i in range(self.latent_dim)] # todo add losses
        self.csv = CSVLogger('{}/progress.csv'.format(self.savepath), *csv_header)

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

        self.log.info('(beta) VAE for {} with beta = {} and |z| = {} and learning rate of {}'
                      .format(self.network, self.beta, self.latent_dim, self.lr))

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

        # empty buffers
        self.csv.flush()

        newest = '{}/{}/'.format(self.parent_dir, self.name)
        self.log.info('done training!\ncopying files to {}'.format(newest))

        # create, clean & copy
        create_dir(newest)
        clean_dir(newest, with_files=True)
        copytree(self.savepath, newest)

        # as reference, we leave a file containing the foldername of the just copied model
        with open('{}from'.format(newest), 'a') as fi:
            fi.write('{}\n'.format(self.savepath.split('/')[-2]))


# simple test using dsprites dataset
if __name__ == '__main__':
    from forkan.datasets.dsprites import load_dsprites

    # get image size
    shape = (64, 64, 1)
    (data, _) = load_dsprites('translation', repetitions=10)

    v = VAE(load_from='trans')
    # vae = VAE(input_shape=(64, 64, 1), network='dsprites', name='trans')
    # vae.train(data[:128], num_episodes=3)
