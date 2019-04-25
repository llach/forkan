import datetime
import json
import logging
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tabulate import tabulate

from forkan import model_path
from forkan.common import CSVLogger
from forkan.common.tf_utils import scalar_summary
from forkan.common.utils import print_dict, create_dir, clean_dir, copytree
from forkan.models.vae_networks import build_network, build_encoder


class VAE(object):

    def __init__(self, input_shape=None, name='default', network='atari', latent_dim=20, beta=1.0, lr=1e-4,
                 load_from=None, session=None, optimizer=tf.train.AdamOptimizer, with_opt=True, tensorboard=False):

        if input_shape is None:
            assert load_from is not None, 'input shape need to be given if no model is loaded'

        self.log = logging.getLogger('vae')

        if load_from is None: # fresh vae
            # take care of correct input dim: (BATCH, HEIGHT, WIDTH, CHANNELS)
            # add channel dim if not provided
            if len(input_shape) == 2:
                input_shape = input_shape + (1,)

            self.latent_dim = latent_dim
            self.network = network
            self.beta = beta
            self.name = name
            self.lr = lr

            # add batch dim
            self.input_shape = (None,) + input_shape

            self.savename = '{}-b{}-lat{}-lr{}-{}'.format(name, beta, latent_dim, lr,
                                                          datetime.datetime.now().strftime('%Y-%m-%dT%H:%M'))
            self.parent_dir = '{}vae-{}'.format(model_path, network)
            self.savepath = '{}vae-{}/{}/'.format(model_path, network, self.savename)
            create_dir(self.savepath)

            self.log.info('storing files under {}'.format(self.savepath))

            params = locals()
            params.pop('self')
            params.pop('optimizer')
            params.pop('session')

            with open('{}/params.json'.format(self.savepath), 'w') as outfile:
                json.dump(params, outfile)
        else: # load old parameter

            self.savename = load_from
            self.parent_dir = '{}vae-{}'.format(model_path, network)
            self.savepath = '{}vae-{}/{}/'.format(model_path, network, self.savename)

            self.log.info('loading model and parameters from {}'.format(self.savepath))

            try:
                with open('{}/params.json'.format(self.savepath), 'r') as infile:
                    params = json.load(infile)

                for k, v in params.items():
                    setattr(self, k, v)

                # add batch dim
                self.input_shape = (None,) + tuple(self.input_shape)
            except Exception as e:
                self.log.critical('loading {}/params.json failed!\n{}'.format(self.savepath, e))
                exit(0)

        # store number of channels
        self.num_channels = self.input_shape[-1]

        self.tb = tensorboard

        with tf.variable_scope('input-ph'):
            self._input = tf.placeholder(tf.float32, shape=self.input_shape, name='input')

        """ TF Graph setup """
        self.mus, self.logvars, self.z, self._output = \
            build_network(self._input, self.input_shape, latent_dim=self.latent_dim, network_type=self.network)
        print('\n')

        """ Loss """
        # Loss
        # Reconstruction loss
        self.re_loss = K.binary_crossentropy(K.flatten(self._input), K.flatten(self._output))
        self.re_loss *= self.input_shape[1] ** 2  # dont square, use correct dims

        # define kullback leibler divergence
        self.kl_loss = 1 + self.logvars - K.square(self.mus) - K.exp(self.logvars)
        self.kl_loss = -0.5 * K.mean(self.kl_loss, axis=0)
        self.vae_loss = K.mean(self.re_loss + self.beta * K.sum(self.kl_loss))

        # create optimizer
        if with_opt:
            self.train_op = optimizer(learning_rate=self.lr).minimize(self.vae_loss)

        """ TF setup """
        self.s = session if session is not None else tf.Session()
        tf.global_variables_initializer().run(session=self.s)

        # Saver objects handles writing and reading protobuf weight files
        self.saver = tf.train.Saver(var_list=tf.all_variables())

        if load_from is not None:
            self.log.info('restoring graph ... ')
            self.saver.restore(self.s, '{}'.format(self.savepath))
            self.log.info('done!')

        self.log.info('VAE has parameters:')
        print_dict(params, lo=self.log)

        if self.tb:
            self._tensorboard_setup()

        csv_header = ['date', '#episode', '#batch', 'rec-loss', 'kl-loss'] +\
                     ['z{}-kl'.format(i) for i in range(self.latent_dim)]
        self.csv = CSVLogger('{}/progress.csv'.format(self.savepath), *csv_header)

    def __del__(self):
        """ cleanup after object finalization """

        # close tf.Session
        if hasattr(self, 's'):
           self.s.close()

    def stack_encoder(self, inputs):
        mus = []
        for inx in inputs:
            m, _, _ = build_encoder(inx, self.input_shape, latent_dim=self.latent_dim, network_type=self.network)
            mus.append(m)
        return mus

    def _save(self, suffix=''):
        """ Saves current weights """
        self.log.info('saving weights')
        self.saver.save(self.s, f'{self.savepath}{suffix}')

    def _load(self, suffix=''):
        """ Saves weights from suffix """
        self.saver.restore(self.s, f'{self.savepath}{suffix}')

    def _tensorboard_setup(self):
        """ Tensorboard (TB) setup """

        with tf.variable_scope('{}-ph'.format(self.name)):
            self.bps_ph = tf.placeholder(tf.int32, (), name='batches-per-second')
            self.ep_ph = tf.placeholder(tf.int32, (), name='episode')

        scalar_summary('batches-per-second', self.bps_ph)
        scalar_summary('episode', self.ep_ph)

        mu_mean = tf.reduce_mean(self.mus, axis=0)
        vars_mean = tf.reduce_mean(tf.exp(0.5 * self.logvars), axis=0)

        with tf.variable_scope('loss'):
            scalar_summary('reconstruction-loss', self.re_loss)
            scalar_summary('total-loss', self.vae_loss)
            scalar_summary('kl-loss', self.kl_loss)

        with tf.variable_scope('zj_mu'):
            for i in range(self.latent_dim):
                scalar_summary('z{}-mu'.format(i), mu_mean[i])

        with tf.variable_scope('zj_var'):
            for i in range(self.latent_dim):
                scalar_summary('z{}-var'.format(i), vars_mean[i])

        self.merge_op = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter('/Users/llach/vae',
                                            graph=tf.get_default_graph())
            
    def _preprocess_batch(self, batch, norm_fac=None):
        """ preprocesses batch """

        if norm_fac is not None:
            batch = batch * norm_fac

        assert np.max(batch) <= 1, f'normalise input first!, max is {np.max(batch)}, norm_fac {norm_fac}'

        if len(batch.shape) != 4:
            """ completing batch shape if some dimesions are missing """
            # grayscale, one sample
            if len(batch.shape) == 2:
                batch = np.expand_dims(np.expand_dims(batch, axis=-1), axis=0)
            # either  batch of grascale or single multichannel image
            elif len(batch.shape) == 3:
                if batch.shape == self.input_shape[1:]:  # single frame
                    batch = np.expand_dims(batch, axis=0)
                else:  # batch of grayscale
                    batch = np.expand_dims(batch, axis=-1)

        assert len(batch.shape) == 4, 'batch shape mismatch'

        return batch

    def reconstruct(self, batch):
        """ returns reconstructions from batch of frames """
        return self.decode(self.encode_and_sample(batch)[-1])

    def encode(self, batch, norm_fac=None):
        """ encodes frame(s) """

        batch = self._preprocess_batch(batch, norm_fac)
        return self.s.run([self.mus, self.logvars], feed_dict={self._input: batch})

    def encode_and_sample(self, batch):
        """ encodes frame(s) and samples from dists """

        batch = self._preprocess_batch(batch)
        return self.s.run([self.mus, self.logvars, self.z], feed_dict={self._input: batch})

    def decode(self, zs):
        """ dcodes batch of latent representations """

        if len(zs.shape) == 1:
            zs = np.expand_dims(zs, 0)

        assert len(zs.shape) == 2, 'z batch shape mismatch'
        assert zs.shape[-1] == self.latent_dim, 'vae has latent space of {}, got {}'.format(self.latent_dim, zs.shape[-1])

        return self.s.run(self._output, feed_dict={self.z: zs})

    def train(self, dataset, batch_size=32, num_episodes=30, print_freq=10):
        num_samples = len(dataset)

        assert np.max(dataset) <= 1, 'provide normalized dataset!'

        # some sanity checks
        dataset = self._preprocess_batch(dataset)

        self.log.info('Training on {} samples for {} episodes.'.format(num_samples, num_episodes))
        tstart = time.time()
        nb = 1

        # rollout N episodes
        for ep in range(num_episodes):

            # shuffle dataset
            np.random.shuffle(dataset)

            for n, idx in enumerate(np.arange(0, num_samples, batch_size)):
                bps = max(int(nb / (time.time() - tstart)), 1)
                x = dataset[idx:min(idx+batch_size, num_samples), ...]
                if self.tb:
                    _, suma, loss, re_loss, kl_losses = self.s.run([self.train_op, self.merge_op, self.vae_loss,
                                                                  self.re_loss, self.kl_loss],
                                                                 feed_dict={self._input: x,
                                                                            self.bps_ph: bps,
                                                                            self.ep_ph: ep
                                                                            })
                    self.writer.add_summary(suma, nb)
                else:
                    _, loss, re_loss, kl_losses = self.s.run([self.train_op, self.vae_loss, self.re_loss, self.kl_loss],
                                                           feed_dict={self._input: x})

                # mean losses
                re_loss = np.mean(re_loss)
                kl_loss = self.beta * np.sum(kl_losses)

                # increase batch counter
                nb += 1

                self.csv.writeline(
                    datetime.datetime.now().isoformat(),
                    ep,
                    nb,
                    re_loss,
                    kl_loss,
                    *kl_losses
                )

                if n % print_freq == 0 and print_freq is not -1:

                    total_batches = (num_samples // batch_size) * num_episodes

                    perc = ((nb) / total_batches) * 100
                    steps2go = total_batches - nb
                    secs2go = steps2go / bps
                    min2go = secs2go / 60

                    hrs = int(min2go // 60)
                    mins = int(min2go) % 60

                    tab = tabulate([
                        ['name', f'{self.name}-b{self.beta}'],
                        ['episode', ep],
                        ['batch', n],
                        ['bps', bps],
                        ['rec-loss', re_loss],
                        ['kl-loss', kl_loss],
                        ['ETA', '{}h {}min'.format(hrs, mins)],
                        ['done', '{}%'.format(int(perc))],
                    ])

                    print('\n{}'.format(tab))

            self._save()

        newest = '{}/{}/'.format(self.parent_dir, self.name)
        self.log.info('done training!\ncopying files to {}'.format(newest))

        # create, clean & copy
        create_dir(newest)
        clean_dir(newest, with_files=True)
        copytree(self.savepath, newest)

        # as reference, we leave a file containing the foldername of the just copied model
        with open('{}from'.format(newest), 'a') as fi:
            fi.write('{}\n'.format(self.savepath.split('/')[-2]))


if __name__ == '__main__':
    from forkan.datasets import load_uniform_pendulum
    data = load_uniform_pendulum()
    v = VAE(data.shape[1:], name='test', network='pendulum', beta=30.1, latent_dim=5, tensorboard=True)

    en_ints = [tf.placeholder(tf.float32, shape=(None, 64, 64, 1)) for _ in range(2)]
    mus = v.stack_encoder(en_ints)
    randi = np.random.normal(0, 1, (1, 64, 64, 1))

    combined_out = tf.concat(mus, axis=1)
    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        print(s.run(mus, feed_dict={en_ints[0]: randi,
                                    en_ints[1]: randi}))

    v._save()
    v._save('retrain')

    writer = tf.summary.FileWriter('/Users/llach/vae',
                                        graph=tf.get_default_graph())
    # v.train(data[:160], num_episodes=5, print_freq=20)


