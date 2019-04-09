import json
import logging

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from forkan.common.utils import print_dict, create_dir
from forkan.models.vae_networks import build_network


class RetrainVAE(object):

    def __init__(self, rlpath, input_shape, network='pendulum', latent_dim=20, beta=1.0, k=5,
                 init_from=None, with_attrs=False, sess=None):

        self.log = logging.getLogger('vae')

        self.input_shape = (None, ) + input_shape
        self.latent_dim = latent_dim
        self.with_attrs = with_attrs
        self.init_from = init_from
        self.network = network
        self.beta = beta
        self.k = k

        self.savepath = f'{rlpath}/vae/'
        create_dir(self.savepath)

        self.log.info('storing files under {}'.format(self.savepath))

        params = locals()
        params.pop('self')
        params.pop('sess')

        if not self.with_attrs:

            with open(f'{self.savepath}/params.json', 'w') as outfile:
                json.dump(params, outfile)
        else:
            self.log.info('load_base_weights() needs to be called!')

        with tf.variable_scope('vae', reuse=tf.AUTO_REUSE):
            self.X = tf.placeholder(tf.float32, shape=(None, k,) + self.input_shape[1:], name='stacked-vae-input')

        """ TF setup """
        self.s = sess
        assert self.s is not None, 'you need to pass a tf.Session()'

        """ TF Graph setup """

        self.mus = []
        self.logvars = []
        self.z = []
        self.Xhat = []

        for i in range(self.k):
            m, lv, z, xh = \
                build_network(self.X[:, i, ...], self.input_shape, latent_dim=self.latent_dim, network_type=self.network)
            self.mus.append(m)
            self.logvars.append(lv)
            self.z.append(z)
            self.Xhat.append(xh)
        print('\n')

        self.U = tf.concat(self.mus, axis=1)

        # Saver objects handles writing and reading protobuf weight files
        self.saver = tf.train.Saver(var_list=tf.trainable_variables(scope='vae'))

        if init_from:
            self._load_base_weights()

        """ Losses """
        # Loss
        # Reconstruction loss
        rels = []
        for i in range(self.k):
            rels.append(K.binary_crossentropy(K.flatten(self.X[:, i, ...]), K.flatten(self.Xhat[i])) * (self.input_shape[1] ** 2))
        self.re_loss = tf.reduce_mean(rels, axis=0)

        # define kullback leibler divergence
        kls = []
        for i in range(self.k):
            kls.append(-0.5 * K.mean((1 + self.logvars[i] - K.square(self.mus[i]) - K.exp(self.logvars[i])), axis=0))
        self.kl_loss = tf.reduce_mean(kls, axis=0)

        self.vae_loss = K.mean(self.re_loss + self.beta * K.sum(self.kl_loss))

        self.log.info('VAE has parameters:')
        print_dict(params, lo=self.log)

    def __del__(self):
        """ cleanup after object finalization """

        # close tf.Session
        if hasattr(self, 's'):
           self.s.close()

    def save(self, suffix='weights'):
        """ Saves current weights """
        self.log.info('saving weights')
        self.saver.save(self.s, f'{self.savepath}{suffix}')

    def load(self, suffix='weights'):
        """ Saves weights from suffix """
        self.saver.restore(self.s, f'{self.savepath}{suffix}')

    def _load_base_weights(self):
        if self.init_from is not None:
            from forkan import chosen_path
            from shutil import copyfile

            loadp = f'{chosen_path}{self.init_from}/'
            self.log.info(f'loading weights from {loadp} ...')
            self.saver.restore(self.s, f'{loadp}')
            self.log.info('done!')

            # save base weight instance
            self.save('base_weights')
            self.save()

            if self.with_attrs:
                self.log.info('using parameter from init model ...')
                with open(f'{loadp}/params.json', 'r') as infile:
                        params = json.load(infile)

                for k, v in params.items():
                    setattr(self, k, v)

                copyfile(f'{loadp}/params.json', f'{self.savepath}/params.json')

            else:
                copyfile(f'{loadp}/params.json', f'{self.savepath}/params_old.json')
                self.log.info('keeping new parameters')

        else:
            self.log.critical('trying to load weights but did not specify location. exiting.')
            exit(1)

    def _preprocess_batch(self, batch):
        """ preprocesses batch """

        assert np.max(batch) <= 1, 'normalise input first!'

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

    def encode(self, batch):
        """ encodes frame(s) """

        batch = self._preprocess_batch(batch)
        batch = np.expand_dims(batch, 1)
        return self.s.run([self.mus[0], self.logvars[0]], feed_dict={self.X: batch})

    def train(self, dataset, batch_size=155, num_episodes=50, print_freq=5):
        import numpy as np
        import time
        import datetime

        from tabulate import tabulate
        from forkan.common import CSVLogger
        from forkan.common.tf_utils import scalar_summary

        num_samples = len(dataset)

        assert np.max(dataset) <= 1, 'provide normalized dataset!'

        self.log.info('Training on {} samples for {} episodes.'.format(num_samples, num_episodes))
        tstart = time.time()
        nb = 1

        train_op = tf.train.AdamOptimizer().minimize(self.vae_loss)

        csv_header = ['date', '#episode', '#batch', 'rec-loss', 'kl-loss'] + \
                     ['z{}-kl'.format(i) for i in range(self.latent_dim)]
        csv = CSVLogger('{}/progress.csv'.format(self.savepath), *csv_header)

        rel_ph = tf.placeholder(tf.float32, (), name='rec-loss')
        kll_ph = tf.placeholder(tf.float32, (), name='rec-loss')
        klls_ph = [tf.placeholder(tf.float32, (), name=f'z{i}-kl') for i in range(self.latent_dim)]

        scalar_summary('reconstruction-loss', rel_ph, scope='vae-loss')
        scalar_summary('kl-loss', kll_ph, scope='vae-loss')
        for i in range(self.latent_dim):
            scalar_summary(f'z{i}-kl', klls_ph[i], scope='z-kl')
        merged_ = tf.summary.merge_all()
        writer = tf.summary.FileWriter(f'{self.savepath}/board', self.s.graph)

        self.s.run(tf.global_variables_initializer())

        # rollout N episodes
        for ep in range(num_episodes):

            # shuffle dataset
            np.random.shuffle(dataset)

            for n, idx in enumerate(np.arange(0, num_samples, batch_size)):
                bps = max(int(nb / (time.time() - tstart)), 1)
                x = dataset[idx:min(idx+batch_size, num_samples), ...]

                _, loss, re_loss, kl_losses = self.s.run([train_op, self.vae_loss, self.re_loss, self.kl_loss],
                                                             feed_dict={self.X: x})

                # mean losses
                re_loss = np.mean(re_loss)
                kl_loss = self.beta * np.sum(kl_losses)

                fd = {rel_ph: re_loss,
                      kll_ph: kl_loss,
                      }

                for i, kph in enumerate(klls_ph):
                    fd.update({kph: kl_losses[i]})

                suma = self.s.run(merged_, feed_dict=fd)

                writer.add_summary(suma, nb)

                # increase batch counter
                nb += 1

                csv.writeline(
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
                        ['name', f'retrainvae-clean-b{self.beta}'],
                        ['episode', ep],
                        ['batch', n],
                        ['bps', bps],
                        ['rec-loss', re_loss],
                        ['kl-loss', kl_loss],
                        ['ETA', '{}h {}min'.format(hrs, mins)],
                        ['done', '{}%'.format(int(perc))],
                    ])

                    print('\n{}'.format(tab))
            self.save()

        self.save()
        print('training done!')

if __name__ == '__main__':
    from forkan import model_path
    from forkan.datasets import load_uniform_pendulum

    data = load_uniform_pendulum().reshape(6000, 5, 64, 64, 1)

    v = RetrainVAE(f'{model_path}/retrain/', (64, 64, 1), network='pendulum', beta=84, latent_dim=5, sess=tf.Session())
    v.train(data, batch_size=128)