import json
import logging

import tensorflow as tf
import tensorflow.keras.backend as K

from forkan.common.utils import print_dict, create_dir
from forkan.models.vae_networks import build_network


class RetrainVAE(object):

    def __init__(self, rlpath, input_shape, network='pendulum', latent_dim=20, beta=5.5, k=5,
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
