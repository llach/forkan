import time
import numpy as np
import tensorflow as tf

import logging

from tqdm import tqdm
from tabulate import tabulate

from forkan.models.vae_networks import build_network
from forkan.common.tf_utils import scalar_summary, vector_summary

class VAE(object):

    def __init__(self, input_shape, network='atari', latent_dim=10, beta=1, lr=5e-4):

        # take care of correct input dim: (BATCH, HEIGHT, WIDTH, CHANNELS)
        # add channel dim if not provided
        if len(input_shape) == 2:
            input_shape = input_shape + (1,)

        # add batch dim
        self.input_shape = (None,) + input_shape

        self.latent_dim = latent_dim
        self.network = network
        self.beta = beta
        self.lr = lr

        self.num_channels = self.input_shape[-1]

        self.log = logging.getLogger(__name__)

        self._input = tf.placeholder(tf.float32, shape=self.input_shape, name='x')

        """ TF Graph setup """
        net = build_network(self._input, self.input_shape, latent_dim=self.latent_dim, network_type=network)
        (self.mus, self.logvars, self.z, self._output) = net

        """ Loss """
        self.reconstruction_loss = tf.losses.mean_squared_error(self._input, self._output)
        self.dkl_j = -0.5 * (1 + self.logvars - tf.square(self.mus) - tf.exp(self.logvars))
        self.mean_kl_j = tf.reduce_mean(self.dkl_j, axis=0)
        self.dkl_loss = tf.reduce_sum(self.mean_kl_j, axis=0)
        self.total_loss = self.reconstruction_loss + beta * self.dkl_loss

        # create optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        # specify loss function, only include Q network variables for gradient computation
        self.gradients = self.opt.compute_gradients(self.total_loss)

        # create training op
        self.train_op = self.opt.apply_gradients(self.gradients)

        """ TF setup """
        self.s = tf.Session()
        tf.global_variables_initializer().run(session=self.s)

        """ Tensorboard (TB) setup """

        self.fps_ph = tf.placeholder(tf.int32, ())

        scalar_summary('fps', self.fps_ph)

        mu_mean = tf.reduce_mean(self.mus, axis=0)
        vars_mean = tf.reduce_mean(tf.exp(0.5 * self.logvars), axis=0)

        with tf.variable_scope('loss'):
            scalar_summary('total-loss', self.total_loss)
            scalar_summary('mean-dkl', self.dkl_loss)
            scalar_summary('reconstruction-loss', self.reconstruction_loss)

        with tf.variable_scope('zj_kl'):
            for i in range(self.latent_dim):
                scalar_summary('z{}-kl'.format(i), self.mean_kl_j[i])

        with tf.variable_scope('zj_mu'):
            for i in range(self.latent_dim):
                scalar_summary('z{}-mu'.format(i), mu_mean[i])

        with tf.variable_scope('zj_var'):
            for i in range(self.latent_dim):
                scalar_summary('z{}-var'.format(i), vars_mean[i])

        # plot network weights
        with tf.variable_scope('weights'):
            for pv in tf.trainable_variables(): tf.summary.histogram('{}'.format(pv.name), pv)

        # gradient histograms
        with tf.variable_scope('gradients'):
            for g in self.gradients:
                if g[0] is not None:
                    tf.summary.histogram('{}-grad'.format(g[1].name), g[0])

        self.merge_op = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter('/tmp/vae',
                                            graph=tf.get_default_graph())

    def __del__(self):
        """ cleanup after object finalization """

        # close tf.Session
        if hasattr(self, 's'):
           self.s.close()
            
    def _preprocess_batch(self, batch):
        """ preprocesses batch """

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
        self.log.info('encoding batch with shape {}'.format(batch.shape))
        return self.s.run([self.mus, self.logvars], feed_dict={self._input: batch})

    def encode_and_sample(self, batch):
        """ encodes frame(s) and samples from dists """

        batch = self._preprocess_batch(batch)
        self.log.info('encoding and sampling zs for batch with shape {}'.format(batch.shape))
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

        # add dimensions if needed
        if len(dataset.shape) != 4:
            dataset = self._preprocess_batch(dataset)

        self.log.info('Training on {} samples for {} episodes.'.format(num_samples, num_episodes))
        tstart = time.time()
        t = 1

        # rollout N episodes
        for ep in tqdm(range(num_episodes)):

            # shuffle dataset
            np.random.shuffle(dataset)

            for n, idx in enumerate(np.arange(0, num_samples, batch_size)):
                fps = int((t) / (time.time() - tstart))
                x = dataset[idx:min(idx+batch_size, num_samples), ...]
                sum, _, loss, kl_loss = self.s.run([self.merge_op, self.train_op, self.total_loss, self.dkl_loss],
                                                   feed_dict={self._input: x, self.fps_ph: fps})

                t += abs(idx-min(idx+batch_size, num_samples))
                self.writer.add_summary(sum, t)

                if n % print_freq == 0:
                    tab = tabulate([
                        ['episode', ep],
                        ['batch', n],
                        ['fps', fps],
                        ['loss', loss],
                        ['dkl_loss', kl_loss]
                    ])

                    print('\n{}'.format(tab))


if __name__ == '__main__':
    from forkan.datasets.dsprites import load_dsprites
    (data, _) = load_dsprites('translation', repetitions=10)

    v = VAE(data.shape[1:], network='dsprites', beta=5.1)

    # data = np.random.underniform(0, 1, [10000, 84, 84])
    print(v.train(data, num_episodes=1, print_freq=1))


