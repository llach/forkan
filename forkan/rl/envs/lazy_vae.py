import logging
import numpy as np
import tensorflow as tf

from gym import spaces
from forkan.rl import EnvWrapper
from forkan.models import VAE


class LazyVAE(EnvWrapper):

    def __init__(self,
                 env,
                 load_from,
                 **kwargs,
                 ):
        """
        Wraps FrameStack Atari Env and processes frames in a VAE.

        env: gym.Environment
            FrameStackEnv that buffers LazyFrames

        load_from: string
            argument passed to VAE: location of weights
        """

        self.logger = logging.getLogger(__name__)

        # inheriting from EnvWrapper and passing it an env makes spaces available.
        super().__init__(env)

        # separate session for VAE graph & create VAE
        self.vae_s = tf.Session()
        self.vae = VAE(load_from=load_from, sess=self.vae_s)

        bs = self.observation_space.shape[-1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(bs*2*self.vae.latent_dim,))

    def _process(self, obs):
        # normalize obs
        obs = np.asarray(obs) / 255

        # channels -> batch
        obs = np.expand_dims(np.moveaxis(obs, -1, 0), -1)

        return self.vae.encode(obs).flatten()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process(obs), reward, done, info

    def reset(self):
        return self._process(self.env.reset())
