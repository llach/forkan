import logging
import tensorflow as tf

from gym import Space
from forkan.rl import EnvWrapper
from forkan.models import VAE


"""
TODO

- normalize
- batch-pass that mofo to vae
- pass session to restore vae graph in
"""


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

        # separate session for VAE graph
        self.vae_s = tf.Session()

        self.vae = VAE(load_from=load_from, sess=self.vae_s)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()
