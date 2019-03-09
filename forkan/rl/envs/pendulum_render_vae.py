import logging
import numpy as np

from forkan.models import VAE

from gym import spaces
from forkan.rl import EnvWrapper


class PendulumRenderVAEEnv(EnvWrapper):

    def __init__(self,
                 env,
                 **kwargs,
                 ):
        """
        Wraps Pendulum environment, renders it and returns greyscaled and cropped images as obs.


        WARNING this also emulates a VecEnv for A2C with one environment

        env: gym.Environment
            FrameStackEnv that buffers LazyFrames

        load_from: string
            argument passed to VAE: location of weights
        """

        self.logger = logging.getLogger(__name__)

        # inheriting from EnvWrapper and passing it an env makes spaces available.
        super().__init__(env)

        self.v = VAE(load_from='pend-optimal', network='pendulum')
        self.observation_space = spaces.Box(low=-np.infty, high=np.infty, shape=(self.v.latent_dim*2,), dtype=np.float)

        self.old_z = np.zeros([self.v.latent_dim])

    def _process(self, obs):

        zs = self.v.encode(obs)[-1]
        thed = (zs - self.old_z) / 0.05
        con = np.concatenate([zs, thed], axis=-1)
        self.old_z = zs.copy()

        return con

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._process(obs)
