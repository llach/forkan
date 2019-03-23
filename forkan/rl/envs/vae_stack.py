import logging
import numpy as np

from collections import deque

from forkan.models import VAE

from gym import spaces
from forkan.rl import EnvWrapper


class VAEStack(EnvWrapper):

    def __init__(self,
                 env,
                 load_from,
                 k=3,
                 vae_network='pendulum',
                 **kwargs,
                 ):

        self.logger = logging.getLogger(__name__)

        # inheriting from EnvWrapper and passing it an env makes spaces available.
        super().__init__(env)

        self.k = k
        self.v = VAE(load_from=load_from, network=vae_network)
        self.observation_space = spaces.Box(low=-np.infty, high=np.infty, shape=(self.v.latent_dim*self.k,),
                                            dtype=np.float)
        self.vae_name = self.v.savename
        self.q = deque(maxlen=self.k)
        self._reset_queue()

    def _reset_queue(self):
        for _ in range(self.k):
            self.q.appendleft([0]*self.v.latent_dim)

    def _process(self, obs):
        mus, _, _ = self.v.encode(np.expand_dims(obs, 0))
        self.q.appendleft(np.squeeze(mus))

    def _get_obs(self):
        return np.asarray(self.q).flatten()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._process(obs)
        return self._get_obs(), reward, done, info

    def reset(self):
        self._reset_queue()
        obs = self.env.reset()
        self._process(obs)
        return self._get_obs()
