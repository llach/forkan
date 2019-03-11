import logging
import numpy as np

from collections import deque

from forkan.models import VAE

from gym import spaces
from forkan.rl import EnvWrapper


class PendulumVAEStackEnv(EnvWrapper):

    def __init__(self,
                 env,
                 k=4,
                 **kwargs,
                 ):

        self.logger = logging.getLogger(__name__)

        # inheriting from EnvWrapper and passing it an env makes spaces available.
        super().__init__(env)

        self.k = k
        self.v = VAE(load_from='pend-optimal', network='pendulum')
        self.observation_space = spaces.Box(low=-np.infty, high=np.infty, shape=(self.k,), dtype=np.float)

        self.logger.warning('THIS VERSION INCLUDES SOME HORRIBLE HACKS, SUCH AS A HARDCODED INDEX FOR THE LATENT THAT'
                            'REPRESENTS THETA. DON\'T USE FOR REAL RESULTS.')

        self.buf = deque(maxlen=self.k)
        self.reset()

    def _reset_buffer(self):
        for _ in range(self.k):
            self.buf.append(0)

    def _process(self, obs):
        zs = self.v.encode(obs)[-1]
        self.buf.append(zs[0][2])
        return np.asarray(self.buf.copy(), dtype=np.float)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process(obs), reward, done, info

    def reset(self):
        self._reset_buffer()
        obs = self.env.reset()
        return self._process(obs)
