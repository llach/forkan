import logging
import numpy as np

from collections import deque

from forkan.models import VAE

from gym import spaces
from forkan.rl import EnvWrapper


class VecVAEStack(EnvWrapper):

    def __init__(self,
                 env,
                 k=3,
                 load_from='pend-optimal',
                 vae_network='pendulum',
                 **kwargs,
                 ):

        self.logger = logging.getLogger(__name__)

        # inheriting from EnvWrapper and passing it an env makes spaces available.
        super().__init__(env)
        self.nenvs = self.num_envs

        self.k = k
        self.v = VAE(load_from=load_from, network=vae_network)
        self.observation_space = spaces.Box(low=-np.infty, high=np.infty, shape=(self.k*self.v.latent_dim,),
                                            dtype=np.float)

        self.queues = [deque(maxlen=self.k) for _ in range(self.nenvs)]
        self._reset_queues()

        self.logger.warning('THIS VERSION INCLUDES SOME HORRIBLE HACKS, SUCH AS A HARDCODED INDEX FOR THE LATENT THAT'
                            'REPRESENTS THETA. DON\'T USE FOR REAL RESULTS.')

    def _reset_queues(self):
        for q in self.queues:
            for _ in range(self.k):
                q.appendleft([0]*self.v.latent_dim)

    def _process(self, obs):
        _, _, zs = self.v.encode(obs)
        for i in range(self.nenvs):
            self.queues[i].appendleft(zs[i])

    def _get_obs(self):
        return np.asarray(self.queues).reshape([self.nenvs, -1])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._process(obs)
        return self._get_obs(), reward, done, info

    def reset(self):
        self._reset_queues()
        obs = self.env.reset()
        self._process(obs)
        return self._get_obs()
