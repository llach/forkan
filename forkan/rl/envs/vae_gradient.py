import gym
import logging
import numpy as np
from gym import spaces
from gym.utils import seeding
from collections import deque

from forkan.models import VAE


class VAEGradient(gym.Env):

    def __init__(self,
                 env,
                 nsteps,
                 k=3,
                 load_from='pend-optimal',
                 vae_network='pendulum',
                 **kwargs,
                 ):

        self.logger = logging.getLogger(__name__)

        self.env = env
        self.nsteps = nsteps
        self.num_envs = self.env.num_envs
        self.k = k
        self.nbatch = self.nsteps * self.num_envs

        self.v = VAE(load_from=load_from, network=vae_network)

        self.action_space = env.action_space
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.k*self.v.latent_dim, ), dtype=np.float32)

        self.vae_name = self.v.savename
        self.queues = [deque(maxlen=self.k) for _ in range(self.num_envs)]
        self._reset_queues()

        self.obs_buffer = np.zeros((self.num_envs, self.nsteps, self.k, )+self.env.observation_space.shape)
        # step index
        self.sidx = 0

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset_queues(self):
        for q in self.queues:
            for _ in range(self.k):
                q.appendleft([0]*self.v.latent_dim)

    def _process(self, obs):
        mus, _ = self.v.encode(obs)
        for i in range(self.num_envs):
            self.queues[i].appendleft(mus[i].copy())

    def _get_obs(self):
        return np.asarray(self.queues).reshape([self.num_envs, -1])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        for ne in range(self.num_envs):
            for kp in range(self.k):

                step_idx = self.sidx + kp
                if step_idx >= self.nsteps: continue
                self.obs_buffer[ne, step_idx, kp] = obs[ne]
        self.sidx += 1
        self._process(obs)
        return self._get_obs(), reward, done, info

    def apply_gradients_to_vae(self, grads):

        # we reshape obs buffer and gradients to batch of rollouts where every rollout step
        # has shape (LOOKBACK, OBS_SHAPE)
        s = self.obs_buffer.shape
        obs_batch = self.obs_buffer.reshape(s[0]*s[1], *s[2:])
        grads = grads.reshape(self.nbatch, self.k, self.v.latent_dim)



        """ This plots the observations along with gradients (or obs) passed. Debugging only. """
        # import matplotlib.pyplot as plt
        # for li in range(20):
        #     fig, axarr = plt.subplots(2, self.k)
        #     for ki in range(self.k):
        #
        #         axarr[0, ki].imshow(np.squeeze(obs_batch[li, ki]), cmap='Greys_r')
        #         axarr[0, ki].set_title(f'li {li} ki {ki}')
        #
        #         print(grads[li, ki])
        #         axarr[1, ki].bar(np.arange(self.v.latent_dim), grads[li, ki])
        #
        #     plt.show()
        #
        # print(obs_batch.shape)
        # exit(0)

        # we reset the step index and our observation buffer
        self.sidx = 0
        self.obs_buffer = np.zeros((self.num_envs, self.nsteps, self.k,) + self.env.observation_space.shape)

    def reset(self):
        self._reset_queues()
        obs = self.env.reset()
        self._process(obs)
        return self._get_obs()
