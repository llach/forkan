import gym
import logging
import numpy as np
from gym import spaces
from gym.utils import seeding
from collections import deque
import tensorflow as tf

from forkan.models import VAE
from baselines.common.tf_util import get_session


class VecVAEStack(gym.Env):

    def __init__(self,
                 env,
                 k=3,
                 load_from='pend-optimal',
                 vae_network='pendulum',
                 **kwargs,
                 ):

        self.logger = logging.getLogger(__name__)

        self.env = env
        self.num_envs = self.env.num_envs
        self.k = k

        self.sess = tf.Session(graph=tf.Graph())
        # self.sess = get_session() # loading a trained VAE into the RL session destroys everything. DONT DO IT

        self.v = VAE(load_from=load_from, network=vae_network, sess=self.sess)

        self.action_space = env.action_space
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.k*self.v.latent_dim, ), dtype=np.float32)

        self.vae_name = self.v.savename
        self.queues = [deque(maxlen=self.k) for _ in range(self.num_envs)]
        self._reset_queues()

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset_queues(self):
        for q in self.queues:
            for _ in range(self.k):
                q.appendleft([0]*self.v.latent_dim)

    def _process(self, obs):
        with self.sess.graph.as_default():
            mus, _, _ = self.v.encode(obs)
        for i in range(self.num_envs):
            self.queues[i].appendleft(mus[i].copy())

    def _get_obs(self):
        return np.asarray(self.queues).reshape([self.num_envs, -1])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._process(obs)
        return self._get_obs(), reward, done, info

    def reset(self):
        self._reset_queues()
        obs = self.env.reset()
        self._process(obs)
        return self._get_obs()
