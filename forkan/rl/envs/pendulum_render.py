import logging
import numpy as np
from skimage.transform import resize

from gym import spaces
from forkan.rl import EnvWrapper


class PendulumRenderEnv(EnvWrapper):

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

        self.num_envs = 1

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float)

    def _process(self, obs):
        obs = obs / 255  # normalise
        obs = obs[121:256 + 121, 121:256 + 121, :]  # cut out interesting area
        obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])  # inverted greyscale
        obs = resize(obs, (64, 64), mode='reflect', anti_aliasing=True)
        return np.expand_dims(np.expand_dims(obs, -1), 0)

    def step(self, action):
        _, reward, done, info = self.env.step(action)

        if done:
            obs = self.reset()
            return obs, reward, [done], info
        else:
            obs = self.env.render(mode='rgb_array')
            return self._process(obs), reward, [done], info

    def reset(self):
        _ = self.env.reset()
        obs = self.env.render(mode='rgb_array')
        return self._process(obs)
