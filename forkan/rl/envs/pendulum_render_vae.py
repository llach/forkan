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
        self.max_speed = 8
        high = np.array([1., 1., self.max_speed])

        self.v = VAE(load_from='pend-optimal', network='pendulum')
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float)

        self.logger.warning('THIS VERSION INCLUDES SOME HORRIBLE HACKS, SUCH AS A HARDCODED INDEX FOR THE LATENT THAT'
                            'REPRESENTS THETA. DON\' USE FOR REAL RESULTS.' )

        self.old_z = 0

    def _process(self, obs):

        zs = self.v.encode(obs)[-1]
        z = zs[0][2]
        thed = np.clip((z - self.old_z) / 0.05, -self.max_speed, self.max_speed)
        # con = np.concatenate([zs, thed], axis=-1)
        self.old_z = z

        return np.asarray([np.sin(z), np.cos(z), thed])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.old_z = 0
        return self._process(obs)
